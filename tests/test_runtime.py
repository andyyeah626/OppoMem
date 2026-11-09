from dataclasses import replace
import json

import pytest

from oppomem.agent import AgentVariant
from oppomem.config import ExperimentConfig, MemoryConfig
from oppomem.factory import build_agent
from oppomem.games.fixed_rps import FixedRPSAdapter
from oppomem.games.isolated import IsolatedGameAdapter
from oppomem.runtime.episode import execute_episode
from oppomem.runtime.experiment import run_experiment
from oppomem.runtime.offline import OfflineEncoder, OfflineLLM, OfflineNLI
from oppomem.runtime.storage import read, run_lock, write


def run(path, config=None, factory=None):
    config = config or ExperimentConfig()
    return run_experiment(config, path, llm_factory=factory or (lambda: OfflineLLM(config)),
                          embedder=OfflineEncoder(), nli=OfflineNLI())


def test_complete_lifecycle_and_resume_without_model_calls(tmp_path):
    result = run(tmp_path)
    assert result["mode"] == "offline_smoke"
    assert result["train_episodes"] == 4 and result["test_episodes"] == 2
    assert result["final_clusters"] > 0 and result["segments"] > 0
    assert 0 < result["train_injections"] < result["train_retrievals"]
    assert result["test_retrievals"] > 0
    bank = tmp_path / result["bank"]
    original = bank.read_bytes()
    class NoCalls:
        def complete_json(self, **kwargs):
            pytest.fail("A completed run made another model call")
    assert run(tmp_path, factory=NoCalls) == result
    assert bank.read_bytes() == original
    for path in (tmp_path / "evaluation").glob("*.json"):
        record = read(path)["record"]
        assert all(use["selected"] for use in record["trace"]["memory_uses"])
        assert "centroid_memory" not in record["trace"]["learning_audit"]


def test_resume_after_update_failure_reuses_completed_games(tmp_path):
    config = ExperimentConfig()
    class Failure(OfflineLLM):
        def complete_json(self, **kwargs):
            if kwargs["stage"] == "memory_reflection":
                raise RuntimeError("interrupted reflection")
            return super().complete_json(**kwargs)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(tmp_path, config, lambda: Failure(config))
    assert read(tmp_path / "status.json")["state"] == "failed"
    saved = {p: p.read_bytes() for p in (tmp_path / "training/batch_000/episodes").glob("*.json")}
    assert len(saved) == 2
    assert not (tmp_path / "training/batch_000/complete.json").exists()
    result = run(tmp_path)
    assert result["batches"] == 2
    assert all(p.read_bytes() == content for p, content in saved.items())


def test_parallel_evaluation_preserves_order_bank_and_resume(tmp_path, monkeypatch):
    import threading
    from oppomem.runtime import experiment

    config = replace(ExperimentConfig(), test_episodes=6)
    serial = run(tmp_path / "serial", config)
    concurrent = replace(config, evaluation_workers=3)
    controller = threading.get_ident()
    clients = []
    overlap = threading.Barrier(3)
    class Model(OfflineLLM):
        def __init__(self):
            assert threading.get_ident() == controller
            super().__init__(concurrent)
            self.owner = None
            self.is_evaluation = False
            clients.append(self)
        def complete_json(self, **kwargs):
            current = threading.get_ident()
            if self.is_evaluation:
                if self.owner is None:
                    self.owner = current
                    overlap.wait(timeout=5)
                assert self.owner == current, "One evaluation client was shared across threads"
            return super().complete_json(**kwargs)

    original_scoped = experiment.scoped

    def mark_episode_scope(client, scope):
        parts = scope.split("/")
        if isinstance(client, Model) and parts[0] == "episode":
            client.is_evaluation = int(parts[1]) >= config.batches * config.episodes_per_batch
        return original_scoped(client, scope)

    monkeypatch.setattr(experiment, "scoped", mark_episode_scope)
    parallel = run(tmp_path / "parallel", concurrent, Model)
    assert parallel == serial
    assert parallel["test_seeds"] == list(range(config.seed + 4, config.seed + 10))
    assert len({id(client) for client in clients[-6:]}) == 6
    assert len({client.owner for client in clients[-6:]}) == 3
    for name in ("serial", "parallel"):
        assert read(tmp_path / name / parallel["bank"]) == read(tmp_path / "serial" / serial["bank"])
    class NoCalls:
        def complete_json(self, **kwargs):
            pytest.fail("A completed parallel run made another model call")
    assert run(tmp_path / "parallel", concurrent, NoCalls) == parallel


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_invalid_evaluation_workers_fail_before_running(workers):
    with pytest.raises(ValueError, match="evaluation_workers"):
        ExperimentConfig(evaluation_workers=workers)


def test_changed_config_and_corrupt_cache_are_rejected(tmp_path):
    result = run(tmp_path)
    with pytest.raises(ValueError, match="configuration or source"):
        run(tmp_path, replace(ExperimentConfig(), horizon=7))
    bank = tmp_path / result["bank"]
    data = read(bank)
    data["clusters"][0]["response_strategy"] = "Tampered advice"
    write(bank, data)
    with pytest.raises(ValueError, match="artifacts changed"):
        run(tmp_path)


def test_changed_episode_is_rejected(tmp_path):
    run(tmp_path)
    path = next((tmp_path / "training/batch_000/episodes").glob("*.json"))
    cell = read(path)
    cell["record"]["return"] += 1
    write(path, cell)
    with pytest.raises(ValueError, match="Invalid cached episode"):
        run(tmp_path)


def test_eval_disables_learning_and_cleans_up_on_failure(tmp_path):
    config = ExperimentConfig()
    agent = build_agent(variant=AgentVariant.TOM_MEMORY, llm=OfflineLLM(config),
        memory_path=tmp_path / "bank.json", embedder=OfflineEncoder(), nli=OfflineNLI(),
        semantic_config=config.to_dict()["memory"], game=config.game_id)
    class Adapter(FixedRPSAdapter):
        closed = False
        def close(self):
            self.closed = True
    adapter = Adapter(horizon=6)
    execute_episode(adapter=adapter, agent=agent, opponent_id="rockbot", seed=1)
    assert not agent.bank.clusters
    assert agent.learning_enabled and agent.trace is None and adapter.closed
    class Failure(OfflineLLM):
        def complete_json(self, **kwargs):
            raise RuntimeError("model failed")
    agent.llm = Failure(config)
    adapter = Adapter(horizon=6)
    with pytest.raises(RuntimeError, match="model failed"):
        execute_episode(adapter=adapter, agent=agent, opponent_id="rockbot", seed=1, training=True)
    assert agent.learning_enabled and agent.trace is None and adapter.closed


def test_native_engine_boundary_uses_a_disposable_process():
    adapter = IsolatedGameAdapter("fixed_rps", horizon=2)
    try:
        frame = adapter.reset(opponent_id="rockbot", seed=1, seat=0, training=False)
        assert not frame.terminal
        assert adapter.game_id == "open_spiel_roshambo"
        assert adapter.fingerprint()["adapter"] == "FixedRPSAdapter"
        process = adapter._process
        assert process.is_alive()
        assert adapter.step("Paper").reward_delta == 1
        assert adapter.step("Paper").terminal
    finally:
        adapter.close()
    assert adapter._process is None


def test_run_lock_excludes_second_controller_and_releases(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(OSError):
            with run_lock(tmp_path):
                pytest.fail("Two controllers acquired the same run")
    with run_lock(tmp_path):
        pass


@pytest.mark.parametrize("kwargs", [{"retrieval_top_k": 0}, {"merge_similarity_threshold": float("nan")},
                                    {"retrieval_min_entailment": 2}, {"retrieval_backend": "typo"}])
def test_invalid_memory_settings_fail_before_model_calls(kwargs):
    with pytest.raises(ValueError):
        MemoryConfig(**kwargs)


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"memory": {"retreival_top_k": 3}}), encoding="utf-8")
    with pytest.raises(TypeError):
        ExperimentConfig.load(path)


def test_frozen_profile_reuses_bank_and_warmup_without_training(tmp_path):
    import hashlib
    from oppomem.runtime.storage import digest
    from oppomem.config import ModelConfig
    profile = tmp_path / "profile"
    profile.mkdir()
    seed_agent = build_agent(variant=AgentVariant.TOM_MEMORY, llm=OfflineLLM(ExperimentConfig()),
        memory_path=profile / "bank.json", embedder=OfflineEncoder(), nli=OfflineNLI(),
        semantic_config={**ExperimentConfig().to_dict()["memory"], "retrieval_backend": "cosine"},
        game="open_spiel_roshambo")
    seed_agent.bank.save()
    original = (profile / "bank.json").read_bytes()
    write(profile / "winner.json", {"threshold": .74, "tom_start_round": 5,
        "bank_sha256": hashlib.sha256(original).hexdigest()})
    write(profile / "prompt_snapshot.json", {"tom_system": "historic system", "tom_task": "historic task"})
    calls = []
    class Model:
        def complete_json(self, **kwargs):
            assert kwargs['stage'] == 'decision'
            return {"action": "Paper"}
        def complete_text(self, **kwargs):
            calls.append(kwargs)
            return "The opponent repeatedly plays Rock."
    config = replace(ExperimentConfig(), frozen_profile=str(profile), test_episodes=1,
        memory=replace(MemoryConfig(), retrieval_backend="cosine", retrieval_min_similarity=.74))
    result = run_experiment(config, tmp_path / "run", llm_factory=Model,
        embedder=OfflineEncoder(), nli=OfflineNLI())
    assert result['train_episodes'] == 0 and result['batches'] == 0
    assert len(calls) == 1 and calls[0]['system'] == 'historic system'
    assert calls[0]['prompt'].endswith('historic task')
    assert (profile / 'bank.json').read_bytes() == original
    assert read(tmp_path / 'run' / result['bank'])['clusters'] == read(profile / 'bank.json')['clusters']
    with pytest.raises(ValueError, match='original cosine threshold'):
        run_experiment(replace(config, memory=replace(config.memory, retrieval_min_similarity=.8)),
            tmp_path / 'wrong', llm_factory=Model, embedder=OfflineEncoder(), nli=OfflineNLI())
