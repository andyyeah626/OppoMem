"""Guard the live transport and game boundaries without making paid requests."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from oppomem.agent import AgentVariant, ToMMemoryAgent
from oppomem.config import ExperimentConfig, ModelConfig
from oppomem.games.llm_hanabi import LLMHanabiPartner
from oppomem.models.cache import CachedLLM
from oppomem.models.openai import DeepSeekJsonLLM
from oppomem.runtime.storage import read, write


def test_hanabi_first_decision_uses_visible_state_and_legal_actions():
    calls = []
    class Model:
        def complete_json(self, **kwargs):
            calls.append(kwargs)
            return {"action": "7"}
    agent = ToMMemoryAgent(game="adaptive_hanabi", variant=AgentVariant.VANILLA, llm=Model())
    agent.begin_episode()
    decision = agent.decide(observation="HANABI_VISIBLE_STATE", legal_actions=["7", "9"])
    assert decision.decision.action == "7"
    assert len(calls) == 1
    assert "HANABI_VISIBLE_STATE" in calls[0]["prompt"]
    assert '"7"' in calls[0]["prompt"] and '"9"' in calls[0]["prompt"]
    assert "Prisoner's Dilemma" not in calls[0]["prompt"]


def test_deepseek_none_reaches_transport_and_journal_directory_is_created(tmp_path, monkeypatch):
    import sys
    payloads = []
    def create(**kwargs):
        payloads.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            message=SimpleNamespace(content='{"action": "7"}'))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kw:
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))))
    path = tmp_path / "missing" / "client.jsonl"
    client = DeepSeekJsonLLM(api_key="test-only", model="deepseek-flash",
        reasoning_effort="none", request_journal=path)
    assert client.complete_json(stage="test", system="Return JSON.", prompt="Choose.") == {"action": "7"}
    assert payloads[0]["reasoning_effort"] == "none"
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert path.is_file() and "test-only" not in path.read_text()


def test_model_cache_is_scoped_persistent_and_rejects_tampering(tmp_path):
    class Model:
        model, reasoning_effort, calls = "fake", "none", 0
        def complete_json(self, **kwargs):
            self.calls += 1
            return {"call": self.calls}
    model = Model()
    first = CachedLLM(model, tmp_path, "episode/0")
    assert first.complete_json(prompt="same") == {"call": 1}
    assert CachedLLM(model, tmp_path, "episode/0").complete_json(prompt="same") == {"call": 1}
    assert first.scoped("episode/1").complete_json(prompt="same") == {"call": 2}
    assert model.calls == 2
    path = next(p for p in tmp_path.glob("*.json") if read(p)["request"]["scope"] == "episode/0")
    saved = read(path)
    saved["response"]["call"] = 9
    write(path, saved)
    with pytest.raises(ValueError, match="cache changed"):
        first.complete_json(prompt="same")


def test_invalid_api_request_fails_once_and_records_safe_diagnostics(tmp_path, monkeypatch):
    import sys
    class Rejected(Exception):
        status_code = 401
    def create(**kwargs):
        raise Rejected("Credential test-only must never enter the journal")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kw:
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))))
    path = tmp_path / "client.jsonl"
    client = DeepSeekJsonLLM(api_key="test-only", request_journal=path)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        client.complete_json(stage="test", system="JSON", prompt="test")
    assert client.requests_made == 1
    assert client.usage_dicts()[0]["attempts"] == 1
    assert '"http_status": 401' in path.read_text()
    assert "test-only" not in path.read_text()


def test_read_only_model_cache_reuses_response_without_calls_or_external_writes(tmp_path):
    class Model:
        model, reasoning_effort = "fake", "none"
        def complete_json(self, **kwargs):
            return {"action": "7"}
    source, local = tmp_path / "source", tmp_path / "new-run"
    CachedLLM(Model(), source, "episode/0").complete_json(prompt="same")
    source_file = next(source.glob("*.json"))
    original = source_file.read_bytes()
    class NoCalls(Model):
        def complete_json(self, **kwargs):
            pytest.fail("A validated previous response should be reused")
    reader = CachedLLM(NoCalls(), local, read_only_cache=source).scoped("episode/0")
    assert reader.complete_json(prompt="same") == {"action": "7"}
    assert source_file.read_bytes() == original
    copied = read(local / source_file.name)
    assert copied["origin"] == str(source_file.resolve())
    # The local copy has priority and supports independent subsequent resumes.
    source_file.unlink()
    assert reader.complete_json(prompt="same") == {"action": "7"}


@pytest.mark.parametrize("changed", ["scope", "effort"])
def test_read_only_cache_keeps_different_experiments_independent(tmp_path, changed):
    class Model:
        model, reasoning_effort, calls = "fake", "none", 0
        def complete_json(self, **kwargs):
            self.calls += 1
            return {"call": self.calls}
    source = tmp_path / "source"
    CachedLLM(Model(), source, "episode/0").complete_json(prompt="same")
    original = {p.name: p.read_bytes() for p in source.glob("*.json")}
    client = Model()
    scope = "episode/1" if changed == "scope" else "episode/0"
    if changed == "effort":
        client.reasoning_effort = "low"
    CachedLLM(client, tmp_path / "new-run", scope, read_only_cache=source).complete_json(prompt="same")
    assert client.calls == 1
    assert {p.name: p.read_bytes() for p in source.glob("*.json")} == original


@pytest.mark.parametrize("changed", ["request", "response"])
def test_corrupt_read_only_cache_is_rejected_before_copy_or_model_call(tmp_path, changed):
    class Model:
        model, reasoning_effort, calls = "fake", "none", 0
        def complete_json(self, **kwargs):
            self.calls += 1
            return {"action": "7"}
    source, local = tmp_path / "source", tmp_path / "new-run"
    CachedLLM(Model(), source).complete_json(prompt="same")
    path = next(source.glob("*.json"))
    saved = read(path)
    if changed == "request":
        saved["request"]["kwargs"]["prompt"] = "different"
    else:
        saved["response"]["action"] = "9"
    write(path, saved)
    original = path.read_bytes()
    client = Model()
    with pytest.raises(ValueError, match="cache changed"):
        CachedLLM(client, local, read_only_cache=source).complete_json(prompt="same")
    assert client.calls == 0 and not local.exists()
    assert path.read_bytes() == original


def test_read_only_cache_cannot_be_used_as_the_writable_directory(tmp_path):
    with pytest.raises(ValueError, match="must differ"):
        CachedLLM(object(), tmp_path, read_only_cache=tmp_path)


@pytest.mark.parametrize("actions,expected", [([99, "3"], 3), ([True, 3], 3), ([99, 99, 99], None)])
def test_hanabi_partner_only_returns_legal_actions(actions, expected):
    class Model:
        def complete_json(self, **kwargs):
            assert "OWN_VISIBLE" in kwargs["prompt"]
            return {"action": actions.pop(0)}
    partner = LLMHanabiPartner(Model())
    if expected is None:
        with pytest.raises(ValueError, match="legal action"):
            partner.act("OWN_VISIBLE", (3,), {3: "Play 3"})
    else:
        assert partner.act("OWN_VISIBLE", (3,), {3: "Play 3"}) == expected


@pytest.fixture
def deliberation_game(tmp_path):
    from oppomem.deliberation.protocol import Game
    root = tmp_path / "assets"
    (root / "scores_files").mkdir(parents=True)
    (root / "config.txt").write_text("Focal,focal,p1,cooperative,ignored\nOther,other,p2,cooperative,ignored\nThird,third,p3,cooperative,ignored\n")
    for name in ("focal", "other", "third"):
        (root / "scores_files" / f"{name}.txt").write_text("20,30\n10\n")
    (root / "initial_deal.txt").write_text("A1")
    (root / "global_instructions.txt").write_text("Negotiate a deal.")
    (root / "initial_prompts.py").write_text(
        "class InitialPrompt:\n"
        "    def __init__(self, path, name, *args, **kwargs): self.name = name\n"
        "    def return_initial_prompt(self): return self.name\n")
    return Game(root, prompt_source=root)


def test_deliberation_lifecycle_private_boundaries_and_no_call_resume(tmp_path, deliberation_game):
    from oppomem.deliberation.pipeline import run_experiment
    game = deliberation_game
    calls = []
    class Encoder:
        def encode(self, texts):
            return [[1.0, 0.0] for _ in texts]
    class Model:
        def complete_text(self, **kwargs):
            calls.append(kwargs)
            # A participant may see their own plan, never somebody else's.
            name = kwargs["system"].splitlines()[0]
            for other in game.agent_names:
                if other != name:
                    assert f"PRIVATE_{other}" not in kwargs["prompt"]
            return f"<SCRATCHPAD>PRIVATE_REASONING</SCRATCHPAD><ANSWER><DEAL>A1</DEAL></ANSWER><PLAN>PRIVATE_{name}</PLAN>"
        def complete_json(self, **kwargs):
            calls.append(kwargs)
            assert "PRIVATE_" not in kwargs["prompt"]
            if kwargs["stage"] == "deliberation_tom":
                return {"opponent_strategy": "The participant negotiates conditional trades."}
            return {"entries": [{"cluster_id": "delib_conditional",
                "opponent_strategy": "The participant negotiates conditional trades.",
                "response_strategy": "Offer a conditional exchange after checking visible support."}]}
    config = ExperimentConfig(game="deliberation", horizon=3, episodes_per_batch=1, test_episodes=1,
        game_dir=str(game.game_dir), prompt_source=str(game.prompt_source),
        models=ModelConfig(provider="deepseek", model="fake"))
    output = tmp_path / "run"
    result = run_experiment(config, output, llm_factory=Model, embedder=Encoder())
    assert result["train_episodes"] == 2 and result["test_episodes"] == 1
    assert result["final_clusters"] == 1 and result["test_retrievals"] > 0
    assert result["test_valid_finals"] == 1
    assert result["test_successful_finals"] == 1
    assert result["test_seeds"] == [config.seed + 2]
    assert result["test_returns"] == [result["test_mean_return"]]
    assert result["test_return_std"] == 0
    count = len(calls)
    assert run_experiment(config, output, llm_factory=Model, embedder=Encoder()) == result
    assert len(calls) == count
    with pytest.raises(ValueError, match="configuration"):
        run_experiment(replace(config, horizon=4), output, llm_factory=Model, embedder=Encoder())
    bank_path = output / result["bank"]
    bank = read(bank_path)
    bank["clusters"][0]["response_strategy"] = "Changed after evaluation"
    write(bank_path, bank)
    with pytest.raises(ValueError, match="artifacts changed"):
        run_experiment(config, output, llm_factory=Model, embedder=Encoder())
