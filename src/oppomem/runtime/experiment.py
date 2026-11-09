"""One frozen-bank train/update/evaluate pipeline with resumable artifacts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
from pathlib import Path
from statistics import pstdev

from oppomem.agent import AgentVariant
from oppomem.config import ExperimentConfig
from oppomem.factory import build_agent
from oppomem.memory import CentroidMemoryBank
from oppomem.memory.evolution import (
    LLMPool, PooledAssigner, build_evidence, commit, reflect_all, segment_batch,
)
from oppomem.models.semantics import RetrievalEmbedder
from oppomem.models.cache import scoped
from .episode import execute_episode
from .storage import digest, read, run_lock, write


def source_digest():
    root = Path(__file__).resolve().parents[1]
    return digest({p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(root.rglob("*.py"))})


def make_adapter(config, partner_llm=None):
    if config.game == "fixed_rps":
        from oppomem.games.fixed_rps import FixedRPSAdapter
        return FixedRPSAdapter(horizon=config.horizon)
    if config.game == "rps":
        from oppomem.games.isolated import IsolatedGameAdapter
        return IsolatedGameAdapter("rps", horizon=config.horizon)
    if config.game == "ipd":
        from oppomem.games.ipd import AxelrodAdapter
        return AxelrodAdapter(horizon=config.horizon)
    if config.partner == "llm":
        from oppomem.games.hanabi import HanabiAdapter
        from oppomem.games.llm_hanabi import LLMHanabiPartner
        return HanabiAdapter(LLMHanabiPartner(partner_llm))
    from oppomem.games.isolated import IsolatedGameAdapter
    return IsolatedGameAdapter("hanabi", horizon=config.horizon)


def _bank_digest(bank):
    return digest({"metadata": bank.metadata,
                   "clusters": [c.to_dict() for c in bank.clusters.values()]})


def run_experiment(config: ExperimentConfig, output_dir: str | Path, *,
                   llm_factory, embedder, nli, adapter_factory=None, progress=None):
    """Inject model/game dependencies; no credentials or machine paths live here.

    A completed cell or batch is reused only with matching code/configuration and
    bank hashes. An interrupted episode is replayed; paid calls within that episode
    are not guaranteed exactly once. External dependencies are owned by the caller.
    """
    output = Path(output_dir)
    with run_lock(output):
        manifest = {"schema_version": 1, "config": config.to_dict(),
                    "source_sha256": source_digest(),
                    "mode": "offline_smoke" if config.models.provider == "offline" else "live"}
        if config.frozen_profile:
            profile = Path(config.frozen_profile)
            manifest["frozen_profile"] = {name: hashlib.sha256((profile / name).read_bytes()).hexdigest()
                for name in ("bank.json", "winner.json", "prompt_snapshot.json")}
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if read(manifest_path) != manifest:
                raise ValueError("Run configuration or source changed; choose a new output directory")
        else:
            if any(p.name != "run.lock" for p in output.iterdir()):
                raise ValueError("Output directory contains artifacts without a run manifest")
            write(manifest_path, manifest)
        try:
            return _run(config, output, manifest, llm_factory, embedder, nli,
                        adapter_factory, progress)
        except Exception as exc:
            write(output / "status.json", {"state": "failed", "error_type": type(exc).__name__})
            raise


def _run(config, output, manifest, llm_factory, embedder, nli, adapter_factory, progress):
    semantics = asdict(config.memory)
    semantics["tom_nl_horizon"] = config.horizon
    # The online factory and offline updater must use the identical embedding
    # convention; otherwise E5 prefixes differ between construction and retrieval.
    if config.memory.embedding_mode != "none" and not isinstance(embedder, RetrievalEmbedder):
        embedder = RetrievalEmbedder(embedder, config.memory.embedding_mode)
    llm = llm_factory()
    profile = Path(config.frozen_profile) if config.frozen_profile else None
    winner = read(profile / "winner.json") if profile else None
    if winner and (winner["threshold"] != config.memory.retrieval_min_similarity
                   or config.memory.retrieval_backend != "cosine"):
        raise ValueError("Frozen profile requires its original cosine threshold")

    def agent_for(path):
        agent = build_agent(variant=AgentVariant.TOM_MEMORY, llm=llm,
                            memory_path=path, embedder=embedder, nli=nli,
                            semantic_config=semantics, load_existing_memory=path.exists(),
                            game=config.game_id)
        agent.learning_enabled = False
        if profile:
            agent.tom_start_round = winner["tom_start_round"]
            agent.tom_snapshot = read(profile / "prompt_snapshot.json")
        return agent

    bank_path = output / "banks" / "initial.json"
    if not bank_path.exists():
        if profile:
            original = profile / "bank.json"
            if hashlib.sha256(original.read_bytes()).hexdigest() != winner["bank_sha256"]:
                raise ValueError("Frozen source bank hash changed")
            bank = read(original)
            bank["metadata"].update(scope=config.memory.memory_scope,
                protocol_id=config.memory.protocol_id, tom_prompt_variant=config.memory.tom_prompt_variant,
                embedding_mode=config.memory.embedding_mode)
            write(bank_path, bank)
        else:
            agent_for(bank_path).bank.save()
    all_training = []
    updates = []

    def play(path, bank_path, index, *, training, client=None):
        client = llm if client is None else client
        agent = agent_for(bank_path)
        agent.llm = scoped(client, f"episode/{index}/focal")
        training_opponents = getattr(config, "training_opponents", ())
        population = (training_opponents or config.opponents) if training else config.opponents
        opponent = population[index % len(population)]
        seed = config.seed + index
        expected = {"manifest": digest(manifest), "bank": _bank_digest(agent.bank),
                    "opponent": opponent, "seed": seed, "training": training}
        if path.exists():
            cell = read(path)
            if cell.get("inputs") != expected or cell.get("sha256") != digest(cell.get("record")):
                raise ValueError(f"Invalid cached episode: {path.name}")
            return cell["record"]
        adapter = (adapter_factory(training=training, index=index, bank_path=bank_path, agent_for=agent_for, client=client) if adapter_factory else
                   make_adapter(config, scoped(client, f"episode/{index}/partner")))
        record = execute_episode(adapter=adapter, agent=agent,
                                 opponent_id=opponent, seed=seed, training=training,
                                 history_limit=config.history_limit,
                                 injection_dropout=config.injection_dropout if training else 0.0)
        if record["status"] != "normal" or not record["transitions"][-1]["terminal"]:
            raise ValueError("Incomplete episode")
        if config.game != "hanabi" and record["steps"] != config.horizon:
            raise ValueError("Unexpected episode length")
        if any(t.get("evaluator_metrics", {}).get("opponent_fallback") for t in record["transitions"]):
            raise ValueError("Fallback opponent actions cannot enter experiment results")
        record.update(game=config.game_id, run_index=index, cycle=index // len(config.opponents) + 1)
        write(path, {"inputs": expected, "sha256": digest(record), "record": record})
        return record

    for batch in range(0 if profile else config.batches):
        folder = output / "training" / f"batch_{batch:03d}"
        write(output / "status.json", {"state": "training", "batch": batch})
        if progress:
            progress(f"batch {batch + 1}/{config.batches}: play -> segment -> reflect -> commit")
        training_clients = [llm_factory() for _ in range(config.episodes_per_batch)]
        def train(item):
            i, client = item
            return play(folder / "episodes" / f"{i:04d}.json", bank_path,
                        batch * config.episodes_per_batch + i, training=True, client=client)
        with ThreadPoolExecutor(max_workers=min(config.evaluation_workers, len(training_clients))) as pool:
            records = list(pool.map(train, enumerate(training_clients)))
        all_training.extend(records)
        next_path = output / "banks" / f"batch_{batch:03d}.json"
        complete_path = folder / "complete.json"
        inputs = {"bank": digest(read(bank_path)), "records": digest(records)}
        if complete_path.exists():
            completed = read(complete_path)
            if completed["inputs"] != inputs or completed["bank_sha256"] != digest(read(next_path)):
                raise ValueError("Completed batch artifacts changed")
        else:
            pool = LLMPool(lambda slot: scoped(llm_factory(), f"batch/{batch}/update"), 1)
            segments, segmentation = segment_batch(records, pool, cache_dir=folder / "segments",
                max_workers=1, fixed_policy=config.memory.fixed_policy_protocol,
                abstract_actions=config.memory.abstract_rps_memory)
            # Work on a newly loaded object. The preceding frozen bank is never mutated.
            bank = CentroidMemoryBank(bank_path).load()
            bank.path = next_path
            proposals, assignments = PooledAssigner(bank, embedder, nli,
                game=config.game_id, scope=config.memory.memory_scope,
                similarity_threshold=config.memory.merge_similarity_threshold,
                contradiction_threshold=config.memory.nli_contradiction_threshold,
                max_centroid_size=config.memory.max_centroid_size,
                max_cluster_members=config.memory.max_cluster_members).assign(segments)
            by_index = {r["run_index"]: r for r in records}
            payloads = {cid: {
                "canonical_opponent_strategy": proposal.opponent_strategy,
                "previous_response_strategy": proposal.previous_response_strategy,
                "support_segments": len(proposal.segments),
                **build_evidence(proposal.segments, by_index),
            } for cid, proposal in proposals.items() if proposal.segments}
            reflections = reflect_all(payloads, pool, partial_path=folder / "reflections.json",
                                      max_workers=1, abstract_actions=config.memory.abstract_rps_memory)
            operations = commit(bank, proposals, reflections, embedder,
                metadata={"batch_index": batch},
                contrast_by_cluster={cid: p["injection_contrast"] for cid, p in payloads.items()},
                abstract_rps=config.memory.abstract_rps_memory)
            completed = {"inputs": inputs, "bank_sha256": digest(read(next_path)),
                         "segments": len(segments), "clusters": len(bank.clusters),
                         "operations": operations}
            write(folder / "update.json", {"segmentation": segmentation,
                  "assignments": assignments, "evidence": payloads})
            write(complete_path, completed)
        updates.append(completed)
        bank_path = next_path

    write(output / "status.json", {"state": "evaluation"})
    bank_before = digest(read(bank_path))
    offset = 0 if profile else config.batches * config.episodes_per_batch
    # Allocate clients/journal names on the controller thread. Each task owns its
    # request counter; ordered map keeps results stable when completion order varies.
    clients = [llm_factory() for _ in range(config.test_episodes)]
    def evaluate(item):
        i, client = item
        return play(output / "evaluation" / f"{i:04d}.json", bank_path,
                    offset + i, training=False, client=client)
    with ThreadPoolExecutor(max_workers=min(config.evaluation_workers, len(clients))) as pool:
        evaluation = list(pool.map(evaluate, enumerate(clients)))
    if digest(read(bank_path)) != bank_before:
        raise RuntimeError("Evaluation changed the frozen bank")
    uses = [use for row in all_training for use in row["trace"]["memory_uses"]]
    summary = {"mode": manifest["mode"], "game": config.game_id,
               "train_episodes": len(all_training), "test_episodes": len(evaluation),
               "batches": len(updates), "segments": sum(u["segments"] for u in updates),
               "final_clusters": len(read(bank_path)["clusters"]),
               "train_retrievals": len(uses), "train_injections": sum(u["selected"] for u in uses),
               "test_retrievals": sum(len(r["trace"]["memory_uses"]) for r in evaluation),
               "test_mean_return": sum(r["return"] for r in evaluation) / len(evaluation),
               "test_return_std": pstdev(r["return"] for r in evaluation),
               "test_returns": [r["return"] for r in evaluation],
               "test_seeds": [r["seed"] for r in evaluation],
               "test_opponents": [r["opponent_id"] for r in evaluation],
               "test_metrics": [r["evaluator_metrics"] for r in evaluation],
               "bank": bank_path.relative_to(output).as_posix(), "bank_sha256": bank_before}
    write(output / "summary.json", summary)
    write(output / "status.json", {"state": "complete"})
    return summary
