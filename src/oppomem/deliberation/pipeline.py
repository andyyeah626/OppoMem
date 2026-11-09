"""Public-message negotiation with per-participant ToM and frozen batch memory.

The original Deliberation path used an external reviewer. Here the configured
LLM performs that review, using completed training evidence only. Scoring and
public/private message parsing remain in the original game protocol.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
from statistics import pstdev

from oppomem.models.cache import scoped
from oppomem.runtime.experiment import source_digest
from oppomem.runtime.storage import digest, read, run_lock, write
from .memory import PROTOCOL, build_bank, retrieve
from .protocol import Game, parse_response


def _public(turns):
    return [{"agent": row["agent"], "public_answer": row["public_answer"]} for row in turns]


def play_episode(config, game, path, bank, llm, encoder, *, index, training, participant_factory=None):
    seed = config.seed + index
    inputs = {"bank": digest(bank), "seed": seed, "training": training,
              "config": digest(config.to_dict())}
    record = {"inputs": inputs, "turns": [], "complete": False}
    if path.exists():
        saved = read(path)
        if saved["sha256"] != digest(saved["record"]) or saved["record"]["inputs"] != inputs:
            raise ValueError("Deliberation episode inputs or cached content changed")
        record = saved["record"]
        if record["complete"]:
            return record
    order = game.speaker_order(seed, rounds_num=config.horizon)
    for turn_index in range(len(record["turns"]), len(order)):
        name = order[turn_index]
        phase = "opening" if turn_index == 0 else "final" if turn_index == len(order) - 1 else "discussion"
        base_client = llm if name == game.p1 or participant_factory is None else participant_factory(training, index, name)
        client = scoped(base_client, f"deliberation/{index}/{turn_index}/{name}")
        models = []
        if name == game.p1 and phase != "opening":
            visible = game.visible_history(name, record["turns"], window_size=6)
            seen = {row["agent"] for row in visible}
            for target in game.agent_names:
                if target == name or target not in seen:
                    continue
                prompt = json.dumps({"participant": target, "public_history": visible}, ensure_ascii=False)
                raw = client.complete_json(stage="deliberation_tom",
                    system="Describe this participant's negotiation style from public messages only. "
                           "Return JSON with opponent_strategy: a concise behavioral description under "
                           "60 words, without participant names, numeric private scores or concrete deal codes.",
                    prompt=prompt, max_tokens=256)
                description = raw.get("opponent_strategy")
                if not isinstance(description, str) or not description.strip():
                    raise ValueError("Deliberation ToM must be nonempty text")
                match = retrieve(bank, description, encoder, config.memory.retrieval_min_similarity)
                coin = random.Random(f"{seed}/{match['cluster_id'] if match else 'none'}").random()
                enabled = match is not None and (not training or coin >= config.injection_dropout)
                models.append({"participant": target, "description": description,
                               "retrieval": match, "enabled": enabled})
        prompt = game.round_prompt(name, record["turns"], phase, turn_index,
                                   rounds_num=config.horizon, window_size=6)
        if models:
            advice = [{"participant": row["participant"],
                       "opponent_strategy": row["description"],
                       "response_strategy": row["retrieval"]["response_strategy"] if row["enabled"] else None}
                      for row in models]
            prompt += "\nPrivate, fallible interpretations; public evidence takes priority:\n" + json.dumps(advice)
        for attempt in range(3):
            envelope = "\nRequired output format, including every closing tag: <SCRATCHPAD>private reasoning</SCRATCHPAD><ANSWER>public utterance</ANSWER><PLAN>private plan</PLAN>. The ANSWER closing tag </ANSWER> is mandatory before <PLAN>. Use exactly one of each block and never mention these tag names inside the blocks. Only ANSWER is visible to other participants."
            raw = client.complete_text(stage="deliberation_action", system=game.initial_prompt(name) + envelope,
                                       prompt=prompt, max_tokens=2048)
            parsed = parse_response(raw, phase, game.option_counts)
            if parsed["valid_response"]:
                break
            prompt += "\nInvalid formatting. Return exactly: <SCRATCHPAD>private reasoning</SCRATCHPAD><ANSWER>public text</ANSWER><PLAN>private plan</PLAN>. You MUST write </ANSWER> immediately before <PLAN>."
        else:
            raise ValueError("Deliberation response envelope remained malformed")
        # Other participants see public_answer only. Plans remain private to their author.
        record["turns"].append({"index": turn_index, "agent": name, "phase": phase,
            "public_answer": parsed["public_answer"], "plan": parsed.get("plan", ""),
            "tom": models, "final_valid": parsed.get("final_valid")})
        write(path, {"record": record, "sha256": digest(record)})
    record["metrics"] = game.score_episode(record["turns"])
    record["complete"] = True
    write(path, {"record": record, "sha256": digest(record)})
    return record


def review_batch(config, game, records, bank, llm, encoder, batch):
    evidence = [{"public_history": _public(record["turns"]),
                 "focal_models": [row["tom"] for row in record["turns"] if row["agent"] == game.p1],
                 "focal_terminal_return": record["metrics"]["focal_realized_utility"]}
                for record in records]
    prior = [{key: row[key] for key in ("cluster_id", "opponent_strategy", "response_strategy")}
             for row in bank["clusters"]]
    prompt = json.dumps({"prior_entries": prior, "completed_training_games": evidence}, ensure_ascii=False)
    client = scoped(llm, f"deliberation/batch/{batch}/review")
    for attempt in range(3):
        raw = client.complete_json(stage="deliberation_memory_review",
            system="Review completed training negotiations. Return JSON with entries, the full next "
                   "memory bank, using 1 to 8 supported entries. Keep, revise or replace prior entries "
                   "only using the supplied evidence. Each entry has exactly cluster_id (stable ASCII ID), "
                   "opponent_strategy (under 60 words), response_strategy (under 40 words). Describe reusable "
                   "behavior and conditional advice. Do not include names, numeric scores, concrete option "
                   "codes such as A1/B2, or a fixed deal package. Never infer hidden participant utilities.",
            prompt=prompt, max_tokens=2048)
        try:
            entries = raw.get("entries")
            if not isinstance(entries, list) or not 1 <= len(entries) <= 8:
                raise ValueError("entries must contain 1 to 8 memories")
            return build_bank(entries, encoder, metadata={
                "batch": batch, "author": getattr(config, "reflection_model", config.models.model),
                "external_reflection": True,
                "update_policy": "llm_review_of_completed_training_only",
                "encoder": config.models.encoder,
                "retrieval_min_similarity": config.memory.retrieval_min_similarity})
        except ValueError as exc:
            prompt += "\nCorrect the previous invalid bank: " + str(exc) + "\nPrevious output: " + json.dumps(raw)
    raise ValueError("Deliberation review remained invalid after 3 attempts")


def run_experiment(config, output_dir, *, llm_factory, embedder, nli=None, progress=None, participant_factory=None):
    output = Path(output_dir)
    game = Game(Path(config.game_dir), prompt_source=config.prompt_source, use_config_incentives=True)
    assets = [p for p in game.game_dir.rglob("*") if p.is_file() and p.suffix == ".txt"]
    assets.append(game.prompt_source / "initial_prompts.py")
    manifest = {"config": config.to_dict(), "source_sha256": source_digest(), "mode": "live",
                "assets": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}}
    with run_lock(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if read(manifest_path) != manifest:
                raise ValueError("Deliberation configuration, source or game assets changed")
        elif any(p.name != "run.lock" for p in output.iterdir()):
            raise ValueError("Output directory contains artifacts without a manifest")
        else:
            write(manifest_path, manifest)
        try:
            return _run(config, output, game, llm_factory, embedder, progress, participant_factory)
        except Exception as exc:
            write(output / "status.json", {"state": "failed", "error_type": type(exc).__name__})
            raise


def _run(config, output, game, llm_factory, encoder, progress, participant_factory):
    llm = llm_factory()
    bank = {"schema_version": 7, "metadata": {"game": "llm_deliberation", "protocol": PROTOCOL}, "clusters": []}
    initial = output / "banks/initial.json"
    if initial.exists() and read(initial) != bank:
        raise ValueError("Initial bank changed")
    write(initial, bank)
    training = []
    for batch in range(config.batches):
        folder = output / "training" / f"batch_{batch:03d}"
        write(output / "status.json", {"state": "training", "batch": batch})
        if progress:
            progress(f"deliberation batch {batch + 1}/{config.batches}: play -> review -> freeze")
        clients = [llm_factory() for _ in range(config.episodes_per_batch)]
        def train(item):
            i, client = item
            return play_episode(config, game, folder / "episodes" / f"{i:04d}.json", bank, client, encoder,
                    index=batch * config.episodes_per_batch + i, training=True, participant_factory=participant_factory)
        with ThreadPoolExecutor(max_workers=min(config.evaluation_workers, len(clients))) as pool:
            records = list(pool.map(train, enumerate(clients)))
        training.extend(records)
        inputs = {"bank": digest(bank), "episodes": digest(records)}
        path, marker = output / "banks" / f"batch_{batch:03d}.json", folder / "complete.json"
        if marker.exists():
            checkpoint = read(marker)
            if checkpoint["inputs"] != inputs or checkpoint["bank_sha256"] != digest(read(path)):
                raise ValueError("Deliberation batch artifacts changed")
            bank = read(path)
        else:
            bank = review_batch(config, game, records, bank, llm, encoder, batch)
            write(path, bank)
            write(marker, {"inputs": inputs, "bank_sha256": digest(bank)})
    frozen = digest(bank)
    write(output / "status.json", {"state": "evaluation"})
    offset = config.batches * config.episodes_per_batch
    clients = [llm_factory() for _ in range(config.test_episodes)]
    def evaluate(item):
        i, client = item
        return play_episode(config, game, output / "evaluation" / f"{i:04d}.json", bank, client, encoder,
                          index=offset + i, training=False, participant_factory=participant_factory)
    with ThreadPoolExecutor(max_workers=min(config.evaluation_workers, len(clients))) as pool:
        tests = list(pool.map(evaluate, enumerate(clients)))
    if digest(bank) != frozen or digest(read(path)) != frozen:
        raise ValueError("Evaluation changed the bank")
    def uses(records):
        return [model for record in records for row in record["turns"] for model in row["tom"]]
    returns = [r["metrics"]["focal_realized_utility"] for r in tests]
    summary = {"mode": "live", "game": "llm_deliberation", "train_episodes": len(training),
               "test_episodes": len(tests), "batches": config.batches, "final_clusters": len(bank["clusters"]),
               "train_retrievals": sum(m["retrieval"] is not None for m in uses(training)),
               "train_injections": sum(m["enabled"] for m in uses(training)),
               "test_retrievals": sum(m["retrieval"] is not None for m in uses(tests)),
               "test_returns": returns, "test_seeds": [r["inputs"]["seed"] for r in tests],
               "test_mean_return": sum(returns) / len(returns), "test_return_std": pstdev(returns),
               "test_valid_finals": sum(r["metrics"]["final_valid"] for r in tests),
               "test_successful_finals": sum(r["metrics"]["final_success"] for r in tests),
               "bank": path.relative_to(output).as_posix(), "bank_sha256": frozen}
    write(output / "summary.json", summary)
    write(output / "status.json", {"state": "complete"})
    return summary
