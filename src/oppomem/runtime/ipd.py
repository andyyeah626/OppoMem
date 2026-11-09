"""IPD experiment primitives: durable calls, frozen E5 medoids and strict play.

Kept separate from historical RPS/IPD entrypoints so their source manifests remain
valid. No model-generated fallback actions are accepted as experiment samples.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time

import numpy as np

from oppomem.games.ipd import ACTIONS
from oppomem.games.llm_ipd import LLMOpponentIPDAdapter, OPPONENT_SYSTEM, _opponent_prompt
from oppomem.prompts.ipd import PLAYER_SYSTEM as IPD_PLAYER_SYSTEM, player_prompt as ipd_player_prompt
from oppomem.types import CentroidMatch

THRESHOLD = 0.9166886568069458
PROTOCOL = "ipd-best-medoid-nli-8bot-3gpt-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class CallFailure(RuntimeError):
    """Safe-to-display provider failure, without proxy response bodies or secrets."""


class DurableAPI:
    """Thread-safe journal; clients are local to each worker, raw replies saved first."""

    def __init__(self, path, factories, *, sleep=time.sleep):
        self.factories, self.sleep = factories, sleep
        self.local = threading.local()
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, request TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, key TEXT,
            provider TEXT, stage TEXT, status TEXT, detail TEXT, time REAL);
          CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, output TEXT);
        """)
        self.db.execute("UPDATE attempts SET status='remote_outcome_unknown_after_restart' WHERE status='started'")
        self.db.commit()
        self.key_locks = {}

    def client(self, provider):
        if not hasattr(self.local, "clients"):
            self.local.clients = {}
        if provider not in self.local.clients:
            self.local.clients[provider] = self.factories[provider]()
        return self.local.clients[provider]

    @staticmethod
    def validate(output, stage):
        if not isinstance(output, dict):
            raise ValueError("JSON object required")
        if stage in ("decision", "opponent_decision") and output.get("action") not in ACTIONS:
            raise ValueError("IPD legal action required")
        if stage == "tom" and not str(output.get("opponent_strategy", "")).strip():
            raise ValueError("ToM required")
        if stage == "memory_windows" and not isinstance(output.get("windows"), list):
            raise ValueError("windows required")
        if stage == "memory_reflection" and not str(output.get("response_strategy", "")).strip():
            raise ValueError("nonempty reflection required")
        return output

    def call(self, provider, scope, *, stage, system, prompt, max_tokens=None):
        client = self.client(provider)
        request = dict(provider=provider, model=client.model, scope=scope, stage=stage,
                       system=system, prompt=prompt, max_tokens=max_tokens,
                       reasoning=getattr(client, "reasoning_effort", "minimal"))
        key = digest(request)
        with self.lock:
            lock = self.key_locks.setdefault(key, threading.Lock())
        with lock:
            return self._call_locked(key, request, client)

    def _call_locked(self, key, request, client):
        stage, provider = request["stage"], request["provider"]
        with self.lock:
            row = self.db.execute("SELECT output FROM responses WHERE key=?", (key,)).fetchone()
            if row:
                return self.validate(json.loads(row[0]), stage)
            received = self.db.execute("SELECT detail FROM attempts WHERE key=? AND status='received' ORDER BY id DESC", (key,)).fetchall()
            for (detail,) in received:
                try:
                    output = self.validate(json.loads(json.loads(detail)["raw"]), stage)
                except (ValueError, TypeError, KeyError):
                    continue
                self.db.execute("INSERT OR REPLACE INTO responses VALUES (?,?)", (key, json.dumps(output)))
                self.db.commit()
                return output
            self.db.execute("INSERT OR IGNORE INTO requests VALUES (?,?)", (key, json.dumps(request)))
            self.db.commit()
        budget = max(client.MIN_TOKEN_BUDGET, request["max_tokens"] or 512)
        for attempt in range(6):
            with self.lock:
                ident = self.db.execute("INSERT INTO attempts(key,provider,stage,status,detail,time) VALUES (?,?,?,?,?,?)",
                    (key, provider, stage, "started", json.dumps({"budget": budget}), time.time())).lastrowid
                self.db.commit()
            try:
                raw, usage = client._generate(system=request["system"], prompt=request["prompt"], token_budget=budget)
                with self.lock:
                    self.db.execute("UPDATE attempts SET status='received',detail=? WHERE id=?",
                        (json.dumps({"raw": raw, "usage": usage, "budget": budget}), ident))
                    self.db.commit()
                output = self.validate(json.loads(raw), stage)
                with self.lock:
                    self.db.execute("INSERT OR REPLACE INTO responses VALUES (?,?)", (key, json.dumps(output)))
                    self.db.commit()
                return output
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                safe = {"exception_type": type(exc).__name__, "status_code": status}
                with self.lock:
                    self.db.execute("UPDATE attempts SET status='failed',detail=? WHERE id=? AND status='started'",
                                    (json.dumps(safe), ident))
                    self.db.commit()
                if status is not None and status < 500 and status not in (408, 409, 429):
                    raise CallFailure(f"{provider}/{stage}: {type(exc).__name__} HTTP {status}") from None
                if attempt == 5:
                    raise CallFailure(f"{provider}/{stage}: retry exhausted ({type(exc).__name__}, HTTP {status})") from None
                if status is None and isinstance(exc, (ValueError, RuntimeError)):
                    budget = min(client.MAX_TOKEN_BUDGET, budget * 2)
                self.sleep(min(60, 2 ** (attempt + 1)))
        raise AssertionError("unreachable")

    def stats(self):
        with self.lock:
            totals = Counter()
            for provider, detail in self.db.execute("SELECT provider, detail FROM attempts WHERE status='received'"):
                for name, value in zip(("input_tokens", "output_tokens", "reasoning_tokens"), json.loads(detail)["usage"]):
                    totals[f"{provider}/{name}"] += value
            return {"successful_unique_requests": self.db.execute("SELECT count(*) FROM responses").fetchone()[0],
                    "attempt_statuses": dict(self.db.execute("SELECT status,count(*) FROM attempts GROUP BY status")),
                    "tokens": dict(totals)}


class ScopedLLM:
    def __init__(self, api, provider, scope):
        self.api, self.provider, self.scope = api, provider, scope
        self.requests_made = 0
        self.events = []

    def complete_json(self, **kwargs):
        self.requests_made += 1
        return self.api.call(self.provider, self.scope, **kwargs)


class StrictIPDAdapter(LLMOpponentIPDAdapter):
    def _next_opponent_action(self):
        # A failed call aborts the cell; cached successful decisions reconstruct the
        # same prefix on resume. Never turn provider failure into a random opponent.
        self.calls += 1
        raw = self.llm.complete_json(stage="opponent_decision", system=OPPONENT_SYSTEM,
            prompt=_opponent_prompt(horizon=self.horizon, view=self._opponent_view))
        return self._parse(raw), False


class PlainTextIPDOpponentAdapter(LLMOpponentIPDAdapter):
    """IPD opponent using the current RPS-style plain-text action contract."""

    def _next_opponent_action(self):
        self.calls += 1
        rows = [{"round": row["round"], "self": row["you"],
                 "opponent": row["them"], "reward": row["payoff"]}
                for row in self._opponent_view]
        raw = self.llm.complete_json(
            stage="opponent_decision", system=IPD_PLAYER_SYSTEM,
            prompt=ipd_player_prompt(rows, horizon=self.horizon),
        )
        return self._parse(raw), False


class LocalModels:
    def __init__(self, cache):
        from oppomem.models.semantics import MiniLMEmbedder, RetrievalEmbedder, DebertaNLIScorer
        import torch
        torch.set_num_threads(2)
        self.lock = threading.RLock()
        self.mini = MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", cache_dir=str(cache))
        self.e5 = RetrievalEmbedder(MiniLMEmbedder("intfloat/e5-base-v2", cache_dir=str(cache)), "e5")
        self.nli = DebertaNLIScorer("cross-encoder/nli-deberta-v3-xsmall", cache_dir=str(cache))


def make_index(bank, members, models):
    rows = []
    for cid in sorted(bank.clusters):
        entries = members.get(cid, [])
        if len(entries) != bank.clusters[cid].centroid_size:
            raise ValueError(f"Incomplete true membership: {cid}")
        texts = [m["opponent_strategy"] for m in entries]
        if not texts:
            raise ValueError(f"Empty membership: {cid}")
        with models.lock:
            vectors = np.concatenate([models.e5.encode_documents(
                ["Opponent strategy: " + t for t in texts[i:i + 32]]) for i in range(0, len(texts), 32)])
        center = vectors.mean(axis=0)
        selected = int(np.argmax(vectors @ center))
        rows.append({"cluster_id": cid, "support": len(entries), "medoid_text": texts[selected],
                     "medoid_member": entries[selected], "medoid_embedding": vectors[selected].tolist()})
    return {"encoder": "intfloat/e5-base-v2", "pooling": "mean", "prefix_mode": "e5",
            "top_k": 8, "threshold": THRESHOLD, "clusters": rows}


class BestMedoidRetriever:
    def __init__(self, index, models):
        self.index, self.models = index, models
        self.ids = [r["cluster_id"] for r in index["clusters"]]
        self.vectors = np.array([r["medoid_embedding"] for r in index["clusters"]], dtype=np.float32)
        self.min_similarity = THRESHOLD

    def rank(self, text, bank, *, audit=None):
        # ToMMemoryAgent passes ToMState.retrieval_text(), which already includes
        # the semantic prefix. RPS used the bare hypothesis for directional NLI.
        text = text.removeprefix("Opponent strategy: ")
        audit = audit if audit is not None else {}
        audit.update(backend="best_medoid_nli", threshold=THRESHOLD, selected=False)
        if not self.ids:
            audit["reason"] = "empty_bank"
            return []
        with self.models.lock:
            q = self.models.e5.encode_query(["Opponent strategy: " + text])[0]
            order = np.argsort(-(self.vectors @ q), kind="stable")[:8]
            scores = self.models.nli.score_many([(text, bank.clusters[self.ids[i]].opponent_strategy) for i in order])
        winner = max(range(len(order)), key=lambda j: scores[j].entailment)
        cid, score = self.ids[order[winner]], float(scores[winner].entailment)
        audit.update(top8=[self.ids[i] for i in order], entailments=[s.entailment for s in scores],
                     cluster_id=cid, similarity=score, metric="entailment", selected=score >= THRESHOLD,
                     reason="accepted" if score >= THRESHOLD else "below_threshold")
        if score < THRESHOLD:
            return []
        c = bank.clusters[cid]
        return [CentroidMatch(cluster_id=cid, opponent_strategy=c.opponent_strategy,
                              response_strategy=c.response_strategy, score=score, metric="entailment")]


def validate_record(record, item, horizon):
    if record.get("status") != "normal" or record.get("steps") != horizon:
        raise ValueError("Incomplete episode")
    for key in ("seed", "opponent_id"):
        if record.get(key) != item[key]:
            raise ValueError(f"Episode {key} mismatch")
    transitions = record["transitions"]
    turns = record["trace"]["turns"]
    if len(transitions) != horizon or len(turns) != horizon or not transitions[-1]["terminal"]:
        raise ValueError("Incomplete trajectory")
    payoff = {("Cooperate", "Cooperate"): (3, 3), ("Cooperate", "Defect"): (0, 5),
              ("Defect", "Cooperate"): (5, 0), ("Defect", "Defect"): (1, 1)}
    total, opponent_total = 0, 0
    for i, (t, turn) in enumerate(zip(transitions, turns)):
        if t["step"] != i or turn["round_index"] != i or t.get("explored"):
            raise ValueError("Invalid round sequence")
        metrics = t.get("evaluator_metrics", {})
        if metrics.get("opponent_fallback") or metrics.get("opponent_fallbacks"):
            raise ValueError("Fallback trajectory")
        r, o = payoff[(t["action"], t["opponent_event"])]
        total += r
        opponent_total += o
        if (t["reward_delta"] != r or turn["reward"] != r or t["cumulative_return"] != total
            or turn["agent_action"] != t["action"] or turn["opponent_action"] != t["opponent_event"]):
            raise ValueError("Payoff/trace mismatch")
    if record["return"] != total or record["evaluator_metrics"]["opponent_return"] != opponent_total:
        raise ValueError("Return mismatch")
    arm = item["arm"]
    toms, uses = record["trace"]["tom_chain"], record["trace"]["memory_uses"]
    if (arm == "vanilla" and (toms or uses)) or (arm != "vanilla" and len(toms) != horizon):
        raise ValueError("Arm ToM mismatch")
    if arm == "tom_only" and any(u["selected"] for u in uses):
        raise ValueError("Memory leaked into tom_only")


def summarize(records):
    rewards = np.array([r["return"] for r in records], dtype=float)
    opponents = np.array([r["evaluator_metrics"]["opponent_return"] for r in records], dtype=float)
    turns = [t for r in records for t in r["trace"]["turns"]]
    uses = [u for r in records for u in r["trace"]["memory_uses"]]
    rng = np.random.default_rng(20260919)
    means = rng.choice(rewards, (10000, len(rewards)), replace=True).mean(axis=1)
    return {"episodes": len(records), "mean_return": float(rewards.mean()),
            "mean_return_ci95": np.quantile(means, [.025, .975]).tolist(),
            "mean_opponent_return": float(opponents.mean()),
            "cooperation_rate": sum(t["agent_action"] == "Cooperate" for t in turns) / len(turns),
            "opponent_cooperation_rate": sum(t["opponent_action"] == "Cooperate" for t in turns) / len(turns),
            "wins": int(sum(rewards > opponents)), "draws": int(sum(rewards == opponents)),
            "losses": int(sum(rewards < opponents)), "retrieval_hit_rate": len(uses) / len(turns),
            "injection_rate": sum(u["selected"] for u in uses) / len(turns)}
