"""Trajectory-grounded manual memory for the focal deliberation agent.

This module never asks an API to reflect. The supervising Codex task reviews the
allowlisted evidence and submits a complete, reasoned bank revision. Public
answers remain verbatim evidence; other agents' plans and private utilities do
not enter the review. Semantic claims must still be checked by that reviewer.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


PROTOCOL = "deliberation-memory-v1"
REVIEW_SCHEMA = "deliberation-memory-review-v1"
PREFIX = "Opponent strategy: "
METRICS = frozenset({
    "final_success", "any_success", "focal_base_utility", "focal_realized_utility",
    "final_valid", "focal_wrong_deals", "focal_wrong_deal_count", "focal_deal_count",
})
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_CONCRETE_ID = re.compile(r"\b(?:p\d+|agent[ _-]*\d+)\b", re.IGNORECASE)
_SCORE_CLAIM = re.compile(
    r"\b(?:score|utility|payoff|threshold)\s*(?:(?:is|of|equals)\s*|[=:]\s*)?-?\d"
    r"|\b\d+(?:\.\d+)?\s+points\b", re.IGNORECASE,
)

# A deliberation "concrete action" is naming an option: one issue letter plus one
# option number pins a full package.  Persisted bank text must never do that, or the
# bank degrades into a lookup table for one fixed deal (see core/rps_memory.py).
_OPTION_RE = re.compile(r"\b[A-E][1-9]\b")
_PACKAGE_RUN_RE = re.compile(r"\b[A-E][1-9](?:(?:\s*[/,+\-]\s*|\s+)[A-E][1-9])+\b")
_PACKAGE_PROSE_RE = re.compile(
    r"\b(?:option|choice|level)\s*[1-9]\s*(?:of|in|for|on)\s*(?:issue|topic|item)?\s*[A-E]\b"
    r"|\b(?:issue|topic|item)\s*[A-E]\s*(?:option|choice|level)\s*[1-9]\b"
    r"|\b[A-E]\s*[=:]\s*[1-9]\b",
    re.IGNORECASE,
)
_ABSTRACT_PACKAGE = "a specific complete package"
_ABSTRACT_OPTION = "one specific option"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(value, name, *, words=None):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    if len(value) > 100_000 or any(ord(c) < 32 and c not in "\n\r\t" for c in value):
        raise ValueError(f"Invalid {name}")
    if words is not None and len(value.split()) > words:
        raise ValueError(f"{name} exceeds {words} words")
    return value


def _memory_text(value, name, words):
    result = _text(value, name, words=words).strip()
    if _CONCRETE_ID.search(result) or _SCORE_CLAIM.search(result):
        raise ValueError(f"{name} must abstract concrete agent identities and numeric scores")
    return result


def has_concrete_package(text) -> bool:
    """True when the text names a concrete option and so pins one fixed deal."""
    value = str(text or "")
    return bool(_OPTION_RE.search(value) or _PACKAGE_PROSE_RE.search(value))


def abstract_opponent_strategy(text) -> str:
    """Remove concrete option names while retaining behavioral structure."""
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return value
    value = _PACKAGE_RUN_RE.sub(_ABSTRACT_PACKAGE, value)
    value = _PACKAGE_PROSE_RE.sub(_ABSTRACT_OPTION, value)
    value = _OPTION_RE.sub(_ABSTRACT_OPTION, value)
    return " ".join(value.split()).strip()


def abstract_response_strategy(text=None) -> str:
    """Preserve useful abstract advice; never fabricate or overwrite a response."""
    value = " ".join(str(text or "").split()).strip()
    if not value:
        raise ValueError("response_strategy must not be empty")
    if has_concrete_package(value):
        raise ValueError("response_strategy must not name a concrete option package")
    return value


def sanitize_cluster_texts(cluster: dict) -> dict:
    """Abstract one cluster dict in place and report which fields changed.

    Raises on a response that names a concrete option: such advice cannot be
    abstracted without inventing content, so it must be rewritten or dropped.
    """
    changed: dict[str, bool] = {}
    for field in ("opponent_strategy", "prototype_strategy"):
        value = cluster.get(field)
        if value:
            replacement = abstract_opponent_strategy(value)
            changed[field] = replacement != value
            cluster[field] = replacement
    response = abstract_response_strategy(cluster.get("response_strategy"))
    changed["response_strategy"] = response != cluster.get("response_strategy")
    cluster["response_strategy"] = response
    for row in cluster.get("members") or []:
        if isinstance(row, dict) and row.get("opponent_strategy"):
            row["opponent_strategy"] = abstract_opponent_strategy(row["opponent_strategy"])
    return changed


def _vector(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Embedding must be a nonempty vector")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError("Embedding must contain finite numbers")
    result = [float(v) for v in value]
    if not any(result):
        raise ValueError("Embedding must have nonzero norm")
    return result


def _validate_bank(bank, *, strict=False):
    """Validate a schema-7 bank.  `strict` additionally enforces option-name hygiene.

    Legacy banks (M0-M3) name concrete option packages, so historical replay must
    keep loading them leniently.  Every bank written from now on is strict.
    """
    if not isinstance(bank, dict) or bank.get("schema_version") != 7:
        raise ValueError("Expected schema_version 7")
    metadata = bank.get("metadata", {})
    if metadata.get("game") != "llm_deliberation" or metadata.get("protocol") != PROTOCOL:
        raise ValueError("Incompatible deliberation memory protocol")
    if not isinstance(bank.get("clusters"), list):
        raise ValueError("Bank clusters must be a list")
    seen, dimension = set(), None
    for cluster in bank["clusters"]:
        ident = cluster.get("cluster_id") if isinstance(cluster, dict) else None
        if not isinstance(ident, str) or not _ID.fullmatch(ident) or ident in seen:
            raise ValueError("Invalid or duplicate cluster_id")
        seen.add(ident)
        description = _memory_text(cluster.get("opponent_strategy"), "opponent_strategy", 60)
        response = _memory_text(cluster.get("response_strategy"), "response_strategy", 40)
        if strict:
            abstract_response_strategy(response)
            if abstract_opponent_strategy(description) != description:
                raise ValueError("opponent_strategy must abstract concrete option choices")
        if cluster.get("prototype_strategy") != description:
            raise ValueError("Prototype text must equal the reviewed description")
        center = _vector(cluster.get("centroid"))
        prototype = _vector(cluster.get("prototype_embedding"))
        if center != prototype:
            raise ValueError("Manual bank centroid and prototype must agree")
        if dimension is not None and len(center) != dimension:
            raise ValueError("Embedding dimensions differ")
        dimension = len(center)
    return bank


def load_bank(path) -> dict:
    """Load a schema-7 deliberation bank, rejecting incompatible or corrupt data."""
    return _validate_bank(_read(path))


def retrieve(bank, description, encoder, threshold=.74, audit=None) -> dict | None:
    """Return the strongest matching reviewed description, or no memory.

    When `audit` is a dict it receives every cluster's similarity, so any other
    threshold can be recomputed offline from a stored episode without paying for
    another game (same contract as CentroidRetriever.rank(audit=...)).
    """
    _validate_bank(bank)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not -1 <= threshold <= 1:
        raise ValueError("threshold must be in [-1, 1]")
    if audit is not None and not isinstance(audit, dict):
        raise ValueError("audit must be a dict")
    _text(description, "description")
    if not bank["clusters"]:
        if audit is not None:
            audit.update({"all_similarities": {}, "selected": None, "threshold": float(threshold)})
        return None
    query = _vector(encoder.encode([PREFIX + description])[0])
    ranked = []
    for cluster in bank["clusters"]:
        vector = _vector(cluster["prototype_embedding"])
        if len(query) != len(vector):
            raise ValueError("Query/bank embedding dimensions differ")
        score = sum(a * b for a, b in zip(query, vector)) / math.sqrt(
            sum(a * a for a in query) * sum(b * b for b in vector))
        ranked.append((min(1.0, max(-1.0, score)), cluster))
    score, selected = sorted(ranked, key=lambda row: (-row[0], row[1]["cluster_id"]))[0]
    if audit is not None:
        audit.update({"all_similarities": {c["cluster_id"]: s for s, c in ranked},
                      "selected": selected["cluster_id"], "threshold": float(threshold)})
    if score < threshold:
        return None
    return {key: selected[key] for key in ("cluster_id", "opponent_strategy", "response_strategy")} | {"score": score}


def _immutable_json(path, value):
    """Commit once with an atomic no-replace hard link; identical retries are safe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"Cannot replace committed file: {path.name}")
        return path
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            if _read(path) != value:
                raise ValueError(f"Cannot replace committed file: {path.name}") from None
    finally:
        Path(temp).unlink(missing_ok=True)
    return path


def _review_retrieval(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Invalid retrieval evidence")
    result = {name: deepcopy(value[name]) for name in (
        "cluster_id", "opponent_strategy", "response_strategy", "score", "selected", "injected"
    ) if name in value}
    for name in ("cluster_id", "opponent_strategy", "response_strategy"):
        if name in result:
            _text(result[name], "retrieval " + name)
    if "score" in result and (isinstance(result["score"], bool) or not isinstance(result["score"], (int, float)) or not math.isfinite(result["score"])):
        raise ValueError("Invalid retrieval score")
    for name in ("selected", "injected"):
        if name in result and not isinstance(result[name], bool):
            raise ValueError("Retrieval injection flags must be boolean")
    return result


def export_review(run_dir, batch: int, records: list[dict], source_bank_path: Path | None) -> Path:
    """Write only focal-visible training evidence to a durable review file."""
    if type(batch) is not int or batch < 0:
        raise ValueError("batch must be a nonnegative integer")
    if (batch == 0) != (source_bank_path is None):
        raise ValueError("Only bootstrap batch0 has no source bank")
    if not isinstance(records, list) or not records:
        raise ValueError("Review requires completed episodes")
    source = load_bank(source_bank_path) if source_bank_path is not None else None
    episodes, seen = [], set()
    for record in records:
        spec = record.get("spec", {})
        ident = _text(spec.get("id"), "episode id")
        if ident in seen or record.get("complete") is not True:
            raise ValueError("Review requires unique completed episodes")
        seen.add(ident)
        focal = spec.get("focal_agent", "p1")
        _text(focal, "focal_agent")
        clean_spec = {key: deepcopy(spec[key]) for key in ("id", "arm", "seed", "bank_sha256", "focal_agent") if key in spec}
        clean_spec["focal_agent"] = focal
        _text(clean_spec.get("arm"), "arm")
        if type(clean_spec.get("seed")) is not int:
            raise ValueError("Episode seed must be an integer")
        if clean_spec.get("bank_sha256") is not None and not re.fullmatch(r"[0-9a-f]{64}", str(clean_spec["bank_sha256"])):
            raise ValueError("Invalid episode bank hash")
        if source and spec.get("bank_sha256") != _sha(source_bank_path):
            raise ValueError("Episode used a different source bank")
        turns, indexes = [], set()
        for turn in record.get("turns", []):
            index = turn.get("index")
            if type(index) is not int or index < 0 or index in indexes:
                raise ValueError("Invalid or duplicate turn index")
            if indexes and index <= max(indexes):
                raise ValueError("Public turns must remain in chronological order")
            indexes.add(index)
            clean = {"index": index, "agent": _text(turn.get("agent"), "turn agent"),
                     "public_answer": _text(turn.get("public_answer"), "public answer")}
            if "phase" in turn:
                clean["phase"] = _text(turn["phase"], "phase")
            if turn["agent"] == focal:
                plan = turn.get("plan")
                if plan is not None:
                    if not isinstance(plan, str):
                        raise ValueError("focal plan must be a string")
                    if plan.strip():
                        clean["plan"] = _text(plan, "focal plan")
                clean["tom"] = []
                for tom in turn.get("tom", []) or []:
                    clean["tom"].append({"opponent": _text(tom.get("opponent"), "ToM opponent"),
                        "description": _text(tom.get("description"), "ToM description"),
                        "retrieval": _review_retrieval(tom.get("retrieval"))})
            turns.append(clean)
        if not turns:
            raise ValueError("Review episode has no public trajectory")
        metrics = {}
        for key, value in record.get("metrics", {}).items():
            if key in METRICS:
                if value is not None and (not isinstance(value, (int, float, bool)) or not math.isfinite(value)):
                    raise ValueError("Review metrics must be scalar focal outcomes")
                metrics[key] = value
        episodes.append({"spec": clean_spec, "complete": True, "turns": turns,
            "focal_initial_prompt": _text(record.get("focal_initial_prompt"), "focal_initial_prompt"),
            "metrics": metrics})
    review = {"review_schema": REVIEW_SCHEMA, "batch": batch,
        "source_bank_path": str(Path(source_bank_path).resolve()) if source_bank_path is not None else None,
        "source_bank_sha256": _sha(source_bank_path) if source_bank_path is not None else None,
        "clusters": [{key: c[key] for key in ("cluster_id", "opponent_strategy", "response_strategy")}
                     for c in source["clusters"]] if source else [],
        "episodes": episodes}
    return _immutable_json(Path(run_dir) / "reviews" / f"batch{batch}.json", review)


def _evidence(change, known):
    _text(change.get("reason"), "reason")
    evidence = change.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Every memory decision needs evidence")
    for ref in evidence:
        if not isinstance(ref, dict) or set(ref) != {"episode_id", "turn_index"}:
            raise ValueError("Evidence requires episode_id and turn_index")
        if not isinstance(ref["episode_id"], str) or type(ref["turn_index"]) is not int or (ref["episode_id"], ref["turn_index"]) not in known:
            raise ValueError("Evidence references an unknown episode/turn")


def verify_prototypes(bank, encoder, *, tolerance=1e-6) -> bool:
    """Re-encode every description and require the stored prototype to match.

    Guards against a bank whose text and vector disagree, which would silently
    retrieve the wrong advice (same check as optimization_audit._check_bank).
    """
    _validate_bank(bank)
    clusters = bank["clusters"]
    encoded = encoder.encode([PREFIX + c["opponent_strategy"] for c in clusters])
    if len(encoded) != len(clusters):
        raise ValueError("Encoder returned wrong number of vectors")
    for cluster, vector in zip(clusters, encoded):
        stored, fresh = _vector(cluster["prototype_embedding"]), _vector(vector)
        if len(stored) != len(fresh) or any(abs(a - b) > tolerance for a, b in zip(stored, fresh)):
            raise ValueError(f"Prototype embedding does not match its text: {cluster['cluster_id']}")
    return True


def build_bank(entries, encoder, *, metadata=None, path=None):
    """Encode hand-written abstract entries into a strict, frozen bank.

    Mirrors the RPS candidate path: write the entries, encode locally, verify the
    prototype vector, then freeze.  Strictness is enforced here and in
    `apply_patch`; `load_bank` stays lenient so legacy banks keep replaying.
    """
    if not isinstance(entries, list) or not entries:
        raise ValueError("Bank requires at least one entry")
    clean, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"cluster_id", "opponent_strategy", "response_strategy"}:
            raise ValueError("Entry requires cluster_id, opponent_strategy, response_strategy")
        ident = entry.get("cluster_id")
        if not isinstance(ident, str) or not _ID.fullmatch(ident) or ident in seen:
            raise ValueError("Invalid or duplicate cluster_id")
        seen.add(ident)
        clean.append({
            "cluster_id": ident,
            "opponent_strategy": abstract_opponent_strategy(
                _memory_text(entry["opponent_strategy"], "opponent_strategy", 60)),
            "response_strategy": abstract_response_strategy(
                _memory_text(entry["response_strategy"], "response_strategy", 40)),
        })
    base = {"game": "llm_deliberation", "protocol": PROTOCOL,
            "mechanism": "tom_centroid_v1", "retrieval_backend": "cosine", "use_prototype": True,
            "retrieval_min_similarity": .74, "encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_prefix": PREFIX, "author": "current_codex_task", "external_reflection": False}
    base.update(metadata or {})
    encoded = encoder.encode([PREFIX + entry["opponent_strategy"] for entry in clean])
    if len(encoded) != len(clean):
        raise ValueError("Encoder returned wrong number of vectors")
    clusters = []
    for entry, vector in zip(clean, encoded):
        value = _vector(vector)
        clusters.append({**entry, "centroid": value, "centroid_size": 1,
                         "prototype_strategy": entry["opponent_strategy"],
                         "prototype_embedding": deepcopy(value),
                         "members": [{"opponent_strategy": entry["opponent_strategy"],
                                      "embedding": deepcopy(value)}]})
    bank = {"schema_version": 7, "metadata": base, "clusters": clusters}
    _validate_bank(bank, strict=True)
    return _immutable_json(path, bank) if path is not None else bank


def apply_patch(run_dir, batch: int, patch_path, encoder=None) -> Path:
    """Validate a complete manual revision and commit an immutable M{batch}.json."""
    if type(batch) is not int or batch < 0:
        raise ValueError("batch must be a nonnegative integer")
    run_dir = Path(run_dir)
    review_path = run_dir / "reviews" / f"batch{batch}.json"
    review, patch = _read(review_path), _read(patch_path)
    if not isinstance(patch, dict) or set(patch) != {"source_bank_sha256", "review_sha256", "entries", "deleted"}:
        raise ValueError("Patch requires source_bank_sha256, review_sha256, entries, deleted")
    if review.get("review_schema") != REVIEW_SCHEMA or review.get("batch") != batch:
        raise ValueError("Incompatible review")
    if patch.get("review_sha256") != _sha(review_path) or patch.get("source_bank_sha256") != review.get("source_bank_sha256"):
        raise ValueError("Stale manual review or source bank")
    source_path = review.get("source_bank_path")
    if (batch == 0) != (source_path is None):
        raise ValueError("Invalid bootstrap source")
    source = load_bank(source_path) if source_path else None
    if source_path and _sha(source_path) != review["source_bank_sha256"]:
        raise ValueError("Source bank changed after review")
    prior = {c["cluster_id"]: c for c in source["clusters"]} if source else {}
    entries, deleted = patch.get("entries"), patch.get("deleted")
    if not isinstance(entries, list) or not isinstance(deleted, list):
        raise ValueError("Patch requires entries and deleted lists")
    known = {(episode["spec"]["id"], turn["index"]) for episode in review["episodes"] for turn in episode["turns"]}
    seen, retained = set(), []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"cluster_id", "opponent_strategy", "response_strategy", "reason", "evidence"}:
            raise ValueError("Invalid patch entry")
        ident = entry.get("cluster_id")
        if not isinstance(ident, str) or not _ID.fullmatch(ident) or ident in seen:
            raise ValueError("Invalid or duplicate cluster_id")
        seen.add(ident)
        _evidence(entry, known)
        description = abstract_opponent_strategy(
            _memory_text(entry.get("opponent_strategy"), "opponent_strategy", 60))
        response = abstract_response_strategy(
            _memory_text(entry.get("response_strategy"), "response_strategy", 40))
        retained.append((ident, description, response))
    for entry in deleted:
        if not isinstance(entry, dict) or set(entry) != {"cluster_id", "reason", "evidence"} or entry.get("cluster_id") not in prior or entry["cluster_id"] in seen:
            raise ValueError("Invalid or duplicate deletion")
        seen.add(entry["cluster_id"])
        _evidence(entry, known)
    if not set(prior).issubset(seen):
        raise ValueError("Every previous cluster must be retained or explicitly deleted")
    metadata = {"game": "llm_deliberation", "protocol": PROTOCOL,
        "mechanism": "tom_centroid_v1", "retrieval_backend": "cosine", "use_prototype": True,
        "retrieval_min_similarity": .74, "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_prefix": PREFIX, "batch": batch, "author": "current_codex_task",
        "source_bank_sha256": review["source_bank_sha256"], "review_sha256": _sha(review_path),
        "patch_sha256": _sha(patch_path), "patch_content_sha256": _digest(patch),
        "external_reflection": False}
    target = run_dir / "banks" / f"M{batch}.json"
    if target.exists():
        existing = load_bank(target)
        if existing["metadata"] != metadata or [(c["cluster_id"], c["opponent_strategy"], c["response_strategy"]) for c in existing["clusters"]] != retained:
            raise ValueError("Cannot replace committed memory")
        _immutable_json(run_dir / "reviews" / f"batch{batch}.applied.json", patch)
        return target
    changed = [description for ident, description, _ in retained if ident not in prior or prior[ident]["opponent_strategy"] != description]
    vectors = iter(())
    if changed:
        if encoder is None:
            raise ValueError("An encoder must be supplied when memory descriptions change")
        encoded = encoder.encode([PREFIX + description for description in changed])
        if len(encoded) != len(changed):
            raise ValueError("Encoder returned wrong number of vectors")
        vectors = iter([_vector(v) for v in encoded])
    clusters = []
    for ident, description, response in retained:
        vector = (deepcopy(prior[ident]["prototype_embedding"])
                  if ident in prior and prior[ident]["opponent_strategy"] == description else next(vectors))
        clusters.append({"cluster_id": ident, "opponent_strategy": description, "response_strategy": response,
            "centroid": vector, "centroid_size": 1, "prototype_strategy": description,
            "prototype_embedding": deepcopy(vector), "members": [{"opponent_strategy": description,
                                                                       "embedding": deepcopy(vector)}]})
    bank = _validate_bank({"schema_version": 7, "metadata": metadata, "clusters": clusters}, strict=True)
    _immutable_json(run_dir / "reviews" / f"batch{batch}.applied.json", patch)
    return _immutable_json(target, bank)
