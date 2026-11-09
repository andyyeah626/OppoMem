"""Batch-granular memory evolution: segment a batch, pool it, reflect, commit.

`CentroidMemoryLearner` updates the bank at the end of every episode.  That is the
right granularity for a single training pass but the wrong one for generational
evolution, for two reasons:

  * The bank the agent plays against changes under it mid-run, so there is no window
    in which a fixed bank can be measured.  Here the bank is frozen for a whole batch,
    which makes the batch a clean measurement unit -- and makes the episodes of a
    batch embarrassingly parallel, since nothing writes to the bank while they run.
  * Assignment is decided from one episode's segments at a time, so a cluster boundary
    is drawn on the first two or three segments that ever arrive.  Pooling a batch's
    segments before assigning lets the same greedy rule see 95 episodes at once.

Nothing here replays a game.  `benchmarks/runner.py` serializes turns and tom_chain
with `dataclasses.asdict`, so a saved record rebuilds into the exact `EpisodeTrace`
the learner would have seen, and the whole update phase runs off disk.

The merge gate is deliberately not reimplemented: `PooledAssigner` holds a real
`CentroidMemoryLearner` and calls its helpers, so the batch gate cannot drift away
from the online gate that produced every earlier bank.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import statistics
import threading
from typing import Any, Callable, Iterator

import numpy as np

from oppomem.memory import CentroidMemoryBank, CentroidMemoryLearner, MECHANISM, MEMORY_SYSTEM, REFLECTION_SYSTEM, batch_reflection_prompt, cluster_members, mean_embedding, medoid_index, members_payload, normalize_windows, retain_closest_members, window_prompt
from oppomem.models.semantics import cosine, encode_documents
from oppomem.types import CentroidCluster, EpisodeTrace, ToMState, Turn
from oppomem.memory.rps import abstract_opponent_strategy, abstract_response_strategy


EMBED_PREFIX = "Opponent strategy: "
MEMORY_STATES = ("injected", "withheld", "no_match")


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def rebuild_trace(record: dict[str, Any]) -> EpisodeTrace:
    """Reconstruct the trace object the learner would have seen, from a saved record."""
    trace = record["trace"]
    return EpisodeTrace(
        episode_id=str(trace["episode_id"]),
        game=str(record.get("game", "open_spiel_roshambo")),
        variant=str(record.get("variant", "tom_memory")),
        turns=[Turn(**row) for row in trace["turns"]],
        tom_chain=[ToMState(**row) for row in trace["tom_chain"]],
    )


# ---- one LLM client per worker slot ---------------------------------------------------

class LLMPool:
    """Lease-based pool of single-threaded LLM clients.

    `OpenAIJsonLLM` is not thread-safe: `requests_made += 1` is a read-modify-write and
    the journal's `attempt_id` is derived from it, so sharing one client across threads
    would corrupt the request audit rather than merely miscount.  Giving each slot its
    own client and its own journal file removes the shared state entirely, and mirrors
    the per-worker journal convention the play phase already uses.
    """

    def __init__(self, factory: Callable[[int], Any], size: int) -> None:
        if size < 1:
            raise ValueError("pool size must be positive")
        self.clients = [factory(index) for index in range(size)]
        self._free: queue.SimpleQueue = queue.SimpleQueue()
        for client in self.clients:
            self._free.put(client)

    @contextmanager
    def lease(self) -> Iterator[Any]:
        client = self._free.get()
        try:
            yield client
        finally:
            self._free.put(client)

    @property
    def requests_made(self) -> int:
        return sum(int(client.requests_made) for client in self.clients)

    def usage_events(self) -> list[dict[str, Any]]:
        return [event.to_dict() for client in self.clients for event in client.events]


# ---- segmentation ---------------------------------------------------------------------

@dataclass
class Segment:
    run_index: int
    cycle: int
    opponent_id: str
    segment_index: int
    opponent_strategy: str
    round_ranges: list[list[int]]
    rounds: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "cycle": self.cycle,
            "opponent_id": self.opponent_id,
            "segment_index": self.segment_index,
            "opponent_strategy": self.opponent_strategy,
            "round_ranges": self.round_ranges,
            "rounds": self.rounds,
        }


def segment_batch(records: list[dict[str, Any]], pool: LLMPool, *, cache_dir: Path,
                  max_workers: int = 8, fixed_policy: bool = False,
                  abstract_actions: bool = False,
                  progress: Callable[[str], None] | None = None,
                  ) -> tuple[list[Segment], dict[str, Any]]:
    """Segment every episode of the batch, in parallel, with a per-episode cache.

    Parallel because it has to be: the segmenter runs ~60 s per episode (measured, see
    experiments/rps_evolve19/preflight), so 95 sequential calls would be ~95 min per
    batch and ~32 h over the 20 batches of the run.

    Cached per episode rather than per batch because an interrupted update must not
    re-pay the calls it already made, and because one file per episode is the only
    write pattern that needs no lock across the worker threads.

    A failure is allowed to propagate.  Silently dropping an episode's segments would
    lose evidence invisibly; raising costs one episode's rework on resume.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(run_index: int) -> Path:
        return cache_dir / f"{int(run_index):04d}.json"

    pending = [row for row in records if not cache_path(row["run_index"]).is_file()]
    done = threading.Lock()
    completed = [0]

    def work(record: dict[str, Any]) -> None:
        trace = rebuild_trace(record)
        with pool.lease() as llm:
            raw = llm.complete_json(
                stage="memory_windows", system=MEMORY_SYSTEM,
                prompt=window_prompt(trace, fixed_policy=fixed_policy,
                                     abstract_actions=abstract_actions),
                max_tokens=600,
            )
        windows, normalization = normalize_windows(
            raw, trace.turns, fixed_policy=fixed_policy, tom_states=trace.tom_chain,
            abstract_actions=abstract_actions,
        )
        save_json(cache_path(record["run_index"]), {
            "run_index": int(record["run_index"]),
            "cycle": int(record["cycle"]),
            "opponent_id": str(record["opponent_id"]),
            "raw_response": raw,
            "parsed_windows": windows,
            "normalization": normalization,
        })
        with done:
            completed[0] += 1
            if progress:
                progress(f"segmented {completed[0]}/{len(pending)} "
                         f"ep{record['run_index']:04d} {record['opponent_id']} "
                         f"windows={len(windows)}")

    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as pool_executor:
            futures = [pool_executor.submit(work, row) for row in pending]
            for future in as_completed(futures):
                future.result()

    segments: list[Segment] = []
    audits = []
    dropped: dict[str, int] = {}
    # Fixed order, by (run_index, segment_index).  Greedy assignment is order-dependent,
    # so the order has to be pinned by the schedule rather than by thread completion.
    for record in sorted(records, key=lambda row: int(row["run_index"])):
        payload = json.loads(cache_path(record["run_index"]).read_text(encoding="utf-8"))
        audits.append(payload)
        for reason_row in payload["normalization"].get("dropped_windows", []):
            reason = str(reason_row.get("reason"))
            dropped[reason] = dropped.get(reason, 0) + 1
        for index, window in enumerate(payload["parsed_windows"]):
            segments.append(Segment(
                run_index=int(payload["run_index"]),
                cycle=int(payload["cycle"]),
                opponent_id=str(payload["opponent_id"]),
                segment_index=index,
                opponent_strategy=str(window["opponent_strategy"]),
                round_ranges=[list(pair) for pair in window["round_ranges"]],
                rounds=[int(value) for value in window["rounds"]],
            ))
    return segments, {
        "episodes": len(records),
        "newly_segmented": len(pending),
        "segments": len(segments),
        "episodes_with_zero_windows": sum(
            1 for row in audits if not row["parsed_windows"]),
        "dropped_window_reasons": dropped,
        "per_episode": audits,
    }


# ---- pooled assignment ----------------------------------------------------------------

@dataclass
class Proposal:
    cluster_id: str
    opponent_strategy: str
    previous_response_strategy: str | None
    centroid: np.ndarray
    centroid_size: int
    base_cluster: CentroidCluster | None
    segments: list[Segment] = field(default_factory=list)
    prototype_vectors: list[np.ndarray] = field(default_factory=list)
    # Member mode only: the retained (text, vector) ToM set, and the medoid member
    # that both `opponent_strategy` and the retrieval prototype are set from.
    members: list[tuple[str, np.ndarray]] = field(default_factory=list)
    prototype: tuple[str, np.ndarray] | None = None


class PooledAssigner:
    """The online merge gate, applied to a whole batch's segments at once."""

    def __init__(self, bank: CentroidMemoryBank, embedder, nli, *, game: str, scope: str,
                 similarity_threshold: float = .82,
                 contradiction_threshold: float = .70,
                 max_centroid_size: int | None = None,
                 max_cluster_members: int | None = None) -> None:
        self.bank = bank
        self.embedder = embedder
        # A learner instance purely for its gate helpers (_key, _nli_contradictions,
        # _cluster_id).  Reaching for them deliberately: reimplementing the bidirectional
        # NLI check here would let the batch gate silently diverge from the online one.
        # The llm slot is unused -- assignment makes no LLM calls -- so it stays None.
        self._gate = CentroidMemoryLearner(
            bank, None, embedder, nli, game, scope,
            similarity_threshold=similarity_threshold,
            contradiction_threshold=contradiction_threshold,
            max_centroid_size=max_centroid_size,
            max_cluster_members=max_cluster_members,
            fixed_policy=False,
        )
        self.similarity_threshold = float(similarity_threshold)
        self.contradiction_threshold = float(contradiction_threshold)
        self.max_centroid_size = (
            int(max_centroid_size) if max_centroid_size is not None else None
        )
        self.max_cluster_members = (
            int(max_cluster_members) if max_cluster_members is not None else None
        )
        self.store_members = self.max_cluster_members is not None

    def assign(self, segments: list[Segment],
               ) -> tuple[dict[str, Proposal], list[dict[str, Any]]]:
        proposals: dict[str, Proposal] = {
            cluster_id: Proposal(
                cluster_id=cluster_id,
                opponent_strategy=cluster.opponent_strategy,
                previous_response_strategy=cluster.response_strategy,
                centroid=np.asarray(cluster.centroid, dtype=np.float32),
                # The bank's historical support, so the running mean keeps accumulating
                # across generations instead of restarting each batch.
                centroid_size=int(cluster.centroid_size),
                base_cluster=cluster,
                prototype_vectors=[],
                members=(cluster_members(cluster) if self.store_members else []),
            )
            for cluster_id, cluster in self.bank.clusters.items()
        }
        next_ordinal = len(proposals)
        operations: list[dict[str, Any]] = []

        # One embedding call for the whole batch rather than one per segment.
        vectors = (encode_documents(self.embedder,
            [EMBED_PREFIX + segment.opponent_strategy for segment in segments])
            if segments else [])

        for segment, vector in zip(segments, vectors):
            vector = np.asarray(vector, dtype=np.float32)
            candidates = sorted(
                ((cosine(vector, proposal.centroid), proposal)
                 for proposal in proposals.values()),
                key=lambda item: (-item[0], item[1].cluster_id),
            )
            eligible = [(score, proposal) for score, proposal in candidates
                        if score >= self.similarity_threshold]
            contradictions = self._gate._nli_contradictions(
                segment.opponent_strategy,
                [(score, {"opponent_strategy": proposal.opponent_strategy})
                 for score, proposal in eligible],
            )
            selected: Proposal | None = None
            selected_similarity = None
            selected_contradiction = None
            nli_candidates = []
            for (similarity, proposal), contradiction in zip(eligible, contradictions):
                nli_candidates.append({
                    "cluster_id": proposal.cluster_id,
                    "cosine": float(similarity),
                    "contradiction": float(contradiction),
                })
                if contradiction >= self.contradiction_threshold:
                    operations.append({
                        "operation": "nli_rejected",
                        "cluster_id": proposal.cluster_id,
                        "run_index": segment.run_index,
                        "segment_index": segment.segment_index,
                        "cosine": float(similarity),
                        "contradiction": float(contradiction),
                    })
                    continue
                if (not self.store_members and self.max_centroid_size is not None
                        and int(proposal.centroid_size) >= self.max_centroid_size):
                    operations.append({
                        "operation": "size_cap_rejected",
                        "cluster_id": proposal.cluster_id,
                        "run_index": segment.run_index,
                        "segment_index": segment.segment_index,
                        "cosine": float(similarity),
                        "centroid_size": int(proposal.centroid_size),
                    })
                    continue
                selected = proposal
                selected_similarity = float(similarity)
                selected_contradiction = float(contradiction)
                break

            if selected is None:
                cluster_id = self._gate._cluster_id(segment.opponent_strategy, next_ordinal)
                while cluster_id in proposals:
                    next_ordinal += 1
                    cluster_id = self._gate._cluster_id(
                        segment.opponent_strategy, next_ordinal)
                next_ordinal += 1
                selected = Proposal(
                    cluster_id=cluster_id,
                    opponent_strategy=segment.opponent_strategy,
                    previous_response_strategy=None,
                    centroid=vector,
                    centroid_size=0,
                    base_cluster=None,
                    prototype_vectors=[],
                    members=[],
                )
                proposals[cluster_id] = selected
                assignment = "new"
            else:
                assignment = "existing"

            if self.store_members:
                # Same rule as the online learner: the centroid is the mean of the
                # retained members, so it is recomputed after the eviction rather than
                # carried as a running mean that would still count the dropped member.
                selected.members.append((segment.opponent_strategy,
                                         np.asarray(vector, dtype=np.float32)))
                selected.members = retain_closest_members(
                    selected.members, self.max_cluster_members)
                selected.centroid = mean_embedding(
                    [member for _, member in selected.members])
                selected.centroid_size = len(selected.members)
                medoid = medoid_index(selected.members, selected.centroid)
                selected.opponent_strategy = selected.members[medoid][0]
                selected.prototype = selected.members[medoid]
            else:
                old_size = int(selected.centroid_size)
                selected.centroid = (
                    (selected.centroid * old_size + vector) / (old_size + 1)
                    if old_size else vector
                )
                selected.centroid_size = old_size + 1
                selected.prototype_vectors.append(np.asarray(vector, dtype=np.float32))
            selected.segments.append(segment)
            operations.append({
                "operation": "assigned",
                "assignment": assignment,
                "cluster_id": selected.cluster_id,
                "run_index": segment.run_index,
                "opponent_id": segment.opponent_id,
                "segment_index": segment.segment_index,
                "cosine": selected_similarity,
                "contradiction": selected_contradiction,
                "nli_candidates": nli_candidates,
            })
        return proposals, operations


# ---- evidence, with the injected/withheld contrast ------------------------------------

def memory_states(record: dict[str, Any]) -> dict[int, str]:
    """Per-round memory state, read straight off the recorded MemoryUse rows.

    `selected` is what the dropout writes through: a retrieved-but-withheld round keeps
    its MemoryUse row with selected=False, so the control arm is recoverable from the
    record without any side channel.
    """
    states: dict[int, str] = {}
    for use in record["trace"].get("memory_uses", []):
        states[int(use["round_index"])] = "injected" if use["selected"] else "withheld"
    return states


def injected_texts(record: dict[str, Any]) -> dict[int, str]:
    return {int(use["round_index"]): str(use["response_strategy"])
            for use in record["trace"].get("memory_uses", [])
            if use["selected"]}


def fallback_rounds(record: dict[str, Any]) -> set[int]:
    """Rounds where the LLM opponent fell back to a uniform random action.

    Those rounds carry no information about the opponent's policy -- the fallback is
    deliberately unbiased precisely so a failure cannot masquerade as a copybot -- so
    they must not enter either the A/B contrast or the reflection evidence.
    """
    rounds = set()
    for transition in record.get("transitions", []):
        metrics = transition.get("evaluator_metrics") or {}
        if metrics.get("opponent_fallback"):
            rounds.add(int(transition["step"]))
    return rounds


def contrast_of(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for state in MEMORY_STATES:
        rewards = [float(row["reward"]) for row in rows if row["memory_state"] == state]
        result[state] = {
            "rounds": len(rewards),
            "mean_reward": round(statistics.fmean(rewards), 4) if rewards else None,
        }
    injected = result["injected"]["mean_reward"]
    withheld = result["withheld"]["mean_reward"]
    result["injected_minus_withheld"] = (
        round(injected - withheld, 4)
        if injected is not None and withheld is not None else None
    )
    return result


def build_evidence(segments: list[Segment], records: dict[int, dict[str, Any]], *,
                   maximum_rows: int = 160) -> dict[str, Any]:
    """Pool one cluster's assigned rounds into anonymized trajectory evidence."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    guidance: list[str] = []
    labels: dict[int, str] = {}
    excluded_fallback = 0

    for segment in segments:
        record = records[segment.run_index]
        if segment.run_index not in labels:
            labels[segment.run_index] = f"episode_{len(labels) + 1:02d}"
        turns = {int(turn["round_index"]): turn for turn in record["trace"]["turns"]}
        states = memory_states(record)
        shown = injected_texts(record)
        skipped = fallback_rounds(record)
        for round_index in segment.rounds:
            key = (segment.run_index, int(round_index))
            if key in seen or int(round_index) not in turns:
                continue
            if int(round_index) in skipped:
                excluded_fallback += 1
                continue
            seen.add(key)
            turn = turns[int(round_index)]
            state = states.get(int(round_index), "no_match")
            if state == "injected" and shown.get(int(round_index)):
                text = shown[int(round_index)]
                if text not in guidance:
                    guidance.append(text)
            rows.append({
                "episode": labels[segment.run_index],
                "round": int(round_index),
                "self_action": turn["agent_action"],
                "target_action": turn["opponent_action"],
                "reward": turn["reward"],
                "response_intent": turn.get("response_intent", ""),
                "memory_state": state,
            })

    # The contrast is computed over ALL rows, then the rows themselves are thinned for
    # the prompt.  Sampling first would make the numbers the model is shown disagree
    # with the numbers the analysis reports.
    contrast = contrast_of(rows)
    sampled = rows
    if len(rows) > maximum_rows:
        indexes = np.linspace(0, len(rows) - 1, maximum_rows, dtype=int)
        sampled = [rows[int(index)] for index in indexes]
    return {
        "evidence": sampled,
        "evidence_rows_total": len(rows),
        "excluded_fallback_rounds": excluded_fallback,
        "guidance_shown": guidance,
        "injection_contrast": contrast,
        "episode_labels": {str(key): value for key, value in labels.items()},
    }


def medoid_strategy(segments: list[Segment], centroid: np.ndarray, embedder) -> str:
    """The member segment closest to the centroid, verbatim.

    The description the clustering produced is the description that gets stored; no
    LLM rewrite, because a rewrite's embedding is what turned retrieval into a style
    match in the washed banks.
    """
    if not segments:
        raise ValueError("cannot take a medoid of an empty cluster")
    if len(segments) == 1:
        return segments[0].opponent_strategy
    vectors = encode_documents(embedder,
        [EMBED_PREFIX + segment.opponent_strategy for segment in segments])
    best = max(range(len(segments)),
               key=lambda index: cosine(np.asarray(vectors[index], dtype=np.float32),
                                        centroid))
    return segments[best].opponent_strategy


# ---- reflection and commit ------------------------------------------------------------

def reflect_all(payloads: dict[str, dict[str, Any]], pool: LLMPool, *,
                partial_path: Path, max_workers: int = 8,
                abstract_actions: bool = False,
                progress: Callable[[str], None] | None = None,
                ) -> dict[str, dict[str, Any]]:
    """One reflection call per touched cluster, in parallel, resumable.

    Parallel for the same reason segmentation is, and safe because a reflection is a
    pure function of one cluster's own payload: clusters do not read each other.  The
    bank is written afterwards, serially, in a fixed order.
    """
    results: dict[str, dict[str, Any]] = {}
    if partial_path.is_file():
        results.update(json.loads(partial_path.read_text(encoding="utf-8")))
    if abstract_actions:
        for cluster_id in list(results):
            try:
                abstract_response_strategy(results[cluster_id].get("response_strategy"))
            except ValueError:
                del results[cluster_id]
    pending = [cluster_id for cluster_id in sorted(payloads)
               if cluster_id not in results]
    lock = threading.Lock()

    def work(cluster_id: str) -> None:
        payload = payloads[cluster_id]
        prompt = batch_reflection_prompt(
            canonical_opponent_strategy=payload["canonical_opponent_strategy"],
            previous_response_strategy=payload["previous_response_strategy"],
            support_segments=payload["support_segments"],
            evidence=payload["evidence"], guidance_shown=payload["guidance_shown"],
            injection_contrast=payload["injection_contrast"],
            abstract_actions=abstract_actions,
            reflection_phase=payload.get("reflection_phase", "bootstrap"),
            retrieved_evidence=payload.get("retrieved_evidence"),
            full_trajectories=payload.get("full_trajectories"),
        )
        attempts = []
        for attempt in range(3 if abstract_actions else 1):
            with pool.lease() as llm:
                raw = llm.complete_json(stage="memory_reflection", system=REFLECTION_SYSTEM,
                                        prompt=prompt, max_tokens=300)
            response = " ".join(str(raw.get("response_strategy") or "").split()).strip()
            attempts.append({"attempt": attempt + 1, "raw_response": raw})
            try:
                if abstract_actions:
                    response = abstract_response_strategy(response)
                break
            except ValueError as exc:
                attempts[-1]["validation_error"] = str(exc)
                # Retain failed requests for inspection without treating them as a bank.
                with lock:
                    save_json(partial_path.with_name(
                        partial_path.stem + "." + cluster_id + ".attempts.json"), attempts)
                if attempt == 2:
                    raise ValueError(f"Invalid RPS reflection for {cluster_id} after 3 attempts") from exc
                prompt += ("\n\nCORRECTION REQUIRED: Your last response was invalid: "
                           + json.dumps(raw, ensure_ascii=False)
                           + ". Return a nonempty executable abstract strategy without concrete "
                           "throw names in English or Chinese. Preserve the useful decision rule.")
        with lock:
            results[cluster_id] = {"response_strategy": response, "raw_response": raw,
                                   "reflection_attempts": attempts}
            save_json(partial_path, results)
            if progress:
                progress(f"reflected {len(results)}/{len(payloads)} {cluster_id} {response[:70]}")

    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(work, cluster_id) for cluster_id in pending]
            for future in as_completed(futures):
                future.result()
    return results


def commit(bank: CentroidMemoryBank, proposals: dict[str, Proposal],
           reflections: dict[str, dict[str, Any]], embedder, *,
           metadata: dict[str, Any],
           contrast_by_cluster: dict[str, dict[str, Any]] | None = None,
           demote_on_nonpositive_contrast: bool = True,
           demote_min_injected_rounds: int = 20,
           store_members: bool | None = None,
           abstract_rps: bool = False) -> list[dict[str, Any]]:
    """Write the reflected clusters back, in a fixed order.

    A cluster whose reflection came back empty is not committed at all -- not even its
    centroid.  The online learner has the same rule, and it matters: committing the
    centroid without the text would leave a cluster that retrieval can match but whose
    stored advice is stale, which is exactly the failure the response strategy is
    supposed to prevent.

    ``store_members`` selects the member mode written by ``PooledAssigner`` when it was
    given a ``max_cluster_members``.  It cannot be read off the proposals -- a legacy
    ``Proposal.members`` is an empty list, which is also what member mode looks like on a
    cluster with no retained text -- so it is an explicit flag, defaulting to the bank's
    own ``max_cluster_members`` metadata.  It MUST agree with the assigner that produced
    the proposals.
    """
    if store_members is None:
        store_members = bank.metadata.get("max_cluster_members") is not None
    if abstract_rps:
        # Fail before modifying any bank object if a reflection is invalid.
        for cluster_id, proposal in proposals.items():
            if proposal.segments or cluster_id in reflections:
                abstract_response_strategy(reflections.get(cluster_id, {}).get("response_strategy"))
    operations: list[dict[str, Any]] = []
    for cluster_id in sorted(proposals):
        proposal = proposals[cluster_id]
        if not proposal.segments:
            if cluster_id in reflections and proposal.base_cluster is not None:
                response = str(reflections[cluster_id].get("response_strategy", "")).strip()
                if abstract_rps:
                    response = abstract_response_strategy(response)
                if response:
                    proposal.base_cluster.response_strategy = response
                    operations.append({"operation": "response_updated", "cluster_id": cluster_id,
                                       "segment_count": 0, "centroid_size": proposal.base_cluster.centroid_size})
            continue
        response = str(reflections.get(cluster_id, {}).get("response_strategy", "")).strip()
        if abstract_rps:
            response = abstract_response_strategy(response)
        if not response:
            operations.append({
                "operation": "reflection_rejected",
                "cluster_id": cluster_id,
                "segment_count": len(proposal.segments),
            })
            continue
        demoted = False
        contrast = (contrast_by_cluster or {}).get(cluster_id, {})
        injected_rounds = int((contrast.get("injected") or {}).get("rounds") or 0)
        delta: float | None = contrast.get("injected_minus_withheld")
        if delta is not None:
            delta = float(delta)
        if (demote_on_nonpositive_contrast
                and injected_rounds >= demote_min_injected_rounds
                and delta is not None
                and float(delta) <= 0.0
                and proposal.previous_response_strategy):
            response = str(proposal.previous_response_strategy).strip()
            demoted = True
        centroid = proposal.centroid.astype(np.float32)
        member_rows = (members_payload(proposal.members)
                       if store_members else None)
        if abstract_rps and member_rows:
            for member in member_rows:
                member["opponent_strategy"] = abstract_opponent_strategy(
                    member["opponent_strategy"]
                )
        if proposal.base_cluster is None:
            # A new cluster gets the medoid of the segments it was born with, not the
            # arbitrary first one.  The online learner has to use the first segment --
            # it only ever sees one episode at a time -- but a batch can afford the
            # better description, which is what the offline recluster already does.
            # In member mode the medoid of the retained set is already in the proposal.
            if store_members:
                prototype_strategy, prototype_vector = proposal.prototype
                prototype_embedding = np.asarray(
                    prototype_vector, dtype=np.float32).tolist()
            else:
                prototype_index = max(
                    range(len(proposal.prototype_vectors)),
                    key=lambda i: cosine(proposal.prototype_vectors[i], centroid),
                ) if proposal.prototype_vectors else None
                prototype_strategy = (proposal.segments[prototype_index].opponent_strategy
                                      if prototype_index is not None else None)
                prototype_embedding = (proposal.prototype_vectors[prototype_index].tolist()
                                       if prototype_index is not None else None)
            bank.clusters[cluster_id] = CentroidCluster(
                cluster_id=cluster_id,
                opponent_strategy=(
                    (abstract_opponent_strategy(proposal.opponent_strategy)
                     if abstract_rps else proposal.opponent_strategy)
                    if store_members else medoid_strategy(
                        proposal.segments, centroid, embedder)
                ),
                response_strategy=response,
                centroid=centroid.tolist(),
                centroid_size=int(proposal.centroid_size),
                prototype_strategy=(abstract_opponent_strategy(prototype_strategy)
                                    if abstract_rps and prototype_strategy else prototype_strategy),
                prototype_embedding=prototype_embedding,
                members=member_rows,
            )
            operation = "created"
        else:
            base = proposal.base_cluster
            base.centroid = centroid.tolist()
            base.centroid_size = int(proposal.centroid_size)
            base.response_strategy = response
            if store_members:
                # The canonical ToM follows the medoid member, so the stored
                # description, the NLI gate, and the retrieval prototype all name the
                # same member instead of drifting apart as the set is re-weighted.
                prototype_strategy, prototype_vector = proposal.prototype
                base.opponent_strategy = (abstract_opponent_strategy(proposal.opponent_strategy)
                                          if abstract_rps else proposal.opponent_strategy)
                base.prototype_strategy = (abstract_opponent_strategy(prototype_strategy)
                                           if abstract_rps and prototype_strategy else prototype_strategy)
                base.prototype_embedding = np.asarray(
                    prototype_vector, dtype=np.float32).tolist()
                base.members = member_rows
            elif proposal.prototype_vectors:
                prototype_index = max(
                    range(len(proposal.prototype_vectors)),
                    key=lambda i: cosine(proposal.prototype_vectors[i], centroid),
                )
                base.prototype_strategy = proposal.segments[prototype_index].opponent_strategy
                base.prototype_embedding = proposal.prototype_vectors[prototype_index].tolist()
            # Outside member mode `opponent_strategy` is deliberately left alone, exactly
            # as the online learner leaves it.  A medoid over only this batch's segments
            # would ignore every earlier segment, and a medoid over all of them is not
            # computable: the bank stores a centroid and a count, never the member texts.
            # So the evolution lives in the centroid and the response strategy, and the
            # cluster's stored ToM stays the one its evidence originally supported.
            operation = "merged"
        operations.append({
            "operation": "demoted_kept_prior" if demoted else operation,
            "cluster_id": cluster_id,
            "segment_count": len(proposal.segments),
            "centroid_size": int(proposal.centroid_size),
            "injection_contrast_delta": delta,
        })
    bank.metadata = {**bank.metadata, "mechanism": MECHANISM, **metadata}
    bank.save()
    return operations
