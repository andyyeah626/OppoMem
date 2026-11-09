"""Full batch evidence keyed by the cluster actually returned during play.

Offline segmentation supplements evidence but cannot redirect a retrieval update.
Concrete actions belong in this transient evidence, never in persisted bank text.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from oppomem.memory.evolution import Segment, contrast_of, fallback_rounds


def build_retrieved_evidence(
    records: dict[int, dict[str, Any]], *,
    assigned_segments_by_cluster: dict[str, list[Segment]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return build_evidence-compatible payloads, with no trajectory sampling.

    Each target sees full episodes containing its retrievals or assigned segments.
    Only its actual retrieval rounds enter its injection contrast. Other rounds
    remain ordered context; withheld retrievals remain targets and carry advice,
    but that advice is never listed as guidance actually shown.
    """
    assigned = assigned_segments_by_cluster or {}
    relevant: dict[str, set[int]] = defaultdict(set)
    segment_rounds: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for cluster_id, segments in assigned.items():
        for segment in segments:
            if segment.run_index not in records:
                raise ValueError(f"Missing episode {segment.run_index} for {cluster_id}")
            relevant[cluster_id].add(segment.run_index)
            segment_rounds[cluster_id].update(
                (segment.run_index, int(index)) for index in segment.rounds
            )

    for run_index, record in records.items():
        turns = record["trace"]["turns"]
        indexes = [int(turn["round_index"]) for turn in turns]
        if len(indexes) != len(set(indexes)):
            raise ValueError(f"Duplicate trajectory round in episode {run_index}")
        for use in record["trace"].get("memory_uses", []):
            cluster_id = str(use["cluster_id"])
            if not cluster_id or int(use["round_index"]) not in indexes:
                raise ValueError(f"Invalid retrieval target or round in episode {run_index}")
            if not isinstance(use["selected"], bool):
                raise ValueError(f"Retrieval selected must be boolean in episode {run_index}")
            relevant[cluster_id].add(run_index)

    labels = {index: f"episode_{position + 1:02d}"
              for position, index in enumerate(sorted(records))}
    result: dict[str, dict[str, Any]] = {}
    for cluster_id in sorted(relevant):
        rows: list[dict[str, Any]] = []
        contrast_rows: list[dict[str, Any]] = []
        shown: list[str] = []
        focal_count = 0
        excluded_fallback = 0
        for run_index in sorted(relevant[cluster_id]):
            record = records[run_index]
            uses_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for use in record["trace"].get("memory_uses", []):
                uses_by_round[int(use["round_index"])].append(use)
            toms = {int(tom["round_index"]): tom
                    for tom in record["trace"].get("tom_chain", [])}
            fallback = fallback_rounds(record)
            for turn in sorted(record["trace"]["turns"], key=lambda row: int(row["round_index"])):
                index = int(turn["round_index"])
                uses = uses_by_round.get(index, [])
                target_uses = [use for use in uses if str(use["cluster_id"]) == cluster_id]
                if len(target_uses) > 1:
                    raise ValueError(f"Duplicate target retrieval in episode {run_index}, round {index}")
                actual_state = ("injected" if any(use["selected"] for use in uses)
                                else "withheld" if uses else "no_match")
                row = {
                    "episode": labels[run_index],
                    "source_run_index": run_index,
                    "round": index,
                    "self_action": turn["agent_action"],
                    "target_action": turn["opponent_action"],
                    "reward": turn["reward"],
                    "observation": turn.get("observation", ""),
                    "inference_context": turn.get("inference_context", ""),
                    "response_intent": turn.get("response_intent", ""),
                    "tom": deepcopy(toms.get(index)),
                    "memory_state": actual_state,
                    "retrieved_memory": deepcopy(uses),
                    "focal_retrieval": bool(target_uses),
                    "offline_assigned_round": (run_index, index) in segment_rounds[cluster_id],
                    "opponent_fallback": index in fallback,
                }
                rows.append(row)
                if not target_uses:
                    continue
                focal_count += 1
                target = target_uses[0]
                advice = str(target.get("response_strategy", ""))
                if target["selected"] and advice and advice not in shown:
                    shown.append(advice)
                if index in fallback:
                    excluded_fallback += 1
                    continue
                contrast_rows.append({
                    "reward": turn["reward"],
                    "memory_state": "injected" if target["selected"] else "withheld",
                })
        result[cluster_id] = {
            "evidence": rows,
            "evidence_rows_total": len(rows),
            "focal_retrieval_rows": focal_count,
            "assigned_segment_rows": sum(bool(row["offline_assigned_round"]) for row in rows),
            "excluded_fallback_rounds": excluded_fallback,
            "guidance_shown": shown,
            "injection_contrast": contrast_of(contrast_rows),
            "episode_labels": {str(index): labels[index] for index in sorted(relevant[cluster_id])},
            "contrast_scope": "Only actual retrieval rounds for this target cluster; descriptive, not causal.",
        }
    return result
