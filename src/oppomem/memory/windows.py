"""Extracted windows responsibilities of the working memory implementation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from oppomem.prompts.history import completed_joint_actions
from oppomem.prompts.contracts import OPPONENT_MODEL_CONTRACT, RESPONSE_CONTRACT, SYSTEM_JSON
from oppomem.models.semantics import cosine, encode_documents, encode_query
from oppomem.types import CentroidCluster, CentroidMatch, EpisodeTrace, UNKNOWN_OPPONENT_STRATEGY
from oppomem.memory.rps import abstract_opponent_strategy, abstract_response_strategy, sanitize_cluster_texts

from .common import _clean_text, _has_target_action, _valid_text


def normalize_windows(raw: object, turns: list[Any], *,
                      fixed_policy: bool = False,
                      tom_states: list[Any] | None = None,
                      abstract_actions: bool = False) -> tuple[list[dict], dict]:
    """Keep valid coarse windows; gaps and overlaps are deliberately untouched."""
    rows = raw.get("windows") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise ValueError("memory_windows must contain a windows list")
    available = sorted(turn.round_index for turn in turns)
    if not available:
        return [], {"available_rounds": [], "covered_rounds": [], "omitted_rounds": []}
    allowed = set(available)
    target_rounds = {turn.round_index for turn in turns if _has_target_action(turn.opponent_action)}
    informative_tom_rounds = (
        {state.round_index for state in tom_states
         if _valid_text(getattr(state, "opponent_strategy", ""))}
        if tom_states is not None else set(allowed)
    )
    lower, upper = available[0], available[-1]
    windows, dropped, clipped = [], [], []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            dropped.append({"index": index, "reason": "not_object"})
            continue
        opponent = _clean_text(row.get("opponent_strategy"))
        if abstract_actions:
            opponent = abstract_opponent_strategy(opponent)
        if not _valid_text(opponent):
            dropped.append({"index": index, "reason": "empty_or_placeholder_text"})
            continue
        ranges = [[lower, upper]] if fixed_policy else row.get("round_ranges")
        if not isinstance(ranges, list):
            dropped.append({"index": index, "reason": "round_ranges_not_list"})
            continue
        selected: set[int] = set()
        normalized_ranges = []
        for interval in ranges:
            if isinstance(interval, dict):
                interval = [interval.get("start"), interval.get("end")]
            if (not isinstance(interval, list) or len(interval) != 2
                    or any(type(point) is not int for point in interval)):
                continue
            start, end = interval
            if start > end:
                start, end = end, start
            clipped_start, clipped_end = max(lower, start), min(upper, end)
            if (clipped_start, clipped_end) != (start, end):
                clipped.append({"index": index, "range": interval,
                                "normalized": [clipped_start, clipped_end]
                                if clipped_start <= clipped_end else None})
            if clipped_start <= clipped_end:
                normalized_ranges.append([clipped_start, clipped_end])
                selected.update(r for r in allowed if clipped_start <= r <= clipped_end)
        selected &= target_rounds & informative_tom_rounds
        if not selected:
            dropped.append({"index": index, "reason": "no_target_action_in_window"})
            continue
        windows.append({
            "opponent_strategy": opponent,
            "round_ranges": normalized_ranges,
            "rounds": sorted(selected),
        })
    if fixed_policy and windows:
        discarded = windows[1:]
        windows = windows[:1]
        windows[0]["round_ranges"] = [[lower, upper]]
        windows[0]["rounds"] = sorted(target_rounds & informative_tom_rounds)
        dropped.extend({"index": index + 1, "reason": "fixed_policy_extra_window"}
                       for index, _ in enumerate(discarded))
    covered = sorted({r for window in windows for r in window["rounds"]})
    return windows, {
        "available_rounds": available,
        "target_action_rounds": sorted(target_rounds),
        "informative_tom_rounds": sorted(informative_tom_rounds),
        "covered_rounds": covered,
        "omitted_rounds": sorted(allowed - set(covered)),
        "overlapping_rounds": sorted(r for r in allowed
                                     if sum(r in w["rounds"] for w in windows) > 1),
        "clipped_membership": clipped,
        "dropped_windows": dropped,
    }
