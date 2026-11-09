"""RPS-specific memory hygiene and situation features.

Concrete throws remain in trajectories and decision prompts.  Only the text that
can survive in a centroid bank is abstracted, so a bank cannot turn into a lookup
table for one fixed throw while the player can still reason from the real history.
"""
from __future__ import annotations

import re
from typing import Any, Iterable


RPS_ACTION_NAMES = ("Rock", "Paper", "Scissors")
RPS_RESPONSE_STRATEGY = "Play the corresponding counter action for the current situation."
_ACTION_RE = re.compile(r"\b(?:rock|paper|scissors?)\b|石头|石頭|剪刀|布", re.IGNORECASE)
_CYCLE_RE = re.compile(
    r"\b(?:rock|paper|scissors)\b(?:\s*(?:->|→|>|,|then|to|/)\s*"
    r"\b(?:rock|paper|scissors)\b){1,}(?:\s*(?:cycle|repeat(?:ing)?))?",
    re.IGNORECASE,
)
_FIXED_RE = re.compile(
    r"\b(?:fixed|constant|always|persistent|single[- ]action)\b[^.!?;]{0,32}"
    r"\b(?:rock|paper|scissors)(?:[- ]only)?\b",
    re.IGNORECASE,
)


def _clean(text: object) -> str:
    return " ".join(str(text or "").split()).strip()


def abstract_opponent_strategy(text: object) -> str:
    """Remove concrete RPS throw names while retaining behavioral structure."""
    value = _clean(text)
    if not value:
        return value
    value = _CYCLE_RE.sub("a recurring multi-action cycle", value)
    value = _FIXED_RE.sub("a fixed single-action preference", value)
    value = _ACTION_RE.sub("one action", value)
    value = re.sub(r"\b(one action\s*(?:[-/]\s*one action)+)\b",
                   "a multi-action pattern", value, flags=re.IGNORECASE)
    return _clean(value)


def abstract_response_strategy(text: object = None) -> str:
    """Preserve useful abstract advice; never fabricate or overwrite a response."""
    value = _clean(text)
    if not value:
        raise ValueError("RPS response_strategy must not be empty")
    if has_concrete_action_name(value):
        raise ValueError("RPS response_strategy contains a concrete action name")
    return value


def has_concrete_action_name(text: object) -> bool:
    return bool(_ACTION_RE.search(str(text or "")))


def situation_query(turns: Iterable[Any], *, round_index: int | None = None) -> str:
    """Build a transient, situation-dependent retrieval suffix.

    This is intentionally never stored in a cluster.  It lets cosine retrieval
    distinguish otherwise similar ToM descriptions using the live trajectory.
    """
    rows = list(turns or [])
    if not rows:
        return "round_phase=opening; no_completed_rounds"
    recent = rows[-4:]
    actions = ",".join(
        f"self:{getattr(row, 'agent_action', '')}/opp:{getattr(row, 'opponent_action', '')}/"
        f"reward:{getattr(row, 'reward', 0):g}" for row in recent
    )
    opponent_actions = [str(getattr(row, "opponent_action", "")) for row in rows]
    repeat = 1
    for index in range(len(opponent_actions) - 2, -1, -1):
        if opponent_actions[index] != opponent_actions[-1]:
            break
        repeat += 1
    phase = "opening" if len(rows) < 4 else "middle" if len(rows) < 14 else "late"
    return (f"round_phase={phase}; completed={len(rows)}; opponent_repeat_streak={repeat}; "
            f"recent_trajectory={actions}; query_round={round_index if round_index is not None else len(rows)}")


def sanitize_cluster_texts(cluster: Any) -> dict[str, bool]:
    """Sanitize one cluster in place and report which fields changed."""
    changed: dict[str, bool] = {}
    for field in ("opponent_strategy", "prototype_strategy"):
        value = getattr(cluster, field, None)
        if value:
            replacement = abstract_opponent_strategy(value)
            changed[field] = replacement != value
            setattr(cluster, field, replacement)
    response = abstract_response_strategy(getattr(cluster, "response_strategy", None))
    changed["response_strategy"] = response != cluster.response_strategy
    cluster.response_strategy = response
    members = getattr(cluster, "members", None)
    if members:
        for row in members:
            if isinstance(row, dict) and row.get("opponent_strategy"):
                value = row["opponent_strategy"]
                row["opponent_strategy"] = abstract_opponent_strategy(value)
        changed["members"] = True
    return changed


__all__ = [
    "RPS_ACTION_NAMES",
    "RPS_RESPONSE_STRATEGY",
    "abstract_opponent_strategy",
    "abstract_response_strategy",
    "has_concrete_action_name",
    "sanitize_cluster_texts",
    "situation_query",
]
