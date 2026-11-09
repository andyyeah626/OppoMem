from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any, Iterable


COMPACT_HISTORY_GAMES = {
    "open_spiel_roshambo",
    "repeated_rps",
    "axelrod_ipd",
    "iterated_pd",
}


def completed_joint_actions(turns: Iterable[Any] | None) -> list[dict[str, Any]]:
    return [{
        "round": turn.round_index,
        "self_action": turn.agent_action,
        "other_action": turn.opponent_action,
        "reward": turn.reward,
    } for turn in (turns or [])]


def action_statistics(turns: Iterable[Any] | None) -> dict[str, Any]:
    """Compact visible statistics, including one-step conditional evidence."""
    rows = list(turns or [])
    other = Counter(turn.opponent_action for turn in rows)
    own = Counter(turn.agent_action for turn in rows)
    by_previous_self: dict[str, Counter] = defaultdict(Counter)
    by_previous_other: dict[str, Counter] = defaultdict(Counter)
    for previous, current in zip(rows, rows[1:]):
        by_previous_self[previous.agent_action][current.opponent_action] += 1
        by_previous_other[previous.opponent_action][current.opponent_action] += 1
    return {
        "completed_rounds": len(rows),
        "other_action_counts": dict(other),
        "self_action_counts": dict(own),
        "other_action_by_previous_self_action": {
            action: dict(counts) for action, counts in sorted(by_previous_self.items())
        },
        "other_action_by_previous_other_action": {
            action: dict(counts) for action, counts in sorted(by_previous_other.items())
        },
    }


def without_embedded_history(observation: str) -> Any:
    """Remove public history already represented by completed_joint_actions."""
    try:
        parsed = json.loads(observation)
    except (TypeError, ValueError):
        return observation
    if not isinstance(parsed, dict):
        return parsed
    compact = dict(parsed)
    compact.pop("completed_joint_actions", None)
    # Training exploration belongs in evaluator configuration, never in the
    # model-visible game state or policy-inference context.
    compact.pop("training_objective", None)
    return compact
