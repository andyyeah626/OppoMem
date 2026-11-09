"""Retained two-stage ToM prompts for non-RPS callers.

RPS uses core.rps_prompts directly. The dedicated user prompts here follow the
same concise style-level description, without prescribing detailed action rules.
"""
from __future__ import annotations

from oppomem.prompts.contracts import OPPONENT_MODEL_CONTRACT
from oppomem.types import UNKNOWN_OPPONENT_STRATEGY

_INSUFFICIENT = ("If the history provides no basis for an opponent description, use exactly:\n"
                 f"{UNKNOWN_OPPONENT_STRATEGY}")

TOM_NL_SYSTEM = "Write in English. Output plain text only, without code fences or labeled fields."

TOM_GAME_CONTEXT = (
    'Repeated simultaneous Rock Paper Scissors. Rock defeats Scissors, Scissors defeats Paper, '
    'and Paper defeats Rock; a win is +1, a draw 0, and a loss -1. Both players choose before '
    "either current-round action is revealed, so TARGET cannot condition the current action on "
    "SELF's current action. COMPLETED_TRAJECTORY lists finished rounds in order with SELF's action, "
    "TARGET's action, and SELF's reward. Legal actions are the exact strings Rock, Paper, and Scissors."
)

TOM_EXTRACT_SYSTEM = "Return exactly one JSON object with keys opponent_strategy and unresolved."


def trajectory_lines(turns, *, self_is_agent: bool = True) -> str:
    """Completed rounds in order, matching ``rps_action_protocol.trajectory_lines``."""
    if not turns:
        return '(none)'
    rows = []
    for turn in turns:
        if self_is_agent:
            self_action, target, reward = turn.agent_action, turn.opponent_action, turn.reward
        else:
            self_action, target, reward = turn.opponent_action, turn.agent_action, -turn.reward
        rows.append(f'round {turn.round_index + 1} | SELF {self_action} | TARGET {target} | SELF_reward {reward}')
    return '\n'.join(rows)


def tom_nl_prompt(turns, *, horizon: int, previous_read: str | None = None,
                  inference_context: str | None = None) -> str:
    """The natural-language read.  ``horizon`` replaces the original's fixed 20."""
    n = len(turns) + 1
    self_score = sum(turn.reward for turn in turns)
    previous = previous_read or 'None'
    return f'''=== GAME_CONTEXT ===
{inference_context or TOM_GAME_CONTEXT}

=== CURRENT_ROUND ===
Decision number: {n}
Completed rounds so far: {n - 1}
Remaining rounds including this choice: {horizon - n + 1}
SELF cumulative score: {self_score}
TARGET cumulative score: {-self_score}
Legal actions: Rock, Paper, Scissors
This block is for SELF. Do not treat it as information TARGET had before TARGET's past actions.

=== COMPLETED_TRAJECTORY ===
{trajectory_lines(turns, self_is_agent=True)}

=== PREVIOUS_OPPONENT_READ ===
{previous}

=== TASK ===
{OPPONENT_MODEL_CONTRACT}
TARGET is the opponent; SELF is our player. Use the previous read as context and
revise it when the completed history warrants a change.
Write 1-2 concise English sentences, without advice for SELF.
{_INSUFFICIENT}
Do not output JSON.'''


def tom_extract_prompt(turns, tom_text: str) -> str:
    return f'''=== COMPLETED_TRAJECTORY ===
{trajectory_lines(turns, self_is_agent=True)}

=== OPPONENT_READ_TEXT ===
{tom_text}

=== SCHEMA ===
Summarize OPPONENT_READ_TEXT faithfully; use the trajectory only to check accuracy.
opponent_strategy: a concise English description of TARGET's overall playing style
and strategy. Preserve its distinguishing meaning without adding specific
action-response rules or advice for SELF. If the text says evidence is insufficient,
use exactly: {UNKNOWN_OPPONENT_STRATEGY}
unresolved: a brief limitation stated in the text that could affect SELF's decision,
or null if none. Do not invent new claims.

Return exactly:
{{"opponent_strategy":"...","unresolved":null}}'''


__all__ = [
    "TOM_NL_SYSTEM",
    "TOM_EXTRACT_SYSTEM",
    "TOM_GAME_CONTEXT",
    "trajectory_lines",
    "tom_nl_prompt",
    "tom_extract_prompt",
]
