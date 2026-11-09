"""Decision prompt and payload for direct centroid top-1 retrieval."""
from __future__ import annotations

import json

from oppomem.prompts.rps import RPS_GAMES, observation_prompt

from oppomem.prompts.history import COMPACT_HISTORY_GAMES, completed_joint_actions, without_embedded_history


def memory_payload(match):
    return {
        "opponent_strategy": match.opponent_strategy,
        "response_strategy": match.response_strategy,
    }


def decision_prompt(game, observation, history, legal_actions, tom, memory,
                    turns=None, *, inference_context: str = ""):
    if game in RPS_GAMES:
        return observation_prompt(observation, turns, tom=tom, memory=memory)
    del history  # The canonical completed trajectory below is the only history copy.
    return '''Choose one legal action to pursue the objective in CURRENT_STATE using
only player-visible information. CURRENT_STATE is authoritative for resources,
cards, actors, timing, and legal choices. COMPLETED_TRAJECTORY contains the exact
ordered public history. Respect the information boundary in GAME_CONTEXT and do not
invent hidden information.

The online tom and CENTROID_MEMORY are fallible interpretations, not additional
observations. Exact public history and the current state take precedence over both.
Use a remembered response only when its conditions are supported here. An unobserved
response must not be assumed certain merely because an interpretation suggests it.
Privately check the chosen action against the current round, the rules, and the
legal actions. Maximize our total reward across the remaining rounds; an immediate
gain need not improve the total return.

Information-gathering is optional. Use it only if the resulting observation could
change later actions and its expected benefit justifies the immediate cost.
Uncertainty alone is not a reason to experiment; an action already supported by the
evidence may be best. Once information is adequate, prioritize reward.

Also state the concise response strategy that the selected action is intended to
carry out under the current ToM. This is a pre-action intent for later reflection,
not a prediction, retrospective explanation, or restatement of the action. Use only
currently visible information and at most 20 words. The intent may pursue reward
even when the opponent's policy remains uncertain.

Return exactly {"action":"one final legal action","response_intent":"one concise
strategy intent"}. Do not include rationale, analysis, evidence, or extra keys. Do
not restate DATA or history.
DATA:\n''' + json.dumps({
        "game": game,
        "GAME_CONTEXT": inference_context,
        "tom": tom,
        "CENTROID_MEMORY": memory,
        "COMPLETED_TRAJECTORY": completed_joint_actions(turns),
        "CURRENT_STATE": {
            "observation": (without_embedded_history(observation)
                            if game in COMPACT_HISTORY_GAMES else observation),
            "legal_actions": legal_actions,
        },
    }, ensure_ascii=False)
