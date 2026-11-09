"""Leakage-resistant fixed-policy induction prompts."""
from __future__ import annotations

import json

from oppomem.prompts.rps import RPS_GAMES, tom_prompt_text
from oppomem.prompts.contracts import OPPONENT_MODEL_CONTRACT

from oppomem.prompts.history import completed_joint_actions, without_embedded_history


PROMPT_VARIANTS = {"A", "B", "C"}


def _variant_instructions(variant: str) -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown ToM prompt variant {variant!r}")
    revision = ""
    perspective = ""
    if variant in {"B", "C"}:
        revision = '''
Compare PREVIOUS_HYPOTHESIS with the newly completed evidence. Keep it if it
still explains the complete trajectory, refine it only where the evidence
requires, and replace it when contradicted. Do not change wording merely
because another description sounds plausible.
'''
    if variant == "C":
        perspective = '''
Before inferring the policy, privately reconstruct what information was
available to the target before each action. Respect the information and timing
boundaries in GAME_CONTEXT.
'''
    return perspective + revision


def tom_prompt(
    game,
    history,
    observation,
    legal_actions,
    turns=None,
    *,
    inference_context: str = "",
    previous_hypothesis: str | None = None,
    variant: str = "A",
):
    """Build one neutral policy-induction call from one canonical trajectory."""
    if game in RPS_GAMES:
        return tom_prompt_text(observation, turns, previous=previous_hypothesis)
    del history  # Kept in the public signature for compatibility; never serialized.
    hanabi_prediction = ('''For Hanabi only, also predict the teammate's NEXT executed action category before it
occurs. Return next_action_probabilities with exactly hint, play, and discard numeric
probabilities in [0,1] summing to 1. Use near-uniform probabilities when evidence is
insufficient. This prediction is diagnostic only and is NOT part of memory retrieval.
''' if game == "hanabi" else "")
    schema = ('''Return exactly {"opponent_strategy":"one concise English description of overall playing style and strategy",
"next_action_probabilities":{"hint":0.34,"play":0.33,"discard":0.33}}.'''
              if game == "hanabi" else
              '''Return exactly {"opponent_strategy":"one concise English description of overall playing style and strategy"}.''')
    prompt = OPPONENT_MODEL_CONTRACT + "\n\n" + _variant_instructions(variant) + '''
TARGET is the other actor; SELF is our actor. Use GAME_CONTEXT and the ordered
COMPLETED_TRAJECTORY. Describe the opponent's overall playing style and strategy
in 1-2 concise English sentences, under 60 words. Use only completed, player-visible
interactions. If the history provides no basis for a description, use exactly:
"Insufficient evidence for a reliable opponent strategy."

''' + hanabi_prediction + schema + '''
Do not output reasoning, alternatives, confidence, labels, or extra keys.
DATA:\n'''
    payload = {
        "GAME_CONTEXT": inference_context,
        "COMPLETED_TRAJECTORY": completed_joint_actions(turns),
        "CURRENT_STATE": {
            "observation": without_embedded_history(observation),
            "legal_actions": legal_actions,
        },
    }
    if variant in {"B", "C"}:
        payload["PREVIOUS_HYPOTHESIS"] = previous_hypothesis
    return prompt + json.dumps(payload, ensure_ascii=False)
