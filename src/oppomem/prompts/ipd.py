"""Plain-text IPD prompts following the current RPS player protocol."""
from __future__ import annotations

import json

PROTOCOL_ID = "ipd-shared-player-text-v1"
IPD_GAMES = {"axelrod_ipd", "iterated_pd", "ipd"}
ACTIONS = ("Cooperate", "Defect")

IPD_RULES = """You are playing a repeated game of Iterated Prisoner's Dilemma.
Your objective is to maximize your total payoff over the match.

In each round, both players choose an action without observing the other player's
choice for that round. After the round, both actions and both payoffs are revealed.

If both players Cooperate, each receives 3. If you Cooperate and the opponent
Defects, you receive 0 and the opponent receives 5. If you Defect and the opponent
Cooperates, you receive 5 and the opponent receives 0. If both Defect, each receives 1."""

ACTION_CONTRACT = "Output only your chosen action: Cooperate or Defect."
PLAYER_SYSTEM = IPD_RULES + "\n\n" + ACTION_CONTRACT

PAYOFF_ONLY_TEXT = ("Your score is your own cumulative payoff, not the sum or equality of both players' payoffs. "
                    "Cooperate and Defect are available actions, not moral choices. "
                    "Choose the action with the greatest expected contribution to your total payoff, "
                    "including consequences for later rounds.")
PAYOFF_ONLY_SYSTEM = IPD_RULES + "\n\n" + PAYOFF_ONLY_TEXT + "\n\n" + ACTION_CONTRACT
TOM_SYSTEM = IPD_RULES

TOM_TASK = """Examine the complete sequence of completed rounds in order, attending to
both your exact action and the opponent's exact action in every round. Keep the
players and rounds distinct, and use the full action sequence to understand the
opponent's overall playing style and strategy. Describe your observations in 1-2
concise English sentences, preserving what distinguishes their approach. The final
description should summarize the strategy rather than retell individual rounds or
list specific action-response rules. If the evidence is limited, say so briefly.
Describe the opponent, not what you should do. Actions in the same round are
simultaneous: an opponent's response to your action can first appear in the NEXT
round. Do not call same-round cooperation forgiveness of a defection. If your
actions have not varied, unconditional cooperation and reciprocal cooperation
remain indistinguishable; do not claim an untested response as an observation."""


def turn_rows(turns):
    return [{"round": t.round_index, "self": t.agent_action,
             "opponent": t.opponent_action, "reward": t.reward}
            for t in (turns or [])]


def history_text(history, *, opponent=False):
    lines = []
    for row in history:
        own, other, reward = row["self"], row["opponent"], float(row["reward"])
        if opponent:
            own, other = other, own
            # IPD is not zero-sum. Derive the other seat's payoff from the actions
            # under the shared rules when an explicit payoff is not supplied.
            reward = float(row["opponent_reward"]) if "opponent_reward" in row else (
                (3 if own == "Cooperate" else 1) if own == other else
                (0 if own == "Cooperate" else 5))
        lines.append(f'Round {row["round"] + 1}: your action = {own}; '
                     f'opponent action = {other}; your payoff = {reward:g}.')
    return "\n".join(lines) or "(No completed rounds.)"


def round_context(history, *, horizon=25, opponent=False, round_number=None):
    number = round_number if round_number is not None else (history[-1]["round"] + 2 if history else 1)
    return (f"Round {number} of {horizon}.\n\n"
            "Completed rounds, from your perspective:\n" +
            history_text(history, opponent=opponent))


def player_prompt(history, *, horizon=25, opponent=False, round_number=None,
                  tom=None, memory=None, unresolved=None):
    prompt = round_context(history, horizon=horizon, opponent=opponent,
                           round_number=round_number)
    if tom or memory:
        context = {}
        if tom:
            context["opponent_strategy"] = tom
        if unresolved:
            context["unresolved"] = unresolved
        if memory:
            context["retrieved_memory"] = {key: memory[key] for key in
                ("opponent_strategy", "response_strategy") if key in memory}
        prompt += ("\n\nAdditional opponent-model context (fallible; evaluate against "
                   "the completed rounds; useful action advice may apply even when "
                   "the opponent description is inaccurate):\n" +
                   json.dumps(context, ensure_ascii=False))
    return prompt + "\n\nChoose your action for this round."


def tom_prompt(turns, *, previous=None, horizon=25):
    prompt = round_context(turn_rows(turns), horizon=horizon)
    if previous:
        prompt += "\n\nPrevious opponent description (may need revision):\n" + previous
    return prompt + "\n\n" + TOM_TASK


def parse_player_action(raw):
    action = str(raw or "").strip()
    if action not in ACTIONS:
        raise ValueError("Expected exactly one legal IPD action in plain text")
    return {"action": action}


def wrap_tom_text(raw):
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Empty opponent description")
    return {"opponent_strategy": text, "unresolved": None}


def structured_text_parser(system):
    if system == PLAYER_SYSTEM:
        return parse_player_action
    if system == TOM_SYSTEM:
        return wrap_tom_text
    return None
