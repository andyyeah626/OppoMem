"""One player protocol for every RPS seat; ToM adds advisory context only."""
from __future__ import annotations

import json

PROTOCOL_ID = "rps-shared-player-text-v1"
RPS_GAMES = {"open_spiel_roshambo", "repeated_rps"}
RPS_RULES = """You are playing a repeated game of Rock-Paper-Scissors.
Your objective is to maximize your total payoff over the match.

In each round, both players choose an action without observing
the other player's choice for that round. After the round,
both actions and the outcome are revealed to both players.

Rock beats Scissors, Scissors beats Paper, and Paper beats Rock.
Your payoff is +1 for a win, 0 for a draw, and -1 for a loss."""

ACTION_CONTRACT = 'Output only your chosen action: Rock, Paper, or Scissors.'
PLAYER_SYSTEM = RPS_RULES + "\n\n" + ACTION_CONTRACT

TOM_TASK = """Examine the complete sequence of completed rounds in order, attending to
both your exact action and the opponent's exact action in every round. Keep the
players and rounds distinct, and use the full action sequence to understand the
opponent's overall playing style and strategy. Describe your observations in 1-2
concise English sentences, preserving what distinguishes their approach. The final
description should summarize the strategy rather than retell individual rounds or
list specific action-response rules. For the final opponent description, abstract
away the names of the three throws: say fixed single-action preference, delayed
countering, switching, repetition, or a multi-action cycle instead of naming a
specific throw. If the evidence is limited, say so briefly. Describe the opponent,
not what you should do. Actions in the same round are simultaneous: matching or
countering your current action is not evidence of reacting to it. Test reactive
hypotheses against your earlier actions, and distinguish observed repetition from
an untested claim about what the opponent would do after a different history."""
# ToM instructions are confined to the dedicated user prompt.
TOM_SYSTEM = RPS_RULES


def turn_rows(turns):
    return [{"round": t.round_index, "self": t.agent_action,
             "opponent": t.opponent_action, "reward": t.reward} for t in (turns or [])]


def history_text(history, *, opponent=False):
    lines = []
    for row in history:
        own, other, reward = row["self"], row["opponent"], row["reward"]
        if opponent:
            own, other, reward = other, own, -reward
        lines.append(f'Round {row["round"] + 1}: your action = {own}; '
                     f'opponent action = {other}; your payoff = {reward:g}.')
    return "\n".join(lines) or "(No completed rounds.)"


def round_context(history, *, horizon, opponent=False, round_number=None):
    number = round_number if round_number is not None else (history[-1]["round"] + 2 if history else 1)
    return (f"Round {number} of {horizon}.\n\n"
            "Completed rounds, from your perspective:\n"
            + history_text(history, opponent=opponent))


def player_prompt(history, *, horizon, opponent=False, round_number=None,
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
            # Only the public memory payload; never IDs, scores, or evaluator data.
            context["retrieved_memory"] = {key: memory[key] for key in
                ("opponent_strategy", "response_strategy") if key in memory}
        prompt += ("\n\nAdditional opponent-model context (fallible; evaluate against "
                   "the completed rounds; useful action advice may apply even when "
                   "the opponent description is inaccurate):\n" + json.dumps(context, ensure_ascii=False))
    return prompt + "\n\nChoose your action for this round."


def observation_prompt(observation, turns=None, *, tom=None, memory=None):
    state = json.loads(observation) if isinstance(observation, str) else observation
    # The environment's full public history is authoritative, even if a separate
    # ToM call deliberately used a history window.
    rows = state.get("completed_joint_actions", turn_rows(turns))
    completed = int(state.get("round", rows[-1]["round"] + 1 if rows else 0))
    horizon = completed + int(state["remaining_rounds"])
    return player_prompt(rows, horizon=horizon, round_number=completed + 1,
                         tom=tom, memory=memory)


def parse_player_action(raw):
    action = raw.strip()
    if action not in ("Rock", "Paper", "Scissors"):
        raise ValueError("Expected exactly one legal action in plain text")
    return {"action": action}


def wrap_tom_text(raw):
    text = raw.strip()
    if not text:
        raise ValueError("Empty opponent description")
    return {"opponent_strategy": text, "unresolved": None}


def structured_text_parser(system):
    """The model emits text; only local code constructs structured fields."""
    if system == PLAYER_SYSTEM:
        return parse_player_action
    if system == TOM_SYSTEM:
        return wrap_tom_text
    return None


def tom_prompt_text(observation, turns=None, *, previous=None):
    prompt = observation_prompt(observation, turns)
    prompt = prompt.removesuffix("\n\nChoose your action for this round.")
    if previous:
        prompt += "\n\nPrevious opponent description (may need revision):\n" + previous
    return prompt + "\n\n" + TOM_TASK
