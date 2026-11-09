from __future__ import annotations

import json
from pathlib import Path
import random
import re
from typing import Any, Callable, Protocol

from oppomem.games.protocol import DecisionFrame, Transition


HANABI_RULES = """Two-player cooperative Hanabi. Maximize the shared final score (0..25).
Build each color from 1 through 5 in order. Playing any other rank loses a life;
losing all 3 lives ends the game with score ZERO, including previously built stacks.
You cannot see your own cards. possible_colors/ranks are hint constraints, NOT
actual identities; all combinations remain possible. A card is certainly playable
only when every possible identity is the next rank of its color. Do not invent colors.
Partner card identities are visible, but partner cannot see them. Hints identify ALL
matching slots and exclude the hinted attribute from nonmatching slots, costing one
info token. Discard restores one token (maximum 8); completing a 5 also restores one.
Prefer certainly playable cards, otherwise informative hints enabling partner plays
or safe discards. Avoid redundant hints and unsupported plays, especially on last life.
Slot numbers are ZERO-based CURRENT positions: removing a card shifts later slots
left and appends any new draw at the end. Re-read current slots after every action.
Action IDs are NOT slot numbers: use the exact current legal_action_labels mapping.
Do not assume unstated hint conventions. Deck empty gives each player one final turn.
"""


def visible_state(raw: str) -> dict[str, Any]:
    """Translate only the observer's public/hinted view; never read hidden state."""
    header, hands = raw.split("Hands:\n", 1)
    hands, tail = hands.split("Deck size:", 1)
    piles = dict((color, int(rank)) for color, rank in re.findall(r"([RYGWB])(\d)", header))
    result = {"life_tokens": int(re.search(r"Life tokens: (\d+)", header)[1]),
              "info_tokens": int(re.search(r"Info tokens: (\d+)", header)[1]),
              "fireworks": piles, "deck_size": int(tail.splitlines()[0]),
              "discards": tail.split("Discards:", 1)[1].strip().split()}
    blocks = hands.strip().split("-----")
    if len(blocks) != 2 or set(piles) != set("RYGWB"):
        raise ValueError("Unexpected two-player Hanabi observation format")
    for side, block in zip(("own_hand", "partner_hand"), blocks):
        cards = []
        for line in block.splitlines():
            if "||" not in line:
                continue
            actual, knowledge = line.split("||")
            hinted, possible = knowledge.strip().split("|")
            colors = [c for c in possible if c in "RYGWB"]
            ranks = [int(c) for c in possible if c.isdigit()]
            if not colors or not ranks:
                raise ValueError("Empty Hanabi hint constraints")
            card = {"slot": len(cards), "possible_colors": colors, "possible_ranks": ranks,
                    "certainly_playable": all(r == piles[c] + 1 for c in colors for r in ranks),
                    "already_built": all(r <= piles[c] for c in colors for r in ranks)}
            if side == "partner_hand":
                card["visible_card"] = actual.strip()
            elif actual.strip() != "XX":
                raise ValueError("Own hidden cards must not be exposed")
            cards.append(card)
        result[side] = cards
    return result


class HanabiPartner(Protocol):
    def reset(self, seed: int) -> None: ...

    def act(
        self, observation: str, legal_actions: tuple[int, ...],
        action_labels: dict[int, str],
    ) -> int: ...


class RuleHanabiPartner:
    """Small deterministic partner used to validate the public interface."""

    def __init__(self, profile: str = "hint_first") -> None:
        if profile not in {"hint_first", "discard_first", "play_first"}:
            raise ValueError(f"Unknown rule partner {profile!r}")
        self.profile = profile
        self._rng = random.Random(0)

    def reset(self, seed: int) -> None:
        self._rng.seed(seed)

    def act(self, observation: str, legal_actions: tuple[int, ...],
            action_labels: dict[int, str]) -> int:
        priorities = {
            "hint_first": ("Reveal", "Discard", "Play"),
            "discard_first": ("Discard", "Reveal", "Play"),
            "play_first": ("Play", "Reveal", "Discard"),
        }[self.profile]
        for prefix in priorities:
            candidates = [a for a in legal_actions if prefix in action_labels[a]]
            if candidates:
                return candidates[self._rng.randrange(len(candidates))]
        if not legal_actions:
            raise ValueError("Partner was asked to act without legal actions")
        return legal_actions[0]


class AdaptiveHanabiCheckpointPartner:
    """Stable bridge for Adaptive-Hanabi/OBL checkpoint policies.

    The repository's native CUDA actors do not share OpenSpiel's Python bot
    interface, so loading is explicit. ``loader`` must return an object exposing
    reset(seed) and act(observation, legal_actions, action_labels).
    """

    def __init__(self, checkpoint: str | Path,
                 loader: Callable[[Path], HanabiPartner]) -> None:
        self.checkpoint = Path(checkpoint)
        self.loader = loader
        self._policy: HanabiPartner | None = None

    def reset(self, seed: int) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if self._policy is None:
            self._policy = self.loader(self.checkpoint)
        self._policy.reset(seed)

    def act(self, observation: str, legal_actions: tuple[int, ...],
            action_labels: dict[int, str]) -> int:
        if self._policy is None:
            raise RuntimeError("Call reset before checkpoint partner act")
        return int(self._policy.act(observation, legal_actions, action_labels))


class HanabiAdapter:
    game_id = "adaptive_hanabi"

    def __init__(self, partner: HanabiPartner, *, players: int = 2) -> None:
        if players != 2:
            raise ValueError("The first adapter revision supports two-player Hanabi")
        self.partner = partner
        self.players = players
        self._rng = random.Random(0)
        self._state = None
        self._seat = 0
        self._total = 0.0
        self._history: list[dict[str, Any]] = []

    def _resolve_chance(self) -> None:
        import pyspiel

        while self._state is not None and self._state.current_player() == pyspiel.PlayerId.CHANCE:
            outcomes = self._state.chance_outcomes()
            threshold = self._rng.random()
            running = 0.0
            selected = outcomes[-1][0]
            for action, probability in outcomes:
                running += probability
                if threshold <= running:
                    selected = action
                    break
            self._state.apply_action(selected)

    def reset(self, *, opponent_id: str, seed: int, seat: int = 0,
              training: bool = False) -> DecisionFrame:
        import pyspiel

        if seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        self._rng.seed(seed)
        self.partner.reset(seed ^ 0xA11CE)
        self._seat = seat
        self._total = 0.0
        self._history = []
        game = pyspiel.load_game("hanabi", {"players": 2, "hand_size": 5})
        self._state = game.new_initial_state()
        self._resolve_chance()
        while not self._state.is_terminal() and self._state.current_player() != self._seat:
            self._play_partner()
        return self._frame(training=training)

    def _labels(self) -> dict[int, str]:
        return {int(a): self._state.action_to_string(int(a))
                for a in self._state.legal_actions()}

    def _observation(self, seat=None) -> str:
        seat = self._seat if seat is None else seat
        return json.dumps({
            "game": "Two-player Hanabi",
            **visible_state(self._state.observation_string(seat)),
            "recent_public_actions": [{**row, "actor": row["actor"] if seat == self._seat
                                       else {"self": "partner", "partner": "self"}[row["actor"]]}
                                      for row in self._history[-6:]],
            "objective": "Maximize the shared fireworks score while respecting private hands.",
            "legal_action_labels": {str(k): v for k, v in self._labels().items()},
        }, ensure_ascii=False, separators=(",", ":"))

    def _frame(self, *, training: bool = False) -> DecisionFrame:
        terminal = bool(self._state.is_terminal())
        return DecisionFrame(
            observation=(json.dumps({"game": "Two-player Hanabi", "terminal": True,
                                     "completed_public_actions": self._history},
                                    ensure_ascii=False, separators=(",", ":"))
                         if terminal else self._observation()),
            legal_actions=(() if terminal else tuple(str(a) for a in self._state.legal_actions())),
            cumulative_return=self._total,
            terminal=terminal,
            inference_context=HANABI_RULES,
        )

    def _apply(self, action: int, actor: str) -> str:
        before = visible_state(self._state.observation_string(self._state.current_player()))
        label = self._labels()[action]
        self._state.apply_action(action)
        self._resolve_chance()
        after = visible_state(self._state.observation_string(self._seat))
        self._history.append({"actor": actor, "action": label,
            "life_lost": before["life_tokens"] - after["life_tokens"],
            "fireworks_gained": sum(after["fireworks"].values()) - sum(before["fireworks"].values())})
        return label

    def _play_partner(self) -> str:
        observation = self._observation(1 - self._seat)
        legal = tuple(int(a) for a in self._state.legal_actions())
        labels = self._labels()
        action = int(self.partner.act(observation, legal, labels))
        if action not in legal:
            raise ValueError(f"Partner returned illegal Hanabi action {action}")
        return self._apply(action, "partner")

    def step(self, action: str) -> Transition:
        if self._state is None:
            raise RuntimeError("Call reset first")
        if self._state.is_terminal() or self._state.current_player() != self._seat:
            raise RuntimeError("It is not the focal player's decision")
        legal = tuple(int(a) for a in self._state.legal_actions())
        try:
            action_id = int(action)
        except ValueError as error:
            raise ValueError(f"Hanabi action must be an integer ID, got {action!r}") from error
        if action_id not in legal:
            raise ValueError(f"Illegal Hanabi action ID {action_id}")
        self._apply(action_id, "self")
        events = []
        while not self._state.is_terminal() and self._state.current_player() != self._seat:
            events.append(self._play_partner())
        reward = 0.0
        if self._state.is_terminal():
            terminal_total = float(self._state.returns()[self._seat])
            reward = terminal_total - self._total
            self._total = terminal_total
        frame = self._frame()
        opponent_event = (json.dumps(events, ensure_ascii=False) if events else
                          "Game ended. Own terminal payoff only.")
        public = visible_state(self._state.observation_string(self._seat))
        return Transition(frame=frame, opponent_event=opponent_event,
                          reward_delta=reward, terminal=frame.terminal,
                          evaluator_metrics={"public_action_count": len(self._history),
                              "fireworks_score": sum(public["fireworks"].values()),
                              "life_tokens": public["life_tokens"], "deck_size": public["deck_size"],
                              "bomb_out": public["life_tokens"] == 0,
                              "misplays": {actor: sum(row["life_lost"] for row in self._history if row["actor"] == actor)
                                           for actor in ("self", "partner")}})

    def close(self) -> None:
        self._state = None

    def fingerprint(self) -> dict[str, Any]:
        import pyspiel

        return {
            "adapter": type(self).__name__,
            "protocol_id": "hanabi-indexed-public-view-v2",
            "pyspiel_version": getattr(pyspiel, "__version__", "unknown"),
            "partner_backend": type(self.partner).__name__,
        }
