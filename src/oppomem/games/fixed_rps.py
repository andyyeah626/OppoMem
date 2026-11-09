from __future__ import annotations

import json
from typing import Any

from oppomem.games.protocol import DecisionFrame, Transition
from oppomem.games.rps import ACTIONS, INFERENCE_CONTEXT


BEATS = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}
WINNING_PAIRS = {("Rock", "Scissors"), ("Paper", "Rock"), ("Scissors", "Paper")}
OPPONENTS = ("rockbot", "rotatebot", "copybot")


class FixedRPSAdapter:
    """Pure-Python equivalents of the three deterministic Roshambo opponents."""

    game_id = "open_spiel_roshambo"

    def __init__(self, *, horizon: int = 12) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self._round = 0
        self._total = 0.0
        self._opponent_id = ""
        self._history: list[dict[str, Any]] = []

    @staticmethod
    def available_opponents() -> tuple[str, ...]:
        return OPPONENTS

    def reset(self, *, opponent_id: str, seed: int, seat: int = 0,
              training: bool = False) -> DecisionFrame:
        del seed, training
        if seat != 0:
            raise ValueError("FixedRPSAdapter currently supports only seat 0")
        if opponent_id not in OPPONENTS:
            raise ValueError(f"Unknown fixed RPS opponent {opponent_id!r}")
        self._round = 0
        self._total = 0.0
        self._opponent_id = opponent_id
        self._history = []
        return self._frame()

    def _frame(self) -> DecisionFrame:
        terminal = self._round >= self.horizon
        observation = json.dumps({
            "game": "Repeated simultaneous Rock Paper Scissors",
            "round": self._round,
            "remaining_rounds": self.horizon - self._round,
            "cumulative_return": self._total,
            "completed_joint_actions": self._history,
            "timing": (
                "Simultaneous: neither player can observe the other's action for the "
                "round being chosen. An action can depend only on rounds completed "
                "before it."
            ),
            "payoff": "win=+1, draw=0, loss=-1",
        }, ensure_ascii=False, separators=(",", ":"))
        return DecisionFrame(
            observation=observation,
            legal_actions=() if terminal else ACTIONS,
            cumulative_return=self._total,
            terminal=terminal,
            inference_context=INFERENCE_CONTEXT,
        )

    def _opponent_action(self) -> str:
        if self._opponent_id == "rockbot":
            return "Rock"
        if self._opponent_id == "rotatebot":
            return ACTIONS[self._round % len(ACTIONS)]
        if self._round == 0:
            return "Paper"
        return BEATS[self._history[-1]["self"]]

    def step(self, action: str) -> Transition:
        if action not in ACTIONS or self._round >= self.horizon:
            raise ValueError(f"Illegal fixed RPS action {action!r}")
        opponent_action = self._opponent_action()
        reward = (0.0 if action == opponent_action
                  else 1.0 if (action, opponent_action) in WINNING_PAIRS
                  else -1.0)
        self._total += reward
        self._history.append({
            "round": self._round,
            "self": action,
            "opponent": opponent_action,
            "reward": reward,
        })
        self._round += 1
        frame = self._frame()
        return Transition(
            frame=frame,
            opponent_event=opponent_action,
            reward_delta=reward,
            terminal=frame.terminal,
            evaluator_metrics={"round": self._round},
        )

    def close(self) -> None:
        pass

    def fingerprint(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "protocol_id": "fixed-rps-three-opponents-v1",
            "implementation": "pure_python",
            "opponents": list(OPPONENTS),
            "horizon": self.horizon,
        }
