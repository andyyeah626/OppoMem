from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class DecisionFrame:
    """Everything the focal player may use for one decision."""

    observation: str
    legal_actions: tuple[str, ...]
    cumulative_return: float
    terminal: bool = False
    # Adapter-owned rules, timing, and field semantics for policy inference.
    # This is player-visible and must never contain evaluator identity or labels.
    inference_context: str = ""


@dataclass(frozen=True, slots=True)
class Transition:
    """One settled focal decision and the next player-visible frame."""

    frame: DecisionFrame
    opponent_event: str
    reward_delta: float
    terminal: bool
    evaluator_metrics: dict[str, Any] = field(default_factory=dict)


class GameAdapter(Protocol):
    game_id: str

    def reset(
        self, *, opponent_id: str, seed: int, seat: int = 0,
        training: bool = False,
    ) -> DecisionFrame: ...

    def step(self, action: str) -> Transition: ...

    def close(self) -> None: ...

    def fingerprint(self) -> dict[str, Any]: ...
