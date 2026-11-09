from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from oppomem.games.protocol import DecisionFrame, Transition


ACTIONS = ("Cooperate", "Defect")

INFERENCE_CONTEXT = (
    "Iterated Prisoner's Dilemma with simultaneous actions. Per-round SELF payoffs "
    "are 3 for mutual cooperation, 0 when SELF cooperates and TARGET defects, 5 when "
    "SELF defects and TARGET cooperates, and 1 for mutual defection. Each player "
    "chooses using only previously completed rounds and cannot observe the other "
    "player's current action. COMPLETED_TRAJECTORY lists rounds in order with SELF's "
    "action, TARGET's action, and SELF's reward. Legal actions are the exact strings "
    "Cooperate and Defect."
)


def _load_axelrod(source: str | Path | None = None):
    if source is not None:
        resolved = str(Path(source).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    import axelrod as axl

    return axl


# Axelrod's First Tournament entrants, as the library itself defines the set
# (`axelrod.strategies.axelrod_first_strategies`, 15 members).  This is the population
# the study wants: 12 hand-written `FirstBy*` rules plus the three canonical ones.
# `TitForTat` matters most -- its rule is literally "copy the opponent's last action",
# the conditional form the online ToM never forms a hypothesis about in RPS.
FIRST_TOURNAMENT: tuple[str, ...] = (
    "tit_for_tat",
    "grudger",
    "random",
    "first_by_tideman_and_chieruzzi",
    "first_by_nydegger",
    "first_by_grofman",
    "first_by_shubik",
    "first_by_stein_and_rapoport",
    "first_by_davis",
    "first_by_graaskamp",
    "first_by_downing",
    "first_by_feld",
    "first_by_joss",
    "first_by_tullock",
    "first_by_anonymous",
)

# Memory-evolution training pool: rules that stay identifiable on a 25-round
# trace.  FIRST_TOURNAMENT remains the library's full 1980 set for probes and
# truth tables.  Dropped from training because they are not readable here:
# Graaskamp's probe is move 51 (absent at horizon 25), Feld is still ~TFT at
# 25, and Nydegger / Anonymous / Tullock / Tideman / Downing do not compress
# to a short recognizable rule.
TYPICAL_TRAINING: tuple[str, ...] = (
    "tit_for_tat",
    "grudger",
    "random",
    "first_by_joss",
    "first_by_davis",
    "first_by_stein_and_rapoport",
    "first_by_shubik",
    "first_by_grofman",
)


class AxelrodAdapter:
    """A focal LLM player against an official Axelrod-Python Player."""

    game_id = "axelrod_ipd"

    # `alternator` is kept for the four-game pilot, which names it directly; it is not
    # a First Tournament entrant and is absent from FIRST_TOURNAMENT.
    OPPONENTS = {
        "cooperator": "Cooperator",
        "defector": "Defector",
        "tit_for_2_tats": "TitFor2Tats",
        "win_stay_lose_shift": "WinStayLoseShift",
        "alternator": "Alternator",
        "tit_for_tat": "TitForTat",
        "grudger": "Grudger",
        "random": "Random",
        "first_by_tideman_and_chieruzzi": "FirstByTidemanAndChieruzzi",
        "first_by_nydegger": "FirstByNydegger",
        "first_by_grofman": "FirstByGrofman",
        "first_by_shubik": "FirstByShubik",
        "first_by_stein_and_rapoport": "FirstBySteinAndRapoport",
        "first_by_davis": "FirstByDavis",
        "first_by_graaskamp": "FirstByGraaskamp",
        "first_by_downing": "FirstByDowning",
        "first_by_feld": "FirstByFeld",
        "first_by_joss": "FirstByJoss",
        "first_by_tullock": "FirstByTullock",
        "first_by_anonymous": "FirstByAnonymous",
    }

    def __init__(self, *, horizon: int = 12,
                 source: str | Path | None = None) -> None:
        self.horizon = int(horizon)
        self.source = source
        self._axl = None
        self._opponent = None
        self._focal = None
        self._history: list[dict[str, Any]] = []
        self._total = 0.0
        self._opponent_total = 0.0
        self._seed = 0

    def reset(self, *, opponent_id: str, seed: int, seat: int = 0,
              training: bool = False) -> DecisionFrame:
        if seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        if opponent_id not in self.OPPONENTS:
            raise ValueError(f"Unknown Axelrod opponent {opponent_id!r}")
        self._axl = _load_axelrod(self.source)
        self._opponent = getattr(self._axl, self.OPPONENTS[opponent_id])()
        self._focal = self._axl.Cooperator()
        self._opponent.set_seed(int(seed))
        self._focal.set_seed(int(seed) ^ 0x5EED)
        self._history = []
        self._total = 0.0
        self._opponent_total = 0.0
        self._seed = int(seed)
        return self._frame()

    def _frame(self) -> DecisionFrame:
        payload: dict[str, Any] = {
            "game": "Iterated Prisoner's Dilemma",
            "round": len(self._history),
            "remaining_rounds": self.horizon - len(self._history),
            "cumulative_return": self._total,
            "completed_joint_actions": self._history,
            "payoffs": {"CC": 3, "CD": 0, "DC": 5, "DD": 1},
            "timing": (
                "Simultaneous: both actions for this round depend only on completed history."
            ),
        }
        terminal = len(self._history) >= self.horizon
        return DecisionFrame(
            observation=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            legal_actions=() if terminal else ACTIONS,
            cumulative_return=self._total,
            terminal=terminal,
            inference_context=INFERENCE_CONTEXT,
        )

    def step(self, action: str) -> Transition:
        if self._axl is None or self._opponent is None or self._focal is None:
            raise RuntimeError("Call reset first")
        if action not in ACTIONS or len(self._history) >= self.horizon:
            raise ValueError(f"Illegal Axelrod action {action!r}")

        # Crucially, choose the opponent action before either current action is
        # appended. This preserves the simultaneous-move information boundary.
        opponent_action = self._opponent.strategy(self._focal)
        own_action = self._axl.Action.C if action == "Cooperate" else self._axl.Action.D
        self._focal.update_history(own_action, opponent_action)
        self._opponent.update_history(opponent_action, own_action)
        score = self._axl.Game().score((own_action, opponent_action))
        reward = float(score[0])
        opponent_text = "Cooperate" if opponent_action == self._axl.Action.C else "Defect"
        self._total += reward
        self._opponent_total += float(score[1])
        self._history.append({
            "round": len(self._history), "self": action,
            "opponent": opponent_text, "reward": reward,
        })
        frame = self._frame()
        return Transition(
            frame=frame,
            opponent_event=opponent_text,
            reward_delta=reward,
            terminal=frame.terminal,
            evaluator_metrics={"round": len(self._history),
                               "opponent_return": self._opponent_total},
        )

    def close(self) -> None:
        self._opponent = None
        self._focal = None

    def fingerprint(self) -> dict[str, Any]:
        axl = _load_axelrod(self.source)
        return {
            "adapter": type(self).__name__,
            "protocol_id": "axelrod-closedset-policy-induction-v3",
            "axelrod_version": getattr(axl, "__version__", "unknown"),
            "horizon": self.horizon,
            "opponents": sorted(self.OPPONENTS),
        }
