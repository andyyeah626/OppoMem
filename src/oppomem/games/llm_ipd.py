"""An LLM opponent for the Axelrod IPD benchmark.

Subclasses `AxelrodAdapter` for the same reason `llm_opponent.py` subclasses
`RoshamboAdapter`: `_frame()`, the payoff arithmetic and the `Transition` payload are
inherited rather than re-derived, so the agent sees a byte-identical observation
whether the other seat is a library bot or a model.  Without that the two training
pools could not be pooled into one bank.

Training-only.  There is no tape/replay pairing here and deliberately so: pairing
exists to compare two arms against one opponent sequence, and this runner trains a
single arm.  A tape would add the recording asymmetry that made the RPS batch-2
delta uninterpretable (see the batch-2 confound) without buying anything.

One invariant is load-bearing: `self._history` is exactly what the parent builds,
because `_frame()` serializes it into the observation the AGENT reads.  The
opponent's own view -- which needs the opponent's per-round payoff -- is kept in a
SEPARATE list, `_opponent_view`.  Adding a key to `_history` for the opponent's
convenience would silently change the agent's observation and break comparability
with the bot pool.
"""
from __future__ import annotations

import random
from typing import Any

from oppomem.games.ipd import ACTIONS, AxelrodAdapter, _load_axelrod
from oppomem.games.protocol import DecisionFrame, Transition


# Same discipline as `llm_opponent.OPPONENT_SYSTEM`: the output contract comes first
# and in the imperative.  These are reasoning models whose hidden deliberation bills
# against max_tokens, and a wordy prompt lets them spend the whole budget thinking and
# return empty content with finish_reason='length'.
OPPONENT_SYSTEM = (
    "You are playing Iterated Prisoner's Dilemma. Answer with ONE JSON object and "
    "nothing else.\n"
    'Output format, exactly: {"action": "Cooperate"}\n'
    "Your entire reply must be that object. Do not analyse, explain, or think step by "
    "step.\n"
    "Payoffs to you per round: both cooperate 3; you cooperate and they defect 0; you "
    "defect and they cooperate 5; both defect 1. Both players choose before either "
    "current-round action is revealed, so you cannot see their move for the round you "
    "are choosing in, and they cannot see yours. Legal actions are the exact strings "
    "Cooperate and Defect."
)


def _opponent_prompt(*, horizon: int, view: list[dict[str, Any]]) -> str:
    """Completed rounds only, from the opponent's own point of view.

    The current round is absent on purpose: the round is simultaneous, so showing it
    would hand the opponent information the benchmark says it does not have.
    """
    if not view:
        lines = "  (no completed rounds yet)"
    else:
        lines = "\n".join(
            f"  R{row['round'] + 1}: you played {row['you']}, they played {row['them']} "
            f"-> you got {row['payoff']}"
            for row in view
        )
    return (
        f"Round {len(view) + 1} of {horizon}.\n"
        f"Completed rounds, in order:\n{lines}\n\n"
        f"Choose your action for round {len(view) + 1}.\n"
        'Reply exactly {"action": "Cooperate"|"Defect"}.'
    )


class LLMOpponentIPDAdapter(AxelrodAdapter):
    """IPD with a model in the opponent seat."""

    def __init__(
        self,
        *,
        opponent_id: str,
        llm: Any,
        horizon: int = 50,
        source: str | None = None,
        seed_xor: int = 0x5EEDB07,
    ) -> None:
        super().__init__(horizon=horizon, source=source)
        self.opponent_id = str(opponent_id)
        self.llm = llm
        self.seed_xor = int(seed_xor)
        self.seed = 0
        self._opponent_view: list[dict[str, Any]] = []
        self._rng = random.Random(0)
        # Counted, not just swallowed.  The fallback below is deliberately uniform so a
        # broken opponent cannot masquerade as a learnable policy -- but an opponent that
        # falls back on every round *is* a uniform-random opponent, and without this
        # counter a cell where the model never once produced a usable action looks
        # exactly like a cell where it played well.  The usage journal cannot show it
        # either: `complete_json` succeeded, it was `_parse` that rejected the answer.
        self.calls = 0
        self.fallbacks = 0

    def reset(self, *, opponent_id: str, seed: int, seat: int = 0,
              training: bool = False) -> DecisionFrame:
        if seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        if opponent_id != self.opponent_id:
            raise ValueError(f"Unknown LLM opponent {opponent_id!r}")
        # Everything the parent's reset does except instantiate a library bot.
        self._axl = _load_axelrod(self.source)
        self._opponent = None
        self._focal = self._axl.Cooperator()
        self._focal.set_seed(int(seed) ^ 0x5EED)
        self._history = []
        self._opponent_view = []
        # One adapter serves every episode of the run, so the counters are per-episode
        # and reset here rather than accumulating into an unreadable total.
        self.calls = 0
        self.fallbacks = 0
        self._total = 0.0
        self._opponent_total = 0.0
        self._seed = int(seed)
        self.seed = int(seed)
        self._rng = random.Random((int(seed) & 0xFFFFFFFF) ^ self.seed_xor)
        return self._frame()

    def _next_opponent_action(self) -> tuple[str, bool]:
        """Never raises: the agent's decision for this round is already paid for, and
        an exception here destroys the whole cell.

        Returns (action, this_round_fell_back).  The per-round flag is what
        `fallback_rounds` in the evolution pipeline reads; a cumulative
        `fallbacks == calls` bit on the last transition would hide a mixed episode.
        """
        prompt = _opponent_prompt(horizon=self.horizon, view=self._opponent_view)
        self.calls += 1
        try:
            raw = self.llm.complete_json(
                stage="opponent_decision", system=OPPONENT_SYSTEM, prompt=prompt,
            )
            return self._parse(raw), False
        except Exception:  # noqa: BLE001
            self.fallbacks += 1
            # Unbiased fallback.  Mirroring its own last move (or a fixed action) would
            # turn the opponent into a copybot-like policy precisely when it fails,
            # manufacturing the very artefact the bank is meant to learn.
            return self._rng.choice(ACTIONS), True

    @staticmethod
    def _parse(raw: Any) -> str:
        action = str((raw or {}).get("action", "")).strip()
        for candidate in ACTIONS:
            if action.lower() == candidate.lower():
                return candidate
        raise ValueError(f"Opponent returned an unusable action {action!r}")

    def step(self, action: str) -> Transition:
        if self._axl is None or self._focal is None:
            raise RuntimeError("Call reset first")
        if action not in ACTIONS or len(self._history) >= self.horizon:
            raise ValueError(f"Illegal Axelrod action {action!r}")

        # Chosen before either current action is appended, preserving the
        # simultaneous-move information boundary.
        opponent_text, this_round_fallback = self._next_opponent_action()
        own_action = self._axl.Action.C if action == "Cooperate" else self._axl.Action.D
        opponent_action = (self._axl.Action.C if opponent_text == "Cooperate"
                           else self._axl.Action.D)
        self._focal.update_history(own_action, opponent_action)
        score = self._axl.Game().score((own_action, opponent_action))
        reward = float(score[0])
        opponent_reward = float(score[1])
        self._total += reward
        self._opponent_total += opponent_reward
        # Exactly the parent's row -- see the module docstring.
        self._history.append({
            "round": len(self._history), "self": action,
            "opponent": opponent_text, "reward": reward,
        })
        self._opponent_view.append({
            "round": len(self._opponent_view), "you": opponent_text,
            "them": action, "payoff": opponent_reward,
        })
        frame = self._frame()
        return Transition(
            frame=frame,
            opponent_event=opponent_text,
            reward_delta=reward,
            terminal=frame.terminal,
            evaluator_metrics={"round": len(self._history),
                               "opponent_return": self._opponent_total,
                               "opponent_kind": "llm",
                               # The RPS convention, which `core/batch_evolution.py` and
                               # `run_rps_seed18_test.py` already read: `runner.py` keeps
                               # the last transition's metrics, so the cumulative count
                               # lands in the record and a fully-fallen-back opponent
                               # (one that is really uniform random) is visible.
                               "opponent_fallback": this_round_fallback,
                               "opponent_fallbacks": self.fallbacks,
                               "opponent_calls": self.calls},
        )

    def close(self) -> None:
        self._opponent = None
        self._focal = None

    def fingerprint(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "protocol_id": "axelrod-closedset-policy-induction-v4",
            "opponent_id": self.opponent_id,
            "opponent_model": getattr(self.llm, "model", None),
            "horizon": self.horizon,
        }
