"""An LLM opponent for the Roshambo benchmark.

Subclasses `RoshamboAdapter` so that `_frame()`, the payoff arithmetic and the
`Transition` payload are inherited rather than re-derived: the agent must see a
byte-identical observation whether it is playing a seed bot or an LLM, otherwise
the two batches are not comparable.

The one thing the parent cannot do is pair the two arms.  A seed bot is a
deterministic function of `seed` and the joint history, so replaying a seed
replays the opponent.  An LLM is not: its action depends on the whole joint
history, and the two arms play different policies, so from round 1 on they would
face different opponent actions even at the same seed.  Pairing therefore comes
from a recorded tape, not from the seed:

    phase "live"    the vanilla arm queries the model and writes
                    <record_dir>/<seed>.json with the 12 actions it chose
    phase "replay"  the tom_memory arm reads that tape and plays it back

The vanilla pass must therefore finish before the replay pass.  In replay the
opponent is a fixed sequence conditioned on the vanilla arm's history, so the
paired comparison answers "same opponent tape, different memory" -- it removes
opponent sampling noise, and it is not the same thing as both arms facing a live
model.  A missing tape raises rather than silently degrading.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import random
from typing import Any

from oppomem.games.protocol import DecisionFrame, Transition
from oppomem.games.rps import ACTIONS, ACTION_TO_ID, INFERENCE_CONTEXT, RoshamboAdapter


from oppomem.prompts.rps import PLAYER_SYSTEM, player_prompt

OPPONENT_SYSTEM = PLAYER_SYSTEM


def _opponent_prompt(*, horizon: int, history: list[dict[str, Any]]) -> str:
    return player_prompt(history, horizon=horizon, opponent=True)


class LLMOpponentAdapter(RoshamboAdapter):
    """Roshambo with an LLM in the opponent seat, in live or replay mode."""

    def __init__(
        self,
        *,
        opponent_id: str,
        llm: Any,
        horizon: int = 12,
        recall: int = 1,
        record_dir: str | Path | None = None,
        replay_dir: str | Path | None = None,
        seed_xor: int = 0x5EEDB07,
    ) -> None:
        super().__init__(horizon=horizon, recall=recall)
        if record_dir is not None and replay_dir is not None:
            raise ValueError("record_dir and replay_dir are mutually exclusive")
        self.opponent_id = str(opponent_id)
        self.llm = llm
        self.record_dir = Path(record_dir) if record_dir else None
        self.replay_dir = Path(replay_dir) if replay_dir else None
        self.seed_xor = int(seed_xor)
        self.seed = 0
        self._tape: list[str] | None = None
        self._tape_rows: list[dict[str, Any]] = []
        self._rng = random.Random(0)

    # ---- the parent validates against pyspiel's bot list and builds a C bot ----------
    def available_opponents(self) -> tuple[str, ...]:
        return (self.opponent_id,)

    def reset(self, *, opponent_id: str, seed: int, seat: int = 0,
              training: bool = False) -> DecisionFrame:
        import numpy as np
        import pyspiel

        if seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        if opponent_id != self.opponent_id:
            raise ValueError(f"Unknown LLM opponent {opponent_id!r}")

        # Kept identical to the parent so the framing is the same as the bot adapter.
        np.random.seed(int(seed) & 0xFFFFFFFF)
        self._seat = seat
        self._opponent_id = opponent_id
        self._history = []
        self._total = 0.0
        self.seed = int(seed)
        self._rng = random.Random((int(seed) & 0xFFFFFFFF) ^ self.seed_xor)
        game = pyspiel.load_game(
            "repeated_game(stage_game=matrix_rps(),"
            f"num_repetitions={self.horizon},recall={self.recall})"
        )
        self._state = game.new_initial_state()
        self._bot = None
        self._tape_rows = []
        if self.replay_dir is not None:
            self._tape = self._load_tape(int(seed))
        else:
            self._tape = None
        return self._frame()

    # ---- tape handling ---------------------------------------------------------------
    def _tape_path(self, directory: Path, seed: int) -> Path:
        return directory / f"{seed}.json"

    def _load_tape(self, seed: int) -> list[str]:
        path = self._tape_path(self.replay_dir, seed)
        if not path.is_file():
            raise RuntimeError(
                f"No opponent tape for seed {seed} at {path}. The live (vanilla) pass "
                "must run to completion before the replay (tom_memory) pass; refusing "
                "to play an unpaired episode."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        actions = [str(action) for action in payload["actions"]]
        if len(actions) != self.horizon:
            raise RuntimeError(
                f"Opponent tape {path} holds {len(actions)} actions, expected {self.horizon}"
            )
        bad = [action for action in actions if action not in ACTIONS]
        if bad:
            raise RuntimeError(f"Opponent tape {path} holds illegal actions {bad}")
        return actions

    def _write_tape(self) -> None:
        if self.record_dir is None:
            return
        self.record_dir.mkdir(parents=True, exist_ok=True)
        path = self._tape_path(self.record_dir, self.seed)
        payload = {
            "seed": self.seed,
            "opponent_id": self.opponent_id,
            "model": getattr(self.llm, "model", None),
            "horizon": self.horizon,
            "created_at": datetime.now().isoformat(),
            "actions": [row["action"] for row in self._tape_rows],
            "rounds": self._tape_rows,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        os.replace(temporary, path)

    # ---- the opponent's move ---------------------------------------------------------
    def _next_opponent_action(self) -> str:
        if self._tape is not None:
            index = len(self._history)
            if index >= len(self._tape):
                raise RuntimeError("Opponent tape exhausted")
            return self._tape[index]

        prompt = _opponent_prompt(horizon=self.horizon, history=self._history)
        fallback = False
        raw: Any = None
        try:
            # Never raise out of here: the agent's decision for this round has already
            # been paid for, and a raised exception destroys the whole cell.
            raw = self.llm.complete_json(
                stage="opponent_decision", system=OPPONENT_SYSTEM, prompt=prompt,
            )
            action = self._parse(raw)
        except Exception as exc:  # noqa: BLE001
            # Unbiased fallback. Repeating its own last move would turn the opponent into
            # a copybot-like policy precisely when it fails, which is exactly the kind of
            # artefact this batch is trying to measure.
            fallback = True
            action = self._rng.choice(ACTIONS)
            self._tape_rows.append({"round": len(self._history), "action": action,
                                    "fallback": True, "error": f"{type(exc).__name__}: {exc}"})
            self._write_tape()
            return action
        self._tape_rows.append({"round": len(self._history), "action": action,
                                "fallback": fallback, "raw": raw})
        self._write_tape()
        return action

    @staticmethod
    def _parse(raw: Any) -> str:
        action = str((raw or {}).get("action", "")).strip()
        for candidate in ACTIONS:
            if action.lower() == candidate.lower():
                return candidate
        raise ValueError(f"Opponent returned an unusable action {action!r}")

    def step(self, action: str) -> Transition:
        if self._state is None:
            raise RuntimeError("Call reset first")
        if action not in ACTIONS or self._state.is_terminal():
            raise ValueError(f"Illegal Roshambo action {action!r}")
        # The only line that differs from RoshamboAdapter.step: where the opponent's
        # action comes from. Everything downstream is the parent's arithmetic, copied
        # verbatim so the reward and the frame cannot drift.
        opponent_action = self._next_opponent_action()
        opponent_action_id = ACTION_TO_ID[opponent_action]
        own_action_id = ACTION_TO_ID[action]
        joint = ([own_action_id, opponent_action_id] if self._seat == 0
                 else [opponent_action_id, own_action_id])
        self._state.apply_actions(joint)
        reward = float(self._state.rewards()[self._seat])
        self._total += reward
        self._history.append({
            "round": len(self._history),
            "self": action,
            "opponent": opponent_action,
            "reward": reward,
        })
        frame = self._frame()
        if frame.terminal:
            terminal_return = float(self._state.returns()[self._seat])
            if abs(terminal_return - self._total) > 1e-9:
                raise RuntimeError("Roshambo incremental rewards disagree with terminal return")
        return Transition(
            frame=frame,
            opponent_event=opponent_action,
            reward_delta=reward,
            terminal=frame.terminal,
            evaluator_metrics={
                "round": len(self._history),
                "opponent_kind": "llm",
                "opponent_mode": "replay" if self._tape is not None else "live",
                # _tape_rows is empty in replay mode by design: nothing was decided here.
                "opponent_fallback": bool(self._tape_rows[-1].get("fallback"))
                if self._tape_rows else False,
            },
        )

    def close(self) -> None:
        self._state = None
        self._bot = None

    def fingerprint(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "protocol_id": "open-spiel-roshambo-policy-induction-v3",
            "opponent_id": self.opponent_id,
            "opponent_model": getattr(self.llm, "model", None),
            "mode": "replay" if self.replay_dir is not None else "live",
            "horizon": self.horizon,
            "recall": self.recall,
            "inference_context": INFERENCE_CONTEXT,
        }
