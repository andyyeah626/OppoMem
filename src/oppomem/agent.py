from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from oppomem.prompts.decision import decision_prompt, memory_payload
from oppomem.memory import CentroidMemoryBank, CentroidMemoryLearner, CentroidRetriever
from oppomem.models.protocol import JsonLLM
from oppomem.prompts.rps import PLAYER_SYSTEM, TOM_SYSTEM, RPS_GAMES
from oppomem.memory.rps import abstract_opponent_strategy, situation_query
from oppomem.prompts.ipd import IPD_GAMES, PLAYER_SYSTEM as IPD_PLAYER_SYSTEM, TOM_SYSTEM as IPD_TOM_SYSTEM, player_prompt as ipd_player_prompt, tom_prompt as ipd_tom_prompt, turn_rows as ipd_turn_rows
from oppomem.prompts.contracts import SYSTEM_JSON
from oppomem.memory.overlap import strategy_keyword_overlap
from oppomem.prompts.tom_nl import TOM_EXTRACT_SYSTEM, TOM_NL_SYSTEM, tom_extract_prompt, tom_nl_prompt
from oppomem.prompts.tom import tom_prompt
from oppomem.types import CentroidMatch, Decision, EpisodeTrace, MemoryUse, ToMState, Turn, UNKNOWN_OPPONENT_STRATEGY


def _response_intent(value: object, *, max_words: int = 20) -> str:
    """Normalize the short, model-written pre-action intent without inventing one."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()[:max_words])


class AgentVariant(str, Enum):
    VANILLA = "vanilla"
    TOM_MEMORY = "tom_memory"
    ORACLE_MEMORY = "oracle_memory"


@dataclass(slots=True)
class PendingDecision:
    decision: Decision
    observation: str
    legal_actions: list[str]
    inference_context: str
    tom: ToMState | None
    matches: list[CentroidMatch] = field(default_factory=list)
    memory_payloads: list[dict] = field(default_factory=list)


class ToMMemoryAgent:
    """Single current agent: ToM -> centroid top-1 -> decision."""

    def __init__(
        self,
        *,
        game: str,
        variant: AgentVariant,
        llm: JsonLLM,
        bank: CentroidMemoryBank | None = None,
        retriever: CentroidRetriever | None = None,
        learner: CentroidMemoryLearner | None = None,
        tom_prompt_variant: str = "A",
        tom_nl_horizon: int = 20,
        injection_min_margin: float = 0.0,
        injection_tom_overlap_min: float = 0.0,
        abstract_rps: bool = False,
    ) -> None:
        self.game = game
        self.variant = AgentVariant(variant)
        self.llm = llm
        self.bank = bank
        self.retriever = retriever
        self.learner = learner
        self.tom_prompt_variant = str(tom_prompt_variant).upper()
        # "NL" is the natural-language read plus schema extract used by the latest
        # gpt55_nl experiments; A/B/C are the single-call JSON prompts.  The horizon is
        # only read by the NL prompt (its "remaining rounds" line); the JSON prompts
        # carry no round count.
        if self.tom_prompt_variant not in {"A", "B", "C", "NL"}:
            raise ValueError("tom_prompt_variant must be A, B, C, or NL")
        self.tom_nl_horizon = max(1, int(tom_nl_horizon))
        self.injection_min_margin = float(injection_min_margin)
        self.injection_tom_overlap_min = float(injection_tom_overlap_min)
        self.abstract_rps = bool(abstract_rps and game in RPS_GAMES)
        if self.injection_min_margin < 0.0:
            raise ValueError("injection_min_margin must be >= 0")
        if not 0.0 <= self.injection_tom_overlap_min <= 1.0:
            raise ValueError("injection_tom_overlap_min must be in [0,1]")
        self.learning_enabled = True
        self.tom_start_round = 0
        self.tom_snapshot = None
        if self.variant in {AgentVariant.TOM_MEMORY, AgentVariant.ORACLE_MEMORY}:
            if bank is None or retriever is None:
                raise ValueError("tom_memory requires a centroid bank and retriever")
            if bank.metadata.get("mechanism") != "tom_centroid_v1":
                raise ValueError("tom_memory requires a tom_centroid_v1 bank")
            if bank.metadata.get("game") != game:
                raise ValueError("Centroid memory bank belongs to a different game")
        self.trace: EpisodeTrace | None = None

    def begin_episode(self) -> EpisodeTrace:
        self.trace = EpisodeTrace(
            episode_id=uuid.uuid4().hex,
            game=self.game,
            variant=self.variant.value,
        )
        return self.trace

    def _history(self, turns=None) -> str:
        if self.trace is None:
            raise RuntimeError("Call begin_episode first")
        if turns is None:
            return self.trace.trajectory_text()
        initial = ("Before our first decision: " + repr(self.trace.initial_events) + "\n"
                   if self.trace.initial_events else "")
        return initial + "\n".join(
            f"r{turn.round_index}: observation={turn.observation}; "
            f"self={turn.agent_action}; opponent={turn.opponent_action}; reward={turn.reward:+g}"
            for turn in turns
        )

    def decide(
        self,
        observation: str,
        legal_actions: list[str],
        *,
        action_validator: Callable[[str], bool] | None = None,
        tom_observation: str | None = None,
        inference_context: str = "",
        history_limit: int | None = None,
        oracle_cluster_id: str | None = None,
        allow_injection: bool = True,
    ) -> PendingDecision:
        """One round.  `allow_injection=False` withholds a retrieved memory.

        Retrieval still runs and `matches` is still returned, so the round is recorded
        as "retrieved but not injected" rather than disappearing: `record_outcome`
        derives `MemoryUse.selected` from `decision.selected_memory_ids`, which stays
        empty.  That turns a withheld round into the control arm of a within-episode
        randomized comparison instead of a hole in the log.  Nothing memory-derived
        reaches any prompt on such a round -- `payloads` stays empty, and every prompt
        below reads `payloads[0] if payloads else None`.
        """
        if self.trace is None:
            raise RuntimeError("Call begin_episode first")
        round_index = len(self.trace.turns)
        tom = None
        matches: list[CentroidMatch] = []
        payloads: list[dict] = []
        selected_ids: list[str] = []
        prompt_turns = (self.trace.turns[-history_limit:]
                        if history_limit and len(self.trace.turns) > history_limit
                        else self.trace.turns)
        prompt_history = self._history(prompt_turns)

        if self.variant is AgentVariant.TOM_MEMORY and round_index >= self.tom_start_round:
            if oracle_cluster_id is not None:
                raise ValueError("oracle_cluster_id is only valid for oracle_memory")
            previous_hypothesis = (
                self.trace.tom_chain[-1].opponent_strategy
                if self.trace.tom_chain else None
            )
            tom_audit: dict = {"variant": self.tom_prompt_variant}
            if self.tom_snapshot is not None:
                from oppomem.prompts import rps, ipd
                rules = rps if self.game in RPS_GAMES else ipd
                base = (rps.tom_prompt_text(observation, prompt_turns, previous=previous_hypothesis)
                        if self.game in RPS_GAMES else
                        ipd.tom_prompt(prompt_turns, previous=previous_hypothesis, horizon=self.tom_nl_horizon))
                base = base.removesuffix(rules.TOM_TASK) + self.tom_snapshot["tom_task"]
                raw_tom = {"opponent_strategy": self.llm.complete_text(stage="tom",
                    system=self.tom_snapshot["tom_system"], prompt=base, max_tokens=256)}
            elif self.game in RPS_GAMES | IPD_GAMES:
                raw_tom = (self.llm.complete_json(
                    stage="tom", system=(TOM_SYSTEM if self.game in RPS_GAMES else IPD_TOM_SYSTEM),
                    prompt=(tom_prompt(self.game, "", observation, legal_actions,
                                       prompt_turns, previous_hypothesis=previous_hypothesis)
                            if self.game in RPS_GAMES else
                            ipd_tom_prompt(prompt_turns, previous=previous_hypothesis,
                                           horizon=self.tom_nl_horizon)),
                    max_tokens=12288,
                ) if prompt_turns else {"opponent_strategy": UNKNOWN_OPPONENT_STRATEGY})
                tom_audit.update(tom_raw=raw_tom["opponent_strategy"],
                                 tom_extracted=raw_tom, protocol="plain-text-local-wrap")
            elif self.tom_prompt_variant == "NL":
                # Natural-language read first, then a compact schema extract, exactly as
                # rps_tom_text.py drives it.  A zero-action trajectory contains no
                # behavioral evidence, so the calls are skipped there (the latest
                # experiment does the same) and the round carries the insufficient-
                # evidence sentinel, which normalize_windows excludes from memory.
                if prompt_turns:
                    nl_read = self.llm.complete_text(
                        stage="tom_nl",
                        system=TOM_NL_SYSTEM,
                        prompt=tom_nl_prompt(
                            prompt_turns,
                            horizon=self.tom_nl_horizon,
                            previous_read=previous_hypothesis,
                            inference_context=inference_context,
                        ),
                        # Budget is generous on purpose.  max_tokens is a ceiling, not a
                        # reservation, so headroom costs nothing -- while a ceiling set
                        # just above the measured need is expensive: DeepSeek bills hidden
                        # reasoning against it, this stage averages ~2.2k reasoning +
                        # ~2.4k visible tokens, and the prompt asks for 3-8 sentences.  At
                        # the original 4096 (floored from the class's 4000 minimum) the
                        # first attempt ran out mid-sentence 21% of the time and the batch
                        # paid for a full second call, reasoning included, every time.
                        max_tokens=12288,
                    )
                    extracted = self.llm.complete_json(
                        stage="tom_extract",
                        system=TOM_EXTRACT_SYSTEM,
                        prompt=tom_extract_prompt(prompt_turns, nl_read),
                        max_tokens=256,
                    )
                    tom_audit.update(tom_raw=nl_read, tom_extracted=extracted)
                    raw_tom = {"opponent_strategy": extracted.get("opponent_strategy")}
                else:
                    raw_tom = {"opponent_strategy": UNKNOWN_OPPONENT_STRATEGY}
            else:
                raw_tom = self.llm.complete_json(
                    stage="tom",
                    system=SYSTEM_JSON,
                    prompt=tom_prompt(
                        self.game, prompt_history,
                        tom_observation if tom_observation is not None else observation,
                        legal_actions,
                        prompt_turns,
                        inference_context=inference_context,
                        previous_hypothesis=previous_hypothesis,
                        variant=self.tom_prompt_variant,
                    ),
                    max_tokens=150,
                )
                # A zero-action trajectory contains no behavioral evidence. Keep the
                # required one-call-per-round protocol, but prevent first-turn guesses
                # from entering retrieval or the episode memory.
                if not prompt_turns:
                    raw_tom = {"opponent_strategy": UNKNOWN_OPPONENT_STRATEGY}
            if self.abstract_rps:
                # Keep telemetry useful without allowing the model's raw concrete
                # strategy wording to be mistaken for the persisted ToM state.
                for key in ("tom_raw", "tom_extracted"):
                    value = tom_audit.get(key)
                    if isinstance(value, str):
                        tom_audit[key] = abstract_opponent_strategy(value)
                    elif isinstance(value, dict) and value.get("opponent_strategy"):
                        value = dict(value)
                        value["opponent_strategy"] = abstract_opponent_strategy(
                            value["opponent_strategy"]
                        )
                        tom_audit[key] = value
            # SELF's latest completed throw, so retrieval can key on the current
            # interaction state rather than the opponent description alone.
            last_self_action = next(
                (turn.agent_action for turn in reversed(prompt_turns)
                 if turn.agent_action), "",
            )
            tom_text = str(
                raw_tom.get("opponent_strategy") or UNKNOWN_OPPONENT_STRATEGY
            ).strip()
            if self.abstract_rps:
                tom_text = abstract_opponent_strategy(tom_text)
            tom = ToMState(
                round_index=round_index,
                opponent_strategy=tom_text,
                next_action_probabilities=self._action_probabilities(raw_tom)
                if self.game == "hanabi" else {},
                self_last_action=last_self_action,
            )
            self.trace.tom_chain.append(tom)
            if self.tom_prompt_variant == "NL":
                self.trace.learning_audit.setdefault("tom_reads", []).append({
                    "round": round_index, **tom_audit,
                })
            retrieval_audit = {"round": round_index}
            if tom.opponent_strategy.casefold() == UNKNOWN_OPPONENT_STRATEGY.casefold():
                retrieval_audit.update(
                    reason="insufficient_evidence", selected=False,
                    threshold=getattr(self.retriever, "min_similarity", None),
                )
            else:
                query = tom.retrieval_text(
                    situation=(situation_query(prompt_turns, round_index=round_index)
                               if self.abstract_rps else None)
                )
                matches = self.retriever.rank(  # type: ignore[union-attr,arg-type]
                    query, self.bank, audit=retrieval_audit,
                )
            self.trace.learning_audit.setdefault("retrieval_checks", []).append(
                retrieval_audit
            )
            if matches:
                if allow_injection and not self._should_inject(
                    tom, matches[0], retrieval_audit,
                ):
                    self._record_withheld(round_index, matches[0])
                elif allow_injection:
                    selected_ids = [matches[0].cluster_id]
                    payloads = [memory_payload(matches[0])]
                else:
                    self._record_withheld(round_index, matches[0])
        elif self.variant is AgentVariant.ORACLE_MEMORY:
            if oracle_cluster_id is not None:
                cluster = self.bank.clusters.get(oracle_cluster_id)  # type: ignore[union-attr]
                if cluster is None:
                    raise ValueError(f"Unknown oracle cluster {oracle_cluster_id!r}")
                match = CentroidMatch(
                    cluster_id=cluster.cluster_id,
                    opponent_strategy=cluster.opponent_strategy,
                    response_strategy=cluster.response_strategy,
                    score=1.0,
                    metric="oracle",
                )
                matches = [match]
                if allow_injection:
                    selected_ids = [match.cluster_id]
                    payloads = [memory_payload(match)]
                else:
                    self._record_withheld(round_index, match)
        elif oracle_cluster_id is not None:
            raise ValueError("oracle_cluster_id is only valid for oracle_memory")

        actual_decision_prompt = (decision_prompt(
                self.game, observation, prompt_history, legal_actions,
                tom.retrieval_text() if tom else None,
                payloads[0] if payloads else None, prompt_turns,
                inference_context=inference_context,
            ) if self.game not in IPD_GAMES else
                    ipd_player_prompt(ipd_turn_rows(prompt_turns), horizon=self.tom_nl_horizon,
                                      round_number=round_index + 1,
                                      tom=tom.opponent_strategy if tom else None,
                                      memory=payloads[0] if payloads else None))
        if self.abstract_rps:
            from oppomem.runtime.audit import audit_decision_prompt
            audit_decision_prompt(actual_decision_prompt,
                                  tom=tom.retrieval_text() if tom else None,
                                  memory=payloads[0] if payloads else None)
        raw_decision = self.llm.complete_json(
            stage="decision",
            system=(PLAYER_SYSTEM if self.game in RPS_GAMES else
                    IPD_PLAYER_SYSTEM if self.game in IPD_GAMES else SYSTEM_JSON),
            prompt=actual_decision_prompt,
            # Same ceiling argument as the NL read above: the decision stage averages
            # ~0.9k reasoning + ~1.0k visible tokens, so the 4000 floor left ~9% of calls
            # truncated and retried.
            max_tokens=8192,
        )
        action = str(raw_decision.get("action", ""))
        proposed_action = action
        accepts = action_validator or (lambda value: value in legal_actions)

        # Exact legal-action repair is useful for every enumerated environment,
        # including GTBench. Both variants share this serialization safeguard.
        if legal_actions:
            for _ in range(2):
                if accepts(action):
                    break
                raw_decision = self.llm.complete_json(
                    stage="decision",
                    system=(PLAYER_SYSTEM if self.game in RPS_GAMES else
                            IPD_PLAYER_SYSTEM if self.game in IPD_GAMES else SYSTEM_JSON),
                    prompt=(decision_prompt(self.game, observation, "", legal_actions,
                                            tom.retrieval_text() if tom else None,
                                            payloads[0] if payloads else None, prompt_turns)
                            if self.game in RPS_GAMES else
                            ipd_player_prompt(ipd_turn_rows(prompt_turns), horizon=self.tom_nl_horizon,
                                              round_number=round_index + 1,
                                              tom=tom.opponent_strategy if tom else None,
                                              memory=payloads[0] if payloads else None)
                            if self.game in IPD_GAMES else (
                        f"Your proposed action {action!r} is not currently legal. "
                        f"Current observation: {observation}\n"
                        f"Centroid memory: {payloads[0] if payloads else None}\n"
                        "Follow the current action contract exactly. Enumerated actions "
                        f"must match this list: {json.dumps(legal_actions)}. "
                        'Return exactly {"action":"valid action",'
                        '"response_intent":"concise pre-action strategy intent"}. '
                        "The intent must use visible information and be at most 20 words."
                    )),
                    max_tokens=128,
                )
                action = str(raw_decision.get("action", ""))
            if self.game not in RPS_GAMES | IPD_GAMES and not accepts(action) and legal_actions and all(accepts(x) for x in legal_actions):
                indexed = self.llm.complete_json(
                    stage="decision_index",
                    system=SYSTEM_JSON,
                    prompt=(
                        "Choose one legal option by its zero-based integer index. "
                        "Do not compose or modify an action. "
                        f"Current observation: {observation}\n"
                        f"History: {prompt_history}\n"
                        f"ToM: {tom.retrieval_text() if tom else None}\n"
                        f"Centroid memory: {payloads[0] if payloads else None}\n"
                        f"Options: {json.dumps(dict(enumerate(legal_actions)))}\n"
                        'Return exactly {"index":0,"response_intent":'
                        '"concise pre-action strategy intent"}. '
                        "The intent must use visible information and be at most 20 words."
                    ),
                    max_tokens=64,
                )
                index = indexed.get("index")
                if type(index) is int and 0 <= index < len(legal_actions):
                    self.trace.learning_audit.setdefault("action_index_repairs", []).append({
                        "round": round_index,
                        "invalid_action": action,
                        "index": index,
                    })
                    action = legal_actions[index]
                    raw_decision = indexed

        if not accepts(action):
            raise ValueError(f"LLM returned illegal action {action!r}; expected one of {legal_actions}")

        self.trace.learning_audit.setdefault("decision_checks", []).append({
            "round": round_index,
            "decision_prompt": actual_decision_prompt,
            "tom_text": tom.retrieval_text() if tom else None,
            "selected_memory_ids": list(selected_ids),
            "proposed_action": proposed_action,
            "executed_action": action,
            "injected_centroid": payloads[0] if payloads else None,
            "serialization_changed_action": action != proposed_action,
        })
        return PendingDecision(
            decision=Decision(
                action=action,
                selected_memory_ids=selected_ids,
                response_intent=_response_intent(raw_decision.get("response_intent")),
            ),
            observation=observation,
            legal_actions=list(legal_actions),
            inference_context=inference_context,
            tom=tom,
            matches=matches,
            memory_payloads=payloads,
        )

    def _should_inject(
        self,
        tom: ToMState | None,
        match: CentroidMatch,
        retrieval_audit: dict,
    ) -> bool:
        """Optional gates: top1 margin and live ToM vs cluster keyword overlap."""
        if self.injection_min_margin > 0.0:
            similarity = retrieval_audit.get("similarity")
            second = retrieval_audit.get("second_similarity")
            if similarity is not None and second is not None:
                if float(similarity) - float(second) < self.injection_min_margin:
                    retrieval_audit["injection_blocked"] = True
                    retrieval_audit["injection_block_reason"] = "below_similarity_margin"
                    return False
        if self.injection_tom_overlap_min > 0.0 and tom is not None:
            overlap = strategy_keyword_overlap(
                tom.opponent_strategy, match.opponent_strategy,
            )
            retrieval_audit["tom_cluster_overlap"] = round(overlap, 4)
            if overlap < self.injection_tom_overlap_min:
                retrieval_audit["injection_blocked"] = True
                retrieval_audit["injection_block_reason"] = "tom_cluster_mismatch"
                return False
        return True

    def _record_withheld(self, round_index: int, match: CentroidMatch) -> None:
        """Log which memory a withheld round would have received.

        The counterfactual is the whole point: without the cluster id and score, a
        withheld round could not be matched against the injected rounds that drew the
        same cluster, and the dropout would cost evidence instead of producing it.
        """
        if self.trace is None:
            raise RuntimeError("Call begin_episode first")
        self.trace.learning_audit.setdefault("injection_dropout", []).append({
            "round": round_index,
            "cluster_id": match.cluster_id,
            "score": float(match.score),
            "metric": match.metric,
            "withheld": True,
        })

    @staticmethod
    def _action_probabilities(raw: object) -> dict[str, float]:
        values = raw.get("next_action_probabilities") if isinstance(raw, dict) else None
        if not isinstance(values, dict):
            return {"hint": 1 / 3, "play": 1 / 3, "discard": 1 / 3}
        try:
            result = {key: max(0.0, float(values[key])) for key in ("hint", "play", "discard")}
        except (KeyError, TypeError, ValueError):
            return {"hint": 1 / 3, "play": 1 / 3, "discard": 1 / 3}
        total = sum(result.values())
        if total <= 0:
            return {"hint": 1 / 3, "play": 1 / 3, "discard": 1 / 3}
        return {key: value / total for key, value in result.items()}

    def record_outcome(
        self,
        pending: PendingDecision,
        opponent_action: str,
        reward: float,
    ) -> None:
        if self.trace is None:
            raise RuntimeError("Call begin_episode first")
        round_index = len(self.trace.turns)
        self.trace.turns.append(Turn(
            round_index=round_index,
            observation=pending.observation,
            legal_actions=pending.legal_actions,
            agent_action=pending.decision.action,
            opponent_action=opponent_action,
            reward=float(reward),
            inference_context=pending.inference_context,
            response_intent=pending.decision.response_intent,
        ))
        for match in pending.matches:
            self.trace.memory_uses.append(MemoryUse(
                round_index=round_index,
                cluster_id=match.cluster_id,
                selected=match.cluster_id in pending.decision.selected_memory_ids,
                opponent_strategy=match.opponent_strategy,
                response_strategy=match.response_strategy,
                score=match.score,
                metric=match.metric,
                agent_action=pending.decision.action,
                opponent_action=opponent_action,
                reward=float(reward),
            ))

    def end_episode(self) -> EpisodeTrace:
        if self.trace is None:
            raise RuntimeError("Call begin_episode first")
        completed = self.trace
        if self.variant is AgentVariant.TOM_MEMORY and self.learning_enabled:
            if self.learner is None:
                raise RuntimeError("Training requires a centroid learner")
            self.learner.learn(completed)
        self.trace = None
        return completed
