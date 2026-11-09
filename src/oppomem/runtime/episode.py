from __future__ import annotations

from collections import Counter
import dataclasses
import random
from typing import Any

from oppomem.agent import AgentVariant, ToMMemoryAgent

from oppomem.games.protocol import GameAdapter


EXPLORATION_SEED_XOR = 0x544F4D
# Its own constant and its own Random instance, deliberately not shared with
# exploration: the two mechanisms must be independently reproducible, and reusing one
# stream would make the injection coins depend on whether exploration was enabled.
INJECTION_SEED_XOR = 0x494E4A


def _execute_episode(
    *, adapter: GameAdapter, agent: ToMMemoryAgent, opponent_id: str,
    seed: int, seat: int = 0, training: bool = False,
    oracle_cluster_id: str | None = None, history_limit: int = 16,
    exploration_epsilon: float = 0.0, injection_dropout: float = 0.0,
    resume_trace: dict | None = None,
) -> dict[str, Any]:
    """Run one adapter episode through the exact core lifecycle.

    `injection_dropout` withholds a retrieved memory with that probability, per
    round.  The coin is drawn BEFORE the agent decides and independently of whether
    retrieval will find anything, so the coin sequence is a pure function of
    (seed, round) and stays identical as the bank grows across generations -- the
    dropout pattern is a property of the schedule, not of the memory state.
    """
    if not 0.0 <= exploration_epsilon <= 1.0:
        raise ValueError("exploration_epsilon must be in [0,1]")
    if not 0.0 <= injection_dropout <= 1.0:
        raise ValueError("injection_dropout must be in [0,1]")
    frame = adapter.reset(
        opponent_id=opponent_id, seed=seed, seat=seat, training=training,
    )
    trace = agent.begin_episode()
    steps = 0
    transitions = []
    exploration_rng = random.Random((int(seed) & 0xFFFFFFFF) ^ EXPLORATION_SEED_XOR)
    injection_rng = random.Random((int(seed) & 0xFFFFFFFF) ^ INJECTION_SEED_XOR)
    injection_enabled = (
        injection_dropout > 0.0
        and agent.variant in {AgentVariant.TOM_MEMORY, AgentVariant.ORACLE_MEMORY}
    )
    if resume_trace is not None:
        from oppomem.types import Turn, ToMState, MemoryUse
        if exploration_epsilon:
            raise ValueError("Checkpoint resume does not support exploration")
        trace.episode_id = resume_trace["episode_id"]
        trace.turns = [Turn(**row) for row in resume_trace["turns"]]
        trace.tom_chain = [ToMState(**row) for row in resume_trace["tom_chain"]]
        trace.memory_uses = [MemoryUse(**row) for row in resume_trace["memory_uses"]]
        trace.learning_audit = resume_trace["learning_audit"]
        steps = len(trace.turns)
        frame = adapter.restore_completed(trace.turns)
        for turn in trace.turns:
            if injection_enabled:
                injection_rng.random()
            transitions.append({"step": turn.round_index, "action": turn.agent_action,
                "model_action": turn.agent_action, "explored": False,
                "opponent_event": turn.opponent_action, "reward_delta": turn.reward,
                "cumulative_return": sum(t.reward for t in trace.turns[:turn.round_index + 1]),
                "terminal": turn.round_index + 1 == len(trace.turns) and frame.terminal,
                "evaluator_metrics": {"restored_from_checkpoint": True}})
    while not frame.terminal:
        # No draw and no audit row when disabled, so a dropout-free run produces a
        # byte-identical record to one made before this parameter existed.
        allow_injection = True
        if injection_enabled:
            allow_injection = injection_rng.random() >= injection_dropout
            agent.trace.learning_audit.setdefault("injection_coin", []).append({
                "round": steps,
                "dropout": injection_dropout,
                "allow_injection": allow_injection,
            })
        pending = agent.decide(
                frame.observation,
                list(frame.legal_actions),
                action_validator=lambda value, legal=frame.legal_actions: value in legal,
                inference_context=frame.inference_context,
                history_limit=history_limit,
                oracle_cluster_id=oracle_cluster_id,
                allow_injection=allow_injection,
            )
        model_action = pending.decision.action
        explored = False
        exploration_enabled = (
            training
            and agent.variant is AgentVariant.TOM_MEMORY
            and exploration_epsilon > 0.0
            and len(frame.legal_actions) > 1
        )
        if exploration_enabled and exploration_rng.random() < exploration_epsilon:
            alternatives = [action for action in frame.legal_actions
                            if action != model_action]
            pending.decision.action = exploration_rng.choice(alternatives)
            pending.decision.response_intent = (
                "Collect unbiased training evidence through uniform exploration."
            )
            explored = True
        agent.trace.learning_audit.setdefault("exploration", []).append({
            "round": steps,
            "enabled": exploration_enabled,
            "epsilon": exploration_epsilon if exploration_enabled else 0.0,
            "model_action": model_action,
            "executed_action": pending.decision.action,
            "explored": explored,
        })
        transition = adapter.step(pending.decision.action)
        agent.record_outcome(
                pending, transition.opponent_event, transition.reward_delta,
            )
        transitions.append({
                "step": steps,
                "action": pending.decision.action,
                "model_action": model_action,
                "explored": explored,
                "opponent_event": transition.opponent_event,
                "reward_delta": transition.reward_delta,
                "cumulative_return": transition.frame.cumulative_return,
                "terminal": transition.terminal,
                "evaluator_metrics": transition.evaluator_metrics,
            })
        frame = transition.frame
        steps += 1
    completed = agent.end_episode()
    if abs(completed.total_reward - frame.cumulative_return) > 1e-8:
        raise RuntimeError(
            f"Core trace return {completed.total_reward} != adapter return "
            f"{frame.cumulative_return}"
        )
    return {
            "opponent_id": opponent_id,
            "seed": int(seed),
            "seat": int(seat),
            "variant": agent.variant.value,
            "training": bool(training),
            "status": "normal",
            "steps": steps,
            "return": frame.cumulative_return,
            "transitions": transitions,
            "trace": {
                "episode_id": completed.episode_id,
                "total_reward": completed.total_reward,
                "turns": [dataclasses.asdict(turn) for turn in completed.turns],
                "tom_chain": [dataclasses.asdict(state) for state in completed.tom_chain],
                "memory_uses": [dataclasses.asdict(use) for use in completed.memory_uses],
                "learning_audit": completed.learning_audit,
            },
            "evaluator_metrics": (transitions[-1].get("evaluator_metrics", {})
                                  if transitions else {}),
    }


def execute_episode(*, adapter, agent, training=False, **kwargs):
    """Own the episode lifecycle; evaluation cannot update memory.

    Frozen training batches can also disable online learning on the agent. Always
    release the adapter and clear unfinished traces, including failed API calls.
    """
    learning = agent.learning_enabled
    agent.learning_enabled = learning and training
    try:
        return _execute_episode(adapter=adapter, agent=agent, training=training, **kwargs)
    finally:
        agent.learning_enabled = learning
        agent.trace = None
        adapter.close()


def update_provenance(
    provenance: dict[str, dict[str, int]], opponent_id: str,
    episode_record: dict[str, Any],
) -> None:
    operations = (episode_record.get("trace", {}).get("learning_audit", {})
                  .get("centroid_memory", {}).get("operations", []))
    counts = Counter(
        row.get("cluster_id") for row in operations
        if row.get("operation") in {"created", "merged"} and row.get("cluster_id")
    )
    target = provenance.setdefault(opponent_id, {})
    for cluster_id, count in counts.items():
        target[cluster_id] = target.get(cluster_id, 0) + int(count)


def oracle_cluster(provenance: dict[str, dict[str, int]],
                   opponent_id: str) -> str | None:
    counts = provenance.get(opponent_id) or {}
    if not counts:
        return None
    return min(counts, key=lambda cluster_id: (-counts[cluster_id], cluster_id))
