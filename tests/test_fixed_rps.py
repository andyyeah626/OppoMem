from oppomem.games.fixed_rps import FixedRPSAdapter

import sys
import pytest


def test_fixed_rps_fake_llm_full_learning_lifecycle(tmp_path):
    import json
    import numpy as np

    from oppomem.runtime.episode import execute_episode
    from oppomem.agent import AgentVariant
    from oppomem.factory import build_agent
    from oppomem.models.semantics import NLIScores

    class LLM:
        def __init__(self): self.stages = []
        def complete_json(self, *, stage, system, prompt, max_tokens=None):
            self.stages.append(stage)
            if stage == "tom":
                return {"opponent_strategy": "The target consistently selects Rock."}
            if stage == "decision":
                assert prompt.startswith("Round ")
                return {"action": "Rock",
                        "response_intent": "test the fixed visible pattern"}
            if stage == "memory_windows":
                return {"windows": [{
                    "opponent_strategy": "The target consistently selects Rock.",
                    "round_ranges": [[1, 11]],
                }]}
            if stage == "memory_reflection":
                return {"response_strategy": "Play Paper while the Rock pattern persists."}
            raise AssertionError(stage)

    class Embedder:
        model_name = "fake"
        def encode(self, texts):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    class NLI:
        def score_many(self, pairs):
            return [NLIScores(0.99, 0.01, 0.0) for _ in pairs]

    llm = LLM()
    agent = build_agent(
        variant=AgentVariant.TOM_MEMORY,
        llm=llm,
        memory_path=tmp_path / "memory.json",
        embedder=Embedder(),
        nli=NLI(),
        semantic_config={
            "memory_mechanism": "tom_centroid_v1",
            "memory_scope": "fixed-rps-test",
            "protocol_id": "fixed-rps-test",
            "fixed_policy_protocol": False,
            "tom_prompt_variant": "A",
        },
        game="open_spiel_roshambo",
    )
    record = execute_episode(
        adapter=FixedRPSAdapter(horizon=12), agent=agent,
        opponent_id="rockbot", seed=99, training=True,
        history_limit=None, exploration_epsilon=.15,
    )
    assert record["steps"] == 12
    assert len(agent.bank.clusters) == 1
    assert llm.stages.count("memory_reflection") == 1
    assert "memory_merge" not in llm.stages
    assert len(llm.stages) == 25  # No ToM call before any completed round.


def play(opponent, actions):
    adapter = FixedRPSAdapter(horizon=len(actions))
    frame = adapter.reset(opponent_id=opponent, seed=1, training=True)
    assert opponent not in frame.observation
    rows = []
    for action in actions:
        transition = adapter.step(action)
        rows.append((transition.opponent_event, transition.reward_delta))
    assert transition.terminal
    return rows


def test_fixed_rps_opponent_rules_and_rewards():
    assert [row[0] for row in play("rockbot", ["Paper"] * 4)] == ["Rock"] * 4
    assert play("rockbot", ["Paper"] * 4) == [("Rock", 1.0)] * 4
    assert [row[0] for row in play("rotatebot", ["Rock"] * 6)] == [
        "Rock", "Paper", "Scissors", "Rock", "Paper", "Scissors",
    ]
    assert [row[0] for row in play(
        "copybot", ["Rock", "Scissors", "Paper", "Rock"],
    )] == ["Paper", "Paper", "Rock", "Scissors"]


@pytest.mark.parametrize("opponent", ["randbot", "r226bot"])
def test_native_roshambo_seed_controls_the_opponent_across_isolated_processes(opponent):
    if sys.platform == "win32":
        pytest.skip("Native OpenSpiel checks run in the supported WSL environment")
    pytest.importorskip("pyspiel")
    from oppomem.games.isolated import IsolatedGameAdapter

    adapter = IsolatedGameAdapter("rps", horizon=32)
    trajectories = []
    try:
        for seed in (20261109, 20261112, 20261115, 20261109):
            # reset closes the preceding worker and starts a fresh native process.
            adapter.reset(opponent_id=opponent, seed=seed, seat=0, training=False)
            assert adapter.fingerprint()["rng"] == "posix_random"
            events = []
            for _ in range(32):
                transition = adapter.step("Rock")
                events.append(transition.opponent_event)
            assert transition.terminal
            trajectories.append(tuple(events))
    finally:
        adapter.close()
    assert trajectories[0] == trajectories[3]
    assert len(set(trajectories[:3])) == 3
