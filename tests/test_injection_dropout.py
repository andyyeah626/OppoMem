"""The 50% injection dropout is the control arm, so it needs its own guarantees.

Three properties are load-bearing for the evolution run and all three are asserted
here rather than assumed:

  * With the feature off, a record is byte-identical to one produced before the
    parameter existed.  Otherwise every earlier experiment becomes incomparable.
  * A withheld round leaks nothing.  If the response strategy still reached any
    prompt, the "control" arm would silently be a treatment arm.
  * The coin sequence is a function of (seed, round) alone -- not of exploration, not
    of how many clusters the bank holds.  Otherwise the dropout pattern would drift
    as memory grows and the comparison would stop being randomized.
"""
from __future__ import annotations

import json

import numpy as np

from oppomem.games.fixed_rps import FixedRPSAdapter
from oppomem.runtime.episode import execute_episode
from oppomem.agent import AgentVariant
from oppomem.factory import build_agent
from oppomem.models.semantics import NLIScores


RESPONSE = "Play Paper while the Rock pattern persists."
OPPONENT = "The target consistently selects Rock."


class RecordingLLM:
    """Fake LLM that keeps every prompt, so leakage can be asserted, not hoped for."""

    def __init__(self) -> None:
        self.stages: list[str] = []
        self.prompts: list[tuple[str, str]] = []

    def complete_json(self, *, stage, system, prompt, max_tokens=None):
        self.stages.append(stage)
        self.prompts.append((stage, prompt))
        if stage == "tom":
            return {"opponent_strategy": OPPONENT}
        if stage == "decision":
            assert prompt.startswith("Round ")
            return {"action": "Rock",
                    "response_intent": "test the fixed visible pattern"}
        if stage == "memory_windows":
            return {"windows": [{"opponent_strategy": OPPONENT, "round_ranges": [[1, 11]]}]}
        if stage == "memory_reflection":
            return {"response_strategy": RESPONSE}
        raise AssertionError(stage)


class Embedder:
    model_name = "fake"

    def encode(self, texts):
        # Every text on the same unit vector, so cosine is always 1.0 and retrieval is
        # guaranteed to fire once the bank is non-empty.  That isolates the dropout as
        # the only reason a round can end up without an injection.
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class NLI:
    def score_many(self, pairs):
        return [NLIScores(0.99, 0.01, 0.0) for _ in pairs]

    def score(self, left, right):
        return NLIScores(0.99, 0.01, 0.0)


def make_agent(tmp_path, name="memory.json", *, load=False):
    return build_agent(
        variant=AgentVariant.TOM_MEMORY,
        llm=RecordingLLM(),
        memory_path=tmp_path / name,
        embedder=Embedder(),
        nli=NLI(),
        semantic_config={
            "memory_mechanism": "tom_centroid_v1",
            "memory_scope": "fixed-rps-test",
            "protocol_id": "fixed-rps-test",
            "fixed_policy_protocol": False,
            "tom_prompt_variant": "A",
            "retrieval_backend": "cosine",
            "retrieval_min_similarity": 0.85,
        },
        load_existing_memory=load,
        game="open_spiel_roshambo",
    )


def run(agent, *, seed=99, dropout=None, epsilon=0.0, learning=True):
    agent.learning_enabled = learning
    kwargs = {} if dropout is None else {"injection_dropout": dropout}
    return execute_episode(
        adapter=FixedRPSAdapter(horizon=12), agent=agent,
        opponent_id="rockbot", seed=seed, training=True,
        history_limit=None, exploration_epsilon=epsilon, **kwargs,
    )


def seed_bank(tmp_path, name):
    """One training episode, to leave a single retrievable cluster behind."""
    agent = make_agent(tmp_path, name)
    run(agent)
    assert len(agent.bank.clusters) == 1
    return agent


def memory_uses(record):
    return record["trace"]["memory_uses"]


# ---- 1. the feature, when off, must not exist -----------------------------------------

def test_dropout_zero_is_byte_identical_to_omitting_the_parameter(tmp_path):
    absent = run(make_agent(tmp_path, "a.json"), dropout=None)
    explicit = run(make_agent(tmp_path, "b.json"), dropout=0.0)
    # episode_id is a fresh uuid per episode; everything else must match exactly.
    absent["trace"]["episode_id"] = explicit["trace"]["episode_id"] = "pinned"
    assert json.dumps(absent, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    assert "injection_coin" not in absent["trace"]["learning_audit"]
    assert "injection_dropout" not in absent["trace"]["learning_audit"]


def test_dropout_zero_leaves_every_retrieved_memory_injected(tmp_path):
    agent = seed_bank(tmp_path, "c.json")
    record = run(agent, seed=7, dropout=0.0, learning=False)
    uses = memory_uses(record)
    assert uses, "retrieval never fired; the test cannot say anything about dropout"
    assert all(row["selected"] for row in uses)
    assert "injection_dropout" not in record["trace"]["learning_audit"]


# ---- 2. a withheld round must leak nothing --------------------------------------------

def test_full_dropout_retrieves_but_never_injects_and_never_leaks(tmp_path):
    agent = seed_bank(tmp_path, "d.json")
    agent.llm.prompts.clear()
    record = run(agent, seed=7, dropout=1.0, learning=False)

    uses = memory_uses(record)
    assert uses, "retrieval never fired; dropout cannot be the reason nothing injected"
    # The rounds are recorded as evidence, not erased: that is what makes them a
    # usable control arm rather than a gap.
    assert all(row["selected"] is False for row in uses)
    assert all(row["response_strategy"] == RESPONSE for row in uses)

    audit = record["trace"]["learning_audit"]
    assert len(audit["injection_dropout"]) == len(uses)
    assert [row["round"] for row in audit["injection_dropout"]] == [
        row["round_index"] for row in uses
    ]
    assert all(row["allow_injection"] is False for row in audit["injection_coin"])

    # Nothing memory-derived reached any prompt, including the repair prompts.
    for stage, prompt in agent.llm.prompts:
        if stage in {"decision", "decision_index"}:
            assert RESPONSE not in prompt, f"{stage} prompt leaked the response strategy"
    for row in audit["decision_checks"]:
        assert row["injected_centroid"] is None


def test_partial_dropout_splits_the_same_cluster_into_both_arms(tmp_path):
    agent = seed_bank(tmp_path, "e.json")
    record = run(agent, seed=4242, dropout=0.5, learning=False)

    uses = memory_uses(record)
    injected = [row for row in uses if row["selected"]]
    withheld = [row for row in uses if not row["selected"]]
    assert injected and withheld, "seed produced a one-sided split; pick another seed"
    # Both arms drew the same cluster, which is precisely why they are comparable.
    assert {row["cluster_id"] for row in uses} == set(agent.bank.clusters)

    audit = record["trace"]["learning_audit"]
    assert len(audit["injection_dropout"]) == len(withheld)
    allowed = {row["round"] for row in audit["injection_coin"] if row["allow_injection"]}
    assert {row["round_index"] for row in injected} <= allowed
    assert not {row["round_index"] for row in withheld} & allowed


# ---- 3. the coin must depend on (seed, round) and nothing else ------------------------

def test_coin_sequence_is_reproducible_and_independent_of_exploration(tmp_path):
    def coins(name, *, epsilon):
        agent = seed_bank(tmp_path, name)
        record = run(agent, seed=31337, dropout=0.5, epsilon=epsilon, learning=False)
        return [row["allow_injection"]
                for row in record["trace"]["learning_audit"]["injection_coin"]]

    first = coins("f.json", epsilon=0.0)
    again = coins("g.json", epsilon=0.0)
    explored = coins("h.json", epsilon=0.9)
    assert first == again
    assert first == explored
    assert len(set(first)) == 2, "seed gave a degenerate coin sequence"


def test_coin_sequence_does_not_depend_on_bank_size(tmp_path):
    """The dropout pattern must be a property of the schedule, not of memory state.

    Otherwise later generations -- which retrieve far more often -- would be compared
    against a differently-randomized control than earlier ones.
    """
    empty = make_agent(tmp_path, "i.json")
    empty_record = run(empty, seed=515, dropout=0.5, learning=False)
    assert not memory_uses(empty_record), "bank should have been empty during play"

    stocked = seed_bank(tmp_path, "j.json")
    stocked_record = run(stocked, seed=515, dropout=0.5, learning=False)
    assert memory_uses(stocked_record), "bank should have been non-empty during play"

    def coins(record):
        return [row["allow_injection"]
                for row in record["trace"]["learning_audit"]["injection_coin"]]

    assert coins(empty_record) == coins(stocked_record)


def test_dropout_out_of_range_is_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        run(make_agent(tmp_path, "k.json"), dropout=1.5)
    with pytest.raises(ValueError):
        run(make_agent(tmp_path, "l.json"), dropout=-0.1)
