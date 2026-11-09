"""The update phase runs off disk and writes the bank, so its logic gets tests.

The properties asserted here are the ones whose failure would be silent: a schedule
that is not opponent-uniform, an assignment that restarts the running centroid each
batch, evidence that mislabels a withheld round as injected, or a commit that stores a
centroid whose response strategy never arrived.
"""


from __future__ import annotations


from collections import Counter


import json


import numpy as np


import pytest


from oppomem.memory.evolution import LLMPool, PooledAssigner, Proposal, Segment, build_evidence, commit, contrast_of, fallback_rounds, memory_states, reflect_all, segment_batch


from oppomem.memory import CentroidMemoryBank, MECHANISM


from oppomem.models.semantics import NLIScores


from oppomem.types import CentroidCluster


class Embedder:
    """Maps a text to a unit vector by its first word, so clusters are controllable."""

    model_name = "fake"
    AXES = {"rock": (1.0, 0.0, 0.0), "paper": (0.0, 1.0, 0.0), "cycle": (0.0, 0.0, 1.0)}

    def encode(self, texts):
        rows = []
        for text in texts:
            key = text.replace("Opponent strategy: ", "").split()[0].lower()
            rows.append(self.AXES.get(key, (0.0, 0.0, 0.0)))
        return np.asarray(rows, dtype=np.float32)


class NLI:
    def __init__(self, contradiction=0.0):
        self.contradiction = contradiction

    def score_many(self, pairs):
        return [NLIScores(0.9, 0.1, self.contradiction) for _ in pairs]

    def score(self, left, right):
        return NLIScores(0.9, 0.1, self.contradiction)


class SegmenterLLM:
    def __init__(self, windows_by_episode):
        self.windows_by_episode = windows_by_episode
        self.requests_made = 0
        self.events = []
        self.prompts = []

    def complete_json(self, *, stage, system, prompt, max_tokens=None):
        self.requests_made += 1
        self.prompts.append((stage, prompt))
        if stage == "memory_windows":
            payload = json.loads(prompt.split("DATA:\n", 1)[1])
            rounds = [row["round"] for row in payload["ONLINE_TOM_CHAIN"]]
            key = tuple(sorted(rounds))
            return {"windows": self.windows_by_episode.get(key, [])}
        if stage == "memory_reflection":
            return {"response_strategy": "Play the counter while the pattern holds."}
        raise AssertionError(stage)


def make_record(run_index, *, opponent_id="rockbot", cycle=1, strategy="rock bias",
                states=None, fallback=(), reward=1.0):
    """A minimal but structurally real record, as benchmarks/runner.py would save it."""
    horizon = 4
    states = states or {}
    turns = [{
        "round_index": index,
        "observation": "{}",
        "legal_actions": ["Rock", "Paper", "Scissors"],
        "agent_action": "Paper",
        "opponent_action": "Rock",
        "reward": reward,
        "inference_context": "ctx",
        "response_intent": "counter the bias",
    } for index in range(horizon)]
    memory_uses = [{
        "round_index": index,
        "cluster_id": "tom_prior",
        "selected": state == "injected",
        "opponent_strategy": strategy,
        "response_strategy": "Prior guidance: keep playing Paper.",
        "score": 0.9,
        "metric": "cosine",
        "agent_action": "Paper",
        "opponent_action": "Rock",
        "reward": reward,
    } for index, state in states.items()]
    return {
        "run_index": run_index,
        "cycle": cycle,
        "opponent_id": opponent_id,
        "game": "open_spiel_roshambo",
        "variant": "tom_memory",
        "return": reward * horizon,
        "transitions": [{
            "step": index,
            "evaluator_metrics": ({"opponent_fallback": True} if index in fallback else {}),
        } for index in range(horizon)],
        "trace": {
            "episode_id": f"ep{run_index}",
            "turns": turns,
            "tom_chain": [{"round_index": index, "opponent_strategy": strategy,
                           "next_action_probabilities": {}} for index in range(horizon)],
            "memory_uses": memory_uses,
            "learning_audit": {},
        },
    }


def empty_bank(path):
    bank = CentroidMemoryBank(path)
    bank.metadata = {"mechanism": MECHANISM, "game": "open_spiel_roshambo",
                     "scope": "t", "protocol_id": "t"}
    bank.save()
    return CentroidMemoryBank(path).load()


def test_memory_states_separates_injected_withheld_and_no_match():
    record = make_record(0, states={0: "injected", 1: "withheld", 2: "injected"})
    assert memory_states(record) == {0: "injected", 1: "withheld", 2: "injected"}
    # Round 3 has no MemoryUse row at all, which is a different thing from withheld.
    assert 3 not in memory_states(record)


def test_fallback_rounds_are_detected_and_excluded_from_evidence():
    record = make_record(0, states={0: "injected", 1: "withheld"}, fallback=(1,))
    assert fallback_rounds(record) == {1}
    built = build_evidence(
        [Segment(0, 1, "gpt-5.6-sol", 0, "rock bias", [[0, 3]], [0, 1, 2, 3])],
        {0: record},
    )
    assert built["excluded_fallback_rounds"] == 1
    assert [row["round"] for row in built["evidence"]] == [0, 2, 3]
    assert built["injection_contrast"]["withheld"]["rounds"] == 0


def test_build_evidence_reports_guidance_and_contrast():
    record = make_record(0, states={0: "injected", 1: "withheld", 2: "withheld"},
                         reward=1.0)
    built = build_evidence(
        [Segment(0, 1, "rockbot", 0, "rock bias", [[0, 3]], [0, 1, 2, 3])], {0: record})
    assert built["guidance_shown"] == ["Prior guidance: keep playing Paper."]
    contrast = built["injection_contrast"]
    assert contrast["injected"]["rounds"] == 1
    assert contrast["withheld"]["rounds"] == 2
    assert contrast["no_match"]["rounds"] == 1
    assert {row["memory_state"] for row in built["evidence"]} == {
        "injected", "withheld", "no_match"}
    # Anonymized: no opponent name reaches the prompt payload.
    assert "rockbot" not in json.dumps(built["evidence"])


def test_contrast_difference_is_none_when_one_arm_is_empty():
    rows = [{"reward": 1.0, "memory_state": "injected"}]
    assert contrast_of(rows)["injected_minus_withheld"] is None


def test_pooled_assignment_groups_by_similarity_and_keeps_running_mean(tmp_path):
    bank = empty_bank(tmp_path / "bank.json")
    segments = [
        Segment(0, 1, "rockbot", 0, "rock bias one", [[0, 3]], [0, 1]),
        Segment(1, 1, "rockbot", 0, "rock bias two", [[0, 3]], [0, 1]),
        Segment(2, 1, "copybot", 0, "cycle through moves", [[0, 3]], [0, 1]),
    ]
    assigner = PooledAssigner(bank, Embedder(), NLI(), game="open_spiel_roshambo",
                              scope="t")
    proposals, operations = assigner.assign(segments)
    touched = {cid: p for cid, p in proposals.items() if p.segments}
    assert len(touched) == 2
    sizes = sorted(p.centroid_size for p in touched.values())
    assert sizes == [1, 2]
    assert [row["assignment"] for row in operations if row["operation"] == "assigned"] == [
        "new", "existing", "new"]


def test_pooled_assignment_continues_the_existing_running_mean(tmp_path):
    """A second batch must extend support, not restart it.

    If it restarted, `centroid_size` would stay tiny forever and every new segment
    would yank the centroid, which is the opposite of the accumulate-and-stabilize
    behaviour the generational design assumes.
    """
    path = tmp_path / "bank.json"
    bank = empty_bank(path)
    bank.clusters["tom_existing"] = CentroidCluster(
        cluster_id="tom_existing", opponent_strategy="rock bias",
        response_strategy="Play Paper.", centroid=[1.0, 0.0, 0.0], centroid_size=7)
    assigner = PooledAssigner(bank, Embedder(), NLI(), game="open_spiel_roshambo",
                              scope="t")
    proposals, _ = assigner.assign(
        [Segment(9, 1, "rockbot", 0, "rock bias again", [[0, 3]], [0])])
    assert proposals["tom_existing"].centroid_size == 8


def test_nli_contradiction_forces_a_new_cluster(tmp_path):
    path = tmp_path / "bank.json"
    bank = empty_bank(path)
    bank.clusters["tom_existing"] = CentroidCluster(
        cluster_id="tom_existing", opponent_strategy="rock bias",
        response_strategy="Play Paper.", centroid=[1.0, 0.0, 0.0], centroid_size=3)
    # Same embedding axis, so cosine passes; the NLI gate is the only thing that can
    # keep them apart.
    assigner = PooledAssigner(bank, Embedder(), NLI(contradiction=0.95),
                              game="open_spiel_roshambo", scope="t")
    proposals, operations = assigner.assign(
        [Segment(0, 1, "rockbot", 0, "rock avoidance", [[0, 3]], [0])])
    assert any(row["operation"] == "nli_rejected" for row in operations)
    new = [cid for cid, p in proposals.items() if p.segments]
    assert new == [cid for cid in new if cid != "tom_existing"]


def test_segment_batch_caches_per_episode_and_pins_order(tmp_path):
    records = [make_record(index, cycle=index + 1) for index in range(3)]
    windows = {(0, 1, 2, 3): [{"opponent_strategy": "rock bias",
                               "round_ranges": [[0, 3]]}]}
    calls = []

    def factory(index):
        llm = SegmenterLLM(windows)
        calls.append(llm)
        return llm

    pool = LLMPool(factory, 2)
    cache = tmp_path / "segments"
    segments, audit = segment_batch(records, pool, cache_dir=cache, max_workers=2)
    assert audit["newly_segmented"] == 3
    assert [segment.run_index for segment in segments] == [0, 1, 2]
    assert len(list(cache.glob("*.json"))) == 3

    # Second pass: fully cached, no further requests, identical order.
    before = pool.requests_made
    again, audit2 = segment_batch(records, pool, cache_dir=cache, max_workers=2)
    assert audit2["newly_segmented"] == 0
    assert pool.requests_made == before
    assert [s.to_dict() for s in again] == [s.to_dict() for s in segments]


def test_reflect_all_is_resumable_and_commit_skips_empty_reflections(tmp_path):
    path = tmp_path / "bank.json"
    bank = empty_bank(path)
    segments = [Segment(0, 1, "rockbot", 0, "rock bias", [[0, 3]], [0, 1, 2, 3])]
    assigner = PooledAssigner(bank, Embedder(), NLI(), game="open_spiel_roshambo",
                              scope="t")
    proposals, _ = assigner.assign(segments)
    cluster_id = next(cid for cid, p in proposals.items() if p.segments)

    record = make_record(0, states={0: "injected", 1: "withheld"})
    built = build_evidence(segments, {0: record})
    payloads = {cluster_id: {
        "canonical_opponent_strategy": "rock bias",
        "previous_response_strategy": None,
        "support_segments": 1,
        "evidence": built["evidence"],
        "guidance_shown": built["guidance_shown"],
        "injection_contrast": built["injection_contrast"],
    }}

    pool = LLMPool(lambda index: SegmenterLLM({}), 2)
    partial = tmp_path / "reflections_partial.json"
    reflections = reflect_all(payloads, pool, partial_path=partial, max_workers=2)
    assert reflections[cluster_id]["response_strategy"]
    assert partial.is_file()

    before = pool.requests_made
    reflect_all(payloads, pool, partial_path=partial, max_workers=2)
    assert pool.requests_made == before

    operations = commit(bank, proposals, reflections, Embedder(), metadata={})
    assert [row["operation"] for row in operations] == ["created"]
    reloaded = CentroidMemoryBank(path).load()
    assert reloaded.clusters[cluster_id].response_strategy
    assert len(reloaded.clusters[cluster_id].centroid) == 3

    # An empty reflection must leave no cluster behind at all.
    bank2 = empty_bank(tmp_path / "bank2.json")
    proposals2, _ = PooledAssigner(bank2, Embedder(), NLI(),
                                   game="open_spiel_roshambo", scope="t").assign(segments)
    rejected = commit(bank2, proposals2, {}, Embedder(), metadata={})
    assert [row["operation"] for row in rejected] == ["reflection_rejected"]
    assert not CentroidMemoryBank(tmp_path / "bank2.json").load().clusters


def test_batch_reflection_prompt_carries_the_contrast_and_hides_mechanics():
    from oppomem.memory import batch_reflection_prompt

    prompt = batch_reflection_prompt(
        canonical_opponent_strategy="rock bias",
        previous_response_strategy="Play Paper.",
        support_segments=4,
        evidence=[{"episode": "episode_01", "round": 0, "self_action": "Paper",
                   "target_action": "Rock", "reward": 1.0, "response_intent": "counter",
                   "memory_state": "injected"}],
        guidance_shown=["Prior guidance: keep playing Paper."],
        injection_contrast={"injected": {"rounds": 1, "mean_reward": 1.0},
                            "withheld": {"rounds": 2, "mean_reward": 0.0},
                            "injected_minus_withheld": 1.0},
    )
    payload = json.loads(prompt.split("DATA:\n", 1)[1])
    assert payload["PRIOR_RESPONSE_STRATEGY"] == "Play Paper."
    assert payload["INJECTION_CONTRAST"]["injected_minus_withheld"] == 1.0
    assert payload["GUIDANCE_SHOWN_ON_INJECTED_ROUNDS"]
    assert "response_strategy" in prompt
    # The model must not be invited to talk about the mechanism it is embedded in.
    assert "Do not mention cluster IDs" in prompt


def test_pooled_assignment_respects_max_centroid_size(tmp_path):
    path = tmp_path / "bank.json"
    bank = empty_bank(path)
    bank.clusters["tom_full"] = CentroidCluster(
        cluster_id="tom_full", opponent_strategy="rock bias",
        response_strategy="Play Paper.", centroid=[1.0, 0.0, 0.0], centroid_size=200)
    assigner = PooledAssigner(
        bank, Embedder(), NLI(), game="open_spiel_roshambo", scope="t",
        max_centroid_size=200,
    )
    proposals, operations = assigner.assign(
        [Segment(0, 1, "rockbot", 0, "rock bias again", [[0, 3]], [0])])
    assert any(row["operation"] == "size_cap_rejected" for row in operations)
    new_ids = [cid for cid, p in proposals.items() if p.segments and cid != "tom_full"]
    assert len(new_ids) == 1


def test_commit_demotes_on_nonpositive_contrast(tmp_path):
    bank = empty_bank(tmp_path / "bank.json")
    bank.clusters["tom_existing"] = CentroidCluster(
        cluster_id="tom_existing", opponent_strategy="rock bias",
        response_strategy="Keep Paper against Rock.", centroid=[1.0, 0.0, 0.0],
        centroid_size=3)
    bank.save()
    bank = CentroidMemoryBank(tmp_path / "bank.json").load()
    proposals = {
        "tom_existing": Proposal(
            cluster_id="tom_existing",
            opponent_strategy="rock bias",
            previous_response_strategy="Keep Paper against Rock.",
            centroid=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            centroid_size=4,
            base_cluster=bank.clusters["tom_existing"],
            segments=[Segment(0, 1, "rockbot", 0, "rock bias", [[0, 3]], [0])],
        ),
    }
    reflections = {"tom_existing": {"response_strategy": "Default Scissors always."}}
    contrast = {
        "tom_existing": {
            "injected": {"rounds": 25, "mean_reward": 0.0},
            "withheld": {"rounds": 25, "mean_reward": 0.1},
            "injected_minus_withheld": -0.1,
        },
    }
    commit(
        bank, proposals, reflections, Embedder(), metadata={},
        contrast_by_cluster=contrast,
    )
    saved = CentroidMemoryBank(tmp_path / "bank.json").load()
    assert saved.clusters["tom_existing"].response_strategy == "Keep Paper against Rock."


def test_llm_pool_rejects_a_zero_size():
    with pytest.raises(ValueError):
        LLMPool(lambda index: SegmenterLLM({}), 0)
