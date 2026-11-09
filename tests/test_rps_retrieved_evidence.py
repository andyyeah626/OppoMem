from copy import deepcopy

import pytest

from oppomem.memory.evolution import Segment
from oppomem.memory.evidence import build_retrieved_evidence


def record(length=4):
    return {"trace": {
        "turns": [{"round_index": index, "agent_action": "Rock",
                   "opponent_action": "Paper", "reward": -1,
                   "observation": f"history before {index}"} for index in range(length)],
        "tom_chain": [{"round_index": 1, "opponent_strategy": "repeats"}],
        "memory_uses": [
            {"round_index": 1, "cluster_id": "wrong-old", "selected": False,
             "response_strategy": "withheld advice", "score": 0.72},
            {"round_index": 2, "cluster_id": "other", "selected": True,
             "response_strategy": "other shown advice", "score": 0.75},
            {"round_index": 3, "cluster_id": "wrong-old", "selected": True,
             "response_strategy": "shown advice", "score": 0.77},
        ],
    }}


def test_wrong_retrieved_cluster_remains_target_without_assigned_segments():
    source = record()
    segment = Segment(7, 1, "vanilla", 0, "different pattern", [[1, 3]], [1, 2, 3])
    built = build_retrieved_evidence({7: source}, assigned_segments_by_cluster={"new": [segment]})
    assert set(built) == {"wrong-old", "other", "new"}
    payload = built["wrong-old"]
    assert [row["round"] for row in payload["evidence"]] == [0, 1, 2, 3]
    assert [row["round"] for row in payload["evidence"] if row["focal_retrieval"]] == [1, 3]
    assert payload["guidance_shown"] == ["shown advice"]
    assert payload["injection_contrast"]["injected"]["rounds"] == 1
    assert payload["injection_contrast"]["withheld"]["rounds"] == 1
    assert built["new"]["focal_retrieval_rows"] == 0
    assert built["new"]["assigned_segment_rows"] == 3
    assert payload["evidence"][1]["retrieved_memory"][0]["score"] == 0.72
    assert payload["evidence"][1]["memory_state"] == "withheld"
    assert payload["evidence"][2]["memory_state"] == "injected"
    assert payload["evidence"][1]["tom"]["opponent_strategy"] == "repeats"


def test_full_trajectory_never_sampled_and_input_unmodified():
    source = record(length=201)
    original = deepcopy(source)
    payload = build_retrieved_evidence({3: source})["wrong-old"]
    assert payload["evidence_rows_total"] == 201
    assert len(payload["evidence"]) == 201
    assert payload["evidence"][-1]["round"] == 200
    assert payload["evidence"][0]["self_action"] == "Rock"
    payload["evidence"][1]["retrieved_memory"][0]["score"] = 0
    assert source == original


def test_fallback_kept_as_labeled_context_but_excluded_from_contrast():
    source = record()
    source["transitions"] = [{"step": 3, "evaluator_metrics": {"opponent_fallback": True}}]
    payload = build_retrieved_evidence({1: source})["wrong-old"]
    assert len(payload["evidence"]) == 4
    assert payload["evidence"][3]["opponent_fallback"] is True
    assert payload["excluded_fallback_rounds"] == 1
    assert payload["injection_contrast"]["injected"]["rounds"] == 0
    assert payload["injection_contrast"]["withheld"]["rounds"] == 1


def test_merged_segment_episodes_deduplicated_and_retrieval_episodes_combined():
    first, second = record(), record()
    second["trace"]["memory_uses"] = []
    segment = Segment(2, 1, "vanilla", 0, "pattern", [[0, 3]], [0, 1, 2, 3])
    built = build_retrieved_evidence({1: first, 2: second},
                                   assigned_segments_by_cluster={"wrong-old": [segment, segment]})
    payload = built["wrong-old"]
    assert len(payload["evidence"]) == 8
    assert payload["assigned_segment_rows"] == 4
    assert payload["focal_retrieval_rows"] == 2
    assert payload["injection_contrast"]["no_match"]["rounds"] == 0


@pytest.mark.parametrize("mutation", ["missing_round", "duplicate_use", "nonbool_selected"])
def test_inconsistent_retrieval_audit_fails_closed(mutation):
    source = record()
    uses = source["trace"]["memory_uses"]
    if mutation == "missing_round":
        uses[0]["round_index"] = 99
    elif mutation == "duplicate_use":
        uses.append(deepcopy(uses[0]))
    else:
        uses[0]["selected"] = "False"
    with pytest.raises(ValueError):
        build_retrieved_evidence({1: source})
