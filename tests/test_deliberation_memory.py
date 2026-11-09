import copy
import hashlib
import json

import pytest

from oppomem.deliberation.memory import METRICS, PREFIX, PROTOCOL, apply_patch, export_review, load_bank, retrieve


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[0.0, 1.0] if "concession" in text else [1.0, 0.0] for text in texts]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode(bank=None, ident="smoke"):
    return {"spec": {"id": ident, "arm": "tom_memory", "seed": 7,
                     "bank_sha256": sha(bank) if bank else None, "focal_agent": "SportCo",
                     "opponent_utilities": "SECRET SPEC"},
            "complete": True, "focal_initial_prompt": "Your own reservation value is 30.",
            "turns": [
                {"index": 0, "agent": "OtherCo", "phase": "opening", "public_answer": "I prefer a central site.",
                 "plan": "SECRET PLAN", "tom": [{"private": "SECRET TOM"}], "private_utility": 91},
                {"index": 1, "agent": "SportCo", "phase": "discussion", "public_answer": "What would you trade for location?",
                 "plan": "Ask which issue is flexible.", "tom": [{"opponent": "OtherCo", "description": "Firm location preference.",
                    "retrieval": {"cluster_id": "delib_firm", "opponent_strategy": "A firm preference.",
                                  "response_strategy": "Ask about flexible issues.", "score": .9,
                                  "injected": True, "oracle": "SECRET RETRIEVAL"}, "private": "SECRET TOM EXTRA"}]},
            ],
            "metrics": {"final_valid": True, "final_success": True, "any_success": True,
                        "focal_base_utility": 50, "focal_realized_utility": 50,
                        "focal_wrong_deal_count": 0, "collective_utility": "SECRET METRIC",
                        "wrong_deals": 12, "opponent_utilities": [91]},
            "oracle": "SECRET ROOT"}


def entry(ident="delib_firm", description="The opponent holds one issue firm but may trade across issues."):
    return {"cluster_id": ident, "opponent_strategy": description,
            "response_strategy": "Ask which other issue is flexible before offering a conditional trade.",
            "reason": "The public response states a preference without revealing flexibility.",
            "evidence": [{"episode_id": "smoke", "turn_index": 0}]}


def patch_for(run, batch, entries, deleted=None):
    review = run / "reviews" / f"batch{batch}.json"
    return {"source_bank_sha256": json.loads(review.read_text(encoding="utf-8"))["source_bank_sha256"],
            "review_sha256": sha(review), "entries": entries, "deleted": deleted or []}


def bootstrap(run):
    export_review(run, 0, [episode()], None)
    patch = write(run / "patch0.json", patch_for(run, 0, [entry()]))
    encoder = Encoder()
    return apply_patch(run, 0, patch, encoder), encoder


def test_review_allowlist_retains_public_and_focal_evidence_only(tmp_path):
    record = episode()
    original = copy.deepcopy(record)
    path = export_review(tmp_path, 0, [record], None)
    text = path.read_text(encoding="utf-8")
    assert "SECRET" not in text
    review = json.loads(text)
    exported = review["episodes"][0]
    assert "plan" not in exported["turns"][0]
    assert "tom" not in exported["turns"][0]
    assert exported["turns"][1]["plan"] == record["turns"][1]["plan"]
    assert exported["focal_initial_prompt"] == record["focal_initial_prompt"]
    assert "wrong_deals" not in exported["metrics"]
    assert set(exported["metrics"]).issubset(METRICS)
    assert record == original
    assert export_review(tmp_path, 0, [record], None) == path
    record["turns"][0]["public_answer"] = "A different answer"
    with pytest.raises(ValueError, match="Cannot replace"):
        export_review(tmp_path, 0, [record], None)


def test_empty_opening_and_final_focal_plans_are_omitted(tmp_path):
    record = episode()
    record["turns"][1]["plan"] = ""
    review = json.loads(export_review(tmp_path, 0, [record], None).read_text(encoding="utf-8"))
    assert "plan" not in review["episodes"][0]["turns"][1]
    assert review["episodes"][0]["turns"][1]["tom"][0]["description"] == "Firm location preference."


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(complete=False),
    lambda r: r["turns"][1].update(index=0),
    lambda r: r["turns"][0].update(index=2),
    lambda r: r["spec"].update(seed={"secret": 1}),
    lambda r: r["metrics"].update(focal_base_utility={"other": 99}),
    lambda r: r["turns"][1]["tom"][0]["retrieval"].update(injected={"secret": 1}),
])
def test_invalid_or_nested_review_fields_rejected(tmp_path, mutation):
    record = episode()
    mutation(record)
    with pytest.raises(ValueError):
        export_review(tmp_path, 0, [record], None)
    assert not (tmp_path / "reviews/batch0.json").exists()


def test_bootstrap_vectors_and_idempotent_apply_without_encoder(tmp_path):
    path, encoder = bootstrap(tmp_path)
    bank = load_bank(path)
    assert bank["metadata"]["game"] == "llm_deliberation"
    assert bank["metadata"]["protocol"] == PROTOCOL
    assert bank["metadata"]["external_reflection"] is False
    cluster = bank["clusters"][0]
    assert cluster["centroid"] == cluster["prototype_embedding"] == [1., 0.]
    assert cluster["prototype_strategy"] == cluster["opponent_strategy"]
    assert encoder.calls == [[PREFIX + cluster["opponent_strategy"]]]
    original = path.read_bytes()
    assert apply_patch(tmp_path, 0, tmp_path / "patch0.json") == path
    assert path.read_bytes() == original


def test_retrieve_threshold_prefix_and_tie_order(tmp_path):
    path, _ = bootstrap(tmp_path)
    bank = load_bank(path)
    second = copy.deepcopy(bank["clusters"][0])
    second["cluster_id"] = "delib_a"
    bank["clusters"].append(second)
    encoder = Encoder()
    match = retrieve(bank, "firm preference", encoder)
    assert match["cluster_id"] == "delib_a"
    assert match["score"] == 1
    assert encoder.calls == [[PREFIX + "firm preference"]]
    assert retrieve(bank, "concession", encoder) is None
    assert retrieve(bank, "concession", encoder, threshold=0) is not None
    with pytest.raises(ValueError, match="threshold"):
        retrieve(bank, "firm", encoder, threshold=float("nan"))
    empty = copy.deepcopy(bank)
    empty["clusters"] = []
    assert retrieve(empty, "firm", encoder) is None


def test_revise_description_reencodes_and_deleted_cluster_stays_deleted(tmp_path):
    path, _ = bootstrap(tmp_path)
    export_review(tmp_path, 1, [episode(path)], path)
    revised = entry(description="The opponent makes a concession after a reciprocal offer.")
    added = entry("delib_new")
    patch = write(tmp_path / "patch1.json", patch_for(tmp_path, 1, [revised, added]))
    encoder = Encoder()
    updated = apply_patch(tmp_path, 1, patch, encoder)
    bank = load_bank(updated)
    assert bank["clusters"][0]["centroid"] == [0., 1.]
    assert bank["clusters"][0]["prototype_embedding"] == [0., 1.]
    assert len(encoder.calls[0]) == 2
    export_review(tmp_path, 2, [episode(updated)], updated)
    deletion = {key: added[key] for key in ("cluster_id", "reason", "evidence")}
    patch2 = write(tmp_path / "patch2.json", patch_for(tmp_path, 2, [revised], [deletion]))
    encoder2 = Encoder()
    result = load_bank(apply_patch(tmp_path, 2, patch2, encoder2))
    assert encoder2.calls == []
    assert [c["cluster_id"] for c in result["clusters"]] == ["delib_firm"]
    assert load_bank(path)["clusters"][0]["centroid"] == [1., 0.]


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p.update(review_sha256="0" * 64), "Stale"),
    (lambda p: p["entries"][0].update(evidence=[{"episode_id": "unknown", "turn_index": 0}]), "unknown"),
    (lambda p: p["entries"][0].update(reason=""), "reason"),
    (lambda p: p["entries"][0].update(response_strategy="word " * 41), "40 words"),
    (lambda p: p["entries"][0].update(opponent_strategy="word " * 61), "60 words"),
    (lambda p: p["entries"][0].update(opponent_strategy="p2 always wants the central site."), "abstract"),
    (lambda p: p["entries"][0].update(opponent_strategy="Its threshold is 50."), "abstract"),
    (lambda p: p["entries"].append(copy.deepcopy(p["entries"][0])), "duplicate"),
])
def test_patch_rejects_invalid_evidence_strings_and_provenance(tmp_path, mutation, error):
    export_review(tmp_path, 0, [episode()], None)
    patch = patch_for(tmp_path, 0, [entry()])
    mutation(patch)
    path = write(tmp_path / "bad.json", patch)
    with pytest.raises(ValueError, match=error):
        apply_patch(tmp_path, 0, path, Encoder())
    assert not (tmp_path / "banks/M0.json").exists()


def test_previous_cluster_requires_explicit_decision(tmp_path):
    source, _ = bootstrap(tmp_path)
    export_review(tmp_path, 1, [episode(source)], source)
    patch = write(tmp_path / "patch1.json", patch_for(tmp_path, 1, []))
    with pytest.raises(ValueError, match="Every previous"):
        apply_patch(tmp_path, 1, patch, Encoder())


def test_bank_mutation_and_committed_revision_cannot_be_silently_reused(tmp_path):
    source, _ = bootstrap(tmp_path)
    export_review(tmp_path, 1, [episode(source)], source)
    revision = patch_for(tmp_path, 1, [entry()])
    patch = write(tmp_path / "patch1.json", revision)
    target = apply_patch(tmp_path, 1, patch, Encoder())
    prior_bytes = target.read_bytes()
    revision["entries"][0]["response_strategy"] = "Ask for a reciprocal concession first."
    write(patch, revision)
    with pytest.raises(ValueError, match="Cannot replace"):
        apply_patch(tmp_path, 1, patch, Encoder())
    assert target.read_bytes() == prior_bytes
    modified = json.loads(source.read_text(encoding="utf-8"))
    modified["metadata"]["tampered"] = True
    write(source, modified)
    with pytest.raises(ValueError, match="Source bank changed"):
        apply_patch(tmp_path, 1, patch, Encoder())


@pytest.mark.parametrize("mutation", [
    lambda b: b.update(schema_version=6),
    lambda b: b["metadata"].update(game="rps"),
    lambda b: b["clusters"][0].update(prototype_embedding=[0., 1.]),
    lambda b: b["clusters"][0].update(centroid=[float("nan"), 0]),
    lambda b: b["clusters"].append(copy.deepcopy(b["clusters"][0])),
])
def test_load_rejects_incompatible_or_corrupt_banks(tmp_path, mutation):
    path, _ = bootstrap(tmp_path)
    bank = load_bank(path)
    mutation(bank)
    wrong = write(tmp_path / "wrong.json", bank)
    with pytest.raises(ValueError):
        load_bank(wrong)
