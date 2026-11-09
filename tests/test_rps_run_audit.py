import json

import pytest

from oppomem.prompts.rps import player_prompt
from oppomem.runtime.audit import RPSAuditError, audit_completed_record, audit_decision_prompt, check_batch_progress, write_memory_report, write_progress


def record(selected=True):
    memory = {"opponent_strategy": "A fixed preference.", "response_strategy": "Infer the preferred action and counter it."}
    tom = "Repeats the previous action."
    prompt = player_prompt([], horizon=20, tom=tom, memory=memory if selected else None)
    return {"trace": {
        "turns": [{"round_index": 0, "reward": 1}],
        "tom_chain": [{"round_index": 0, "opponent_strategy": tom}],
        "memory_uses": [{"round_index": 0, "cluster_id": "c1", "selected": selected, **memory}],
        "learning_audit": {
            "decision_checks": [{"round": 0, "injected_centroid": memory if selected else None, "decision_prompt": prompt}],
            "retrieval_checks": [{"round": 0, "similarity": .75, "selected": True, "reason": "selected"}],
            "injection_dropout": [] if selected else [{"round": 0, "withheld": True}],
        },
    }}


def test_counts_separate_retrieval_from_hidden_memory():
    visible, hidden = audit_completed_record(record()), audit_completed_record(record(False))
    assert visible["threshold_passes"] == hidden["threshold_passes"] == 1
    assert visible["injected"] == 1
    assert hidden["injected"] == 0 and hidden["withheld"] == 1


def test_missing_tom_detected_without_memory():
    with pytest.raises(RPSAuditError, match="ToM missing"):
        audit_decision_prompt("Choose an action", tom="Repeats previous action", memory=None)


def test_missing_payload_and_changed_advice_detected():
    result = record()
    result["trace"]["learning_audit"]["decision_checks"][0]["injected_centroid"] = None
    with pytest.raises(RPSAuditError, match="selected memory"):
        audit_completed_record(result)
    result = record()
    result["trace"]["memory_uses"][0]["response_strategy"] = "Other advice"
    with pytest.raises(RPSAuditError, match="differs"):
        audit_completed_record(result)


def test_batch_guard_combines_workers_and_ignores_tmp(tmp_path):
    for index in range(2):
        (tmp_path / f"{index}.json").write_text(json.dumps({"valid_queries": 10, "threshold_passes": 0}))
    (tmp_path / "ignored.json.tmp").write_text("partial")
    with pytest.raises(RPSAuditError, match="20 valid queries"):
        check_batch_progress(tmp_path)
    (tmp_path / "1.json").write_text(json.dumps({"valid_queries": 10, "threshold_passes": 1}))
    assert check_batch_progress(tmp_path)["threshold_passes"] == 1


def test_progress_persists_and_report_keeps_different_responses(tmp_path):
    progress = tmp_path / "progress" / "episode.json"
    write_progress(progress, record())
    assert json.loads(progress.read_text())["injected"] == 1
    before = {"clusters": [{"cluster_id": "a", "opponent_strategy": "Fixed preference", "response_strategy": "Old advice", "members": []}]}
    after = {"clusters": [{"cluster_id": "a", "opponent_strategy": "Fixed preference", "response_strategy": "Infer preferred action then counter it", "members": []}, {"cluster_id": "b", "opponent_strategy": "Delayed countering", "response_strategy": "Predict the delayed response", "members": []}]}
    old_path, new_path, report_path = (tmp_path / name for name in ("old.json", "new.json", "report.md"))
    old_path.write_text(json.dumps(before))
    new_path.write_text(json.dumps(after))
    totals = write_memory_report(new_path, report_path, old_path, records=[record()])
    output = report_path.read_text(encoding="utf-8")
    assert totals["response_changes"] == 1 and totals["new_clusters"] == 1
    assert "Old advice" in output and "Predict the delayed response" in output


def test_progress_read_retries_transient_replace_gap(tmp_path, monkeypatch):
    from pathlib import Path
    path = tmp_path / "episode.json"
    path.write_text(json.dumps({"valid_queries": 20, "threshold_passes": 8}))
    original = Path.read_text
    calls = []
    def transient(self, *args, **kwargs):
        calls.append(self)
        if len(calls) == 1:
            raise FileNotFoundError(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", transient)
    assert check_batch_progress(tmp_path)["threshold_passes"] == 8
    assert len(calls) == 2
