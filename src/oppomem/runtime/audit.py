"""Small, dependency-free checks and human-readable RPS memory exports."""
from __future__ import annotations

from collections import Counter
import json
import os
import time
from pathlib import Path
from typing import Any


class RPSAuditError(RuntimeError):
    """A logged decision contradicts the experiment protocol."""


def audit_decision_prompt(prompt: str, *, tom: str | None,
                          memory: dict | None) -> dict[str, bool]:
    """Check the actual submitted prompt, before spending a model request."""
    tom_present = not tom or tom in prompt or json.dumps(tom, ensure_ascii=False)[1:-1] in prompt
    memory_present = '"retrieved_memory"' in prompt
    if not tom_present:
        raise RPSAuditError("ToM missing from actual decision prompt")
    if memory_present != bool(memory):
        raise RPSAuditError("Actual memory prompt differs from injection selection")
    if memory:
        for field in ("opponent_strategy", "response_strategy"):
            text = str(memory.get(field, ""))
            if not text or (text not in prompt and json.dumps(text, ensure_ascii=False)[1:-1] not in prompt):
                raise RPSAuditError(f"Injected {field} absent from actual decision prompt")
    return {"tom_present": bool(tom) and tom_present, "memory_present": memory_present}


def audit_completed_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate completed rounds and count retrieval separately from injection.

    Also accepts a trace dict, allowing per-round snapshots before end_episode.
    Prompt checking belongs before the actual LLM call; older records lacking
    actual prompt text remain auditable for selection consistency only.
    """
    trace = record.get("trace", record)
    turns = trace.get("turns", [])
    completed = {int(row["round_index"]) for row in turns}
    audit = trace.get("learning_audit", {})
    decisions = {int(row["round"]): row for row in audit.get("decision_checks", [])
                 if int(row["round"]) in completed}
    uses: dict[int, list[dict]] = {}
    for row in trace.get("memory_uses", []):
        uses.setdefault(int(row["round_index"]), []).append(row)
    for index in completed:
        if index not in decisions:
            raise RPSAuditError(f"Missing decision audit for completed round {index}")
        decision = decisions[index]
        selected = [row for row in uses.get(index, []) if row.get("selected")]
        payload = decision.get("injected_centroid")
        if "selected_memory_ids" in decision:
            expected_ids = [row["cluster_id"] for row in selected]
            if decision["selected_memory_ids"] != expected_ids:
                raise RPSAuditError(f"Round {index}: selected memory IDs disagree with memory uses")
        if bool(selected) != bool(payload) or len(selected) > 1:
            raise RPSAuditError(f"Round {index}: selected memory disagrees with injected payload")
        if payload:
            for field in ("opponent_strategy", "response_strategy"):
                if payload.get(field) != selected[0].get(field):
                    raise RPSAuditError(f"Round {index}: injected {field} differs from selected cluster")
        if "decision_prompt" in decision:
            tom = next((row["opponent_strategy"] for row in trace.get("tom_chain", [])
                        if int(row["round_index"]) == index), None)
            audit_decision_prompt(decision["decision_prompt"], tom=tom, memory=payload)
    queries = [row for row in audit.get("retrieval_checks", [])
               if int(row["round"]) in completed]
    valid = [row for row in queries if row.get("similarity") is not None]
    threshold_passes = sum(bool(row.get("selected")) for row in valid)
    injected = sum(bool(row.get("injected_centroid")) for row in decisions.values())
    withheld = sum(bool(row.get("withheld")) for row in audit.get("injection_dropout", [])
                   if int(row["round"]) in completed)
    return {
        "rounds": len(turns), "valid_queries": len(valid),
        "threshold_passes": threshold_passes, "injected": injected, "withheld": withheld,
        "return": sum(float(row["reward"]) for row in turns),
        "wins": sum(float(row["reward"]) > 0 for row in turns),
        "draws": sum(float(row["reward"]) == 0 for row in turns),
        "losses": sum(float(row["reward"]) < 0 for row in turns),
        "retrieval_reasons": dict(Counter(row.get("reason", "unknown") for row in queries)),
        "cluster_injections": dict(Counter(row["cluster_id"] for rows in uses.values()
                                             for row in rows if row.get("selected"))),
    }


def write_progress(path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    counts = audit_completed_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return counts


def check_batch_progress(progress_dir: str | Path, *, minimum_queries: int = 20, stop_after: int | None = None,
                         enforce_coverage: bool = True) -> dict[str, Any]:
    """Read one atomically written JSON summary per episode in a dedicated directory."""
    counts: Counter = Counter()
    files = sorted(Path(progress_dir).glob("*.json"))
    for path in files:
        # drvfs can briefly expose the replace boundary as a missing directory entry.
        # Never count a partial snapshot; retry and let the next poll collect it.
        row = None
        for attempt in range(3):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                break
            except FileNotFoundError:
                time.sleep(.02)
        if row is None:
            continue
        row = row.get("audit", row)
        for key in ("rounds", "valid_queries", "threshold_passes", "injected", "withheld"):
            counts[key] += int(row.get(key, 0))
    minimum_queries = minimum_queries if stop_after is None else stop_after
    if enforce_coverage and counts["valid_queries"] >= minimum_queries and counts["threshold_passes"] == 0:
        raise RPSAuditError(f"Batch has {counts['valid_queries']} valid queries and zero threshold passes")
    return {"episodes": len(files), **dict(counts)}


def write_memory_report(bank_path: str | Path, report_path: str | Path,
                        previous_bank_path: str | Path | None = None,
                        *, stats: dict | None = None, records=()) -> dict[str, int]:
    """Export every cluster verbatim, with response changes and retained members."""
    bank_path, report_path = Path(bank_path), Path(report_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    previous = json.loads(Path(previous_bank_path).read_text(encoding="utf-8")) if previous_bank_path else {}
    old = {row["cluster_id"]: row for row in previous.get("clusters", [])}
    clusters = bank.get("clusters", [])
    current = {row["cluster_id"] for row in clusters}
    changed = sum(row["cluster_id"] in old and row.get("response_strategy") != old[row["cluster_id"]].get("response_strategy") for row in clusters)
    totals = {"clusters": len(clusters), "new_clusters": len(current - old.keys()),
              "removed_clusters": len(old.keys() - current), "response_changes": changed}
    if records:
        audits = [audit_completed_record(row) for row in records]
        aggregate = Counter()
        per_cluster = Counter()
        for audit in audits:
            for key in ("rounds", "valid_queries", "threshold_passes", "injected", "withheld", "return", "wins", "draws", "losses"):
                aggregate[key] += audit[key]
            per_cluster.update(audit["cluster_injections"])
        stats = {**dict(aggregate), "cluster_injections": dict(per_cluster), **(stats or {})}
    lines = ["# RPS memory 完整检查", "", f"来源：{bank_path.name}", "",
             f"簇数量：{len(clusters)}；新增：{totals['new_clusters']}；移除：{totals['removed_clusters']}；已有簇 response 改动：{changed}。", "",
             "移除的簇不自动视为合并；合并详情以 batch update 的操作记录为准。", ""]
    if stats:
        lines += ["运行统计：", "", "```json", json.dumps(stats, ensure_ascii=False, indent=2), "```", ""]
    for index, row in enumerate(clusters, 1):
        key = row["cluster_id"]
        lines += [f"## {index}. {key}", "", "ToM: " + row.get("opponent_strategy", ""), "",
                  "Prototype: " + str(row.get("prototype_strategy") or ""), "",
                  "Response: " + row.get("response_strategy", ""), ""]
        if key in old and old[key].get("response_strategy") != row.get("response_strategy"):
            lines += ["Previous response: " + old[key].get("response_strategy", ""), ""]
        lines += [f"Retained members: {len(row.get('members') or [])}", ""]
        for member in row.get("members") or []:
            lines += ["- " + member.get("opponent_strategy", "")]
        lines += [""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return totals
