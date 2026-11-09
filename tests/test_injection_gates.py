"""Injection margin and ToM-cluster overlap gates."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oppomem.agent import AgentVariant, ToMMemoryAgent
from oppomem.memory import CentroidMemoryBank, CentroidRetriever, MECHANISM
from oppomem.memory.overlap import strategy_keyword_overlap
from oppomem.types import CentroidCluster, CentroidMatch


class FakeLLM:
    tom_text = "Target repeats Scissors often and alternates Rock and Paper."
    def complete_json(self, *, stage, system, prompt, max_tokens=None):
        if stage == "tom":
            return {"opponent_strategy": self.tom_text}
        if stage == "decision":
            return {"action": "Rock", "response_intent": "counter"}
        raise AssertionError(stage)


class FakeEmbedder:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


def _bank(tmp_path: Path) -> CentroidMemoryBank:
    bank = CentroidMemoryBank(tmp_path / "bank.json")
    bank.metadata = {
        "mechanism": MECHANISM,
        "game": "open_spiel_roshambo",
        "scope": "test",
        "protocol_id": "test",
        "tom_prompt_variant": "A",
    }
    bank.clusters["tom_a"] = CentroidCluster(
        cluster_id="tom_a",
        opponent_strategy="Always plays Rock regardless of history.",
        response_strategy="Play Paper.",
        centroid=[1.0, 0.0, 0.0],
        centroid_size=1,
    )
    bank.clusters["tom_b"] = CentroidCluster(
        cluster_id="tom_b",
        opponent_strategy="Repeats Scissors in streaks.",
        response_strategy="Play Rock after Scissors.",
        centroid=[0.99, 0.01, 0.0],
        centroid_size=1,
    )
    bank.save()
    bank.load()
    return bank


def test_strategy_overlap_low_for_mismatch():
    overlap = strategy_keyword_overlap(
        "Target played Rock, Paper, Rock with no clear pattern.",
        "Plays Paper repeatedly with fixed Paper bias.",
    )
    assert overlap < 0.34


def _warm_episode(agent: ToMMemoryAgent) -> None:
    agent.begin_episode()
    first = agent.decide('{"round":1,"remaining_rounds":19}', ["Rock", "Paper", "Scissors"])
    agent.record_outcome(first, "Paper", 0.0)


def test_margin_gate_withholds(tmp_path):
    bank = _bank(tmp_path)
    retriever = CentroidRetriever(FakeEmbedder(), min_similarity=0.5)
    agent = ToMMemoryAgent(
        game="open_spiel_roshambo",
        variant=AgentVariant.TOM_MEMORY,
        llm=FakeLLM(),
        bank=bank,
        retriever=retriever,
        injection_min_margin=0.03,
    )
    _warm_episode(agent)
    pending = agent.decide('{"round":1,"remaining_rounds":19}', ["Rock", "Paper", "Scissors"])
    audit = agent.trace.learning_audit["retrieval_checks"][-1]
    assert audit.get("injection_block_reason") == "below_similarity_margin"
    assert not pending.decision.selected_memory_ids


def test_overlap_gate_withholds(tmp_path):
    bank = _bank(tmp_path)
    retriever = CentroidRetriever(FakeEmbedder(), min_similarity=0.5)
    agent = ToMMemoryAgent(
        game="open_spiel_roshambo",
        variant=AgentVariant.TOM_MEMORY,
        llm=FakeLLM(),
        bank=bank,
        retriever=retriever,
        injection_tom_overlap_min=0.34,
    )
    _warm_episode(agent)
    pending = agent.decide('{"round":1,"remaining_rounds":19}', ["Rock", "Paper", "Scissors"])
    audit = agent.trace.learning_audit["retrieval_checks"][-1]
    assert audit.get("injection_block_reason") == "tom_cluster_mismatch"
    assert not pending.decision.selected_memory_ids
