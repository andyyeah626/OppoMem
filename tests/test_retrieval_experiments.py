import numpy as np

from oppomem.memory import CentroidMemoryBank, CentroidRetriever, CosineTopKNLIReranker
from oppomem.memory.retrieval_tools import build_hard_negative_pairs, cluster_cohesion, fit_hard_negative_projection, optional_query_rewrite
from oppomem.models.semantics import NLIScores
from oppomem.types import CentroidCluster


class Embedder:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class NLI:
    def score_many(self, pairs):
        return [NLIScores(.9, .1, .0) if "good" in right else NLIScores(.1, .1, .8)
                for _, right in pairs]


def bank():
    result = CentroidMemoryBank("unused.json")
    result.clusters = {
        "bad": CentroidCluster("bad", "bad strategy", "bad response", [1.0, 0.0]),
        "good": CentroidCluster("good", "good strategy", "good response", [1.0, 0.0]),
    }
    return result


def test_topk_reranker_can_replace_dense_top1():
    audit = {}
    match = CosineTopKNLIReranker(Embedder(), NLI(), top_k=2).rank(
        "Opponent strategy: query", bank(), audit=audit)
    assert match[0].cluster_id == "good"
    assert audit["metric"] == "cosine_topk_nli_rerank"
    assert len(audit["candidates"]) == 2


def test_prototype_is_used_when_requested():
    b = bank()
    b.clusters["bad"].prototype_embedding = [0.0, 1.0]
    audit = {}
    CentroidRetriever(Embedder(), use_prototype=True).rank(
        "query", b, audit=audit)
    assert audit["prototype_mode"] is True
    assert audit["candidate_cluster_id"] == "good"


def test_hard_negative_and_cohesion_helpers_are_game_agnostic():
    pairs = build_hard_negative_pairs(
        [("q", "a", "positive")], {"a": [1, 0], "b": [.9, .1]})
    assert pairs[0].negative_cluster == "b"
    report = cluster_cohesion({"a": [[1, 0], [1, 0]]})
    assert report["a"]["size"] == 2
    matrix = fit_hard_negative_projection(
        np.asarray([[1., 0.], [1., 0.]]),
        np.asarray([[0., 1.], [0., 1.]]), steps=2)
    assert matrix.shape == (2, 2)


def test_query_rewrite_is_optional_and_keeps_original():
    assert optional_query_rewrite("q", None) == ("q", None)
    assert optional_query_rewrite("q", lambda _: " rewritten ") == ("q", "rewritten")
