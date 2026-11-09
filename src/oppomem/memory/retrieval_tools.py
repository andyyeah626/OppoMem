"""Small, game-agnostic retrieval experiments.

The production retriever only needs the classes in :mod:`core.centroid_memory`.
This module contains offline helpers for the remaining ablations in the retrieval
plan: hard-negative projection, cluster cohesion diagnostics, and optional query
rewriting.  None of these helpers knows a game's action vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from oppomem.models.semantics import cosine


@dataclass(frozen=True, slots=True)
class HardNegativePair:
    query: str
    positive: str
    negative: str
    positive_cluster: str
    negative_cluster: str


def build_hard_negative_pairs(
    queries: Sequence[tuple[str, str, str]],
    embeddings: dict[str, Sequence[float]],
    *,
    negatives_per_query: int = 1,
) -> list[HardNegativePair]:
    """Pair each query with its nearest *wrong* cluster text.

    ``queries`` contains ``(query_text, positive_cluster_id, positive_text)`` and
    ``embeddings`` maps cluster ids to representative vectors.  The nearest wrong
    cluster is selected using the same metric as production retrieval, so this is a
    faithful hard-negative diagnostic rather than a random negative sampler.
    """
    result: list[HardNegativePair] = []
    for query, positive_id, positive_text in queries:
        candidates = sorted(
            ((cosine(embeddings[positive_id], vector), cluster_id)
             for cluster_id, vector in embeddings.items()
             if cluster_id != positive_id),
            key=lambda row: (-row[0], row[1]),
        )[:max(1, int(negatives_per_query))]
        for _, negative_id in candidates:
            result.append(HardNegativePair(
                query=query,
                positive=positive_text,
                negative=negative_id,
                positive_cluster=positive_id,
                negative_cluster=negative_id,
            ))
    return result


def cluster_cohesion(
    members: dict[str, Sequence[Sequence[float]]],
) -> dict[str, dict[str, Any]]:
    """Measure within-cluster cohesion without assuming anything about a game."""
    report: dict[str, dict[str, Any]] = {}
    for cluster_id, rows in members.items():
        vectors = [np.asarray(row, dtype=np.float32) for row in rows]
        if not vectors:
            report[cluster_id] = {"size": 0, "mean_similarity": None,
                                  "min_similarity": None, "medoid_index": None}
            continue
        centroid = np.mean(vectors, axis=0)
        scores = [cosine(row, centroid) for row in vectors]
        medoid = max(range(len(vectors)), key=lambda i: scores[i])
        report[cluster_id] = {
            "size": len(vectors),
            "mean_similarity": float(np.mean(scores)),
            "min_similarity": float(np.min(scores)),
            "medoid_index": medoid,
        }
    return report


class ProjectedEmbedder:
    """Apply a learned linear projection before cosine retrieval."""

    def __init__(self, base, matrix: Sequence[Sequence[float]]) -> None:
        self.base = base
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.model_name = f"{getattr(base, 'model_name', type(base).__name__)}+projection"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self.base.encode(texts), dtype=np.float32)
        projected = vectors @ self.matrix
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return projected / np.clip(norms, 1e-12, None)


def fit_hard_negative_projection(
    positive: np.ndarray,
    negatives: np.ndarray,
    *,
    steps: int = 400,
    learning_rate: float = 0.02,
    margin: float = 0.10,
) -> np.ndarray:
    """Fit a compact diagonal projection with a triplet hinge objective.

    This is intentionally dependency-free and conservative: it learns only feature
    rescaling, keeping the base encoder frozen.  ``positive`` and ``negatives`` have
    shape ``(n, d)`` and ``negatives[i]`` is the hard negative for row ``i``.
    """
    positive = np.asarray(positive, dtype=np.float32)
    negatives = np.asarray(negatives, dtype=np.float32)
    if positive.shape != negatives.shape or positive.ndim != 2:
        raise ValueError("positive and negatives must have the same 2-D shape")
    scale = np.ones(positive.shape[1], dtype=np.float32)
    for _ in range(max(1, int(steps))):
        p = positive * scale
        n = negatives * scale
        p /= np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None)
        n /= np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-12, None)
        violations = margin - np.sum(p * p, axis=1) + np.sum(p * n, axis=1)
        active = violations > 0
        if not np.any(active):
            break
        # Approximate subgradient for diagonal scaling; normalize after every step.
        grad = np.mean((negative := negatives[active]) ** 2 - positive[active] ** 2, axis=0)
        scale -= float(learning_rate) * grad
        scale = np.clip(scale, 0.05, 20.0)
    return np.diag(scale).astype(np.float32)


def optional_query_rewrite(
    query: str,
    rewriter: Callable[[str], str] | None,
) -> tuple[str, str | None]:
    """Return the original query plus an optional no-new-facts rewrite."""
    if rewriter is None:
        return query, None
    rewritten = " ".join(str(rewriter(query) or "").split()).strip()
    return query, rewritten or None
