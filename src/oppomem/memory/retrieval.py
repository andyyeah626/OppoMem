"""Extracted retrieval responsibilities of the working memory implementation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from oppomem.prompts.history import completed_joint_actions
from oppomem.prompts.contracts import OPPONENT_MODEL_CONTRACT, RESPONSE_CONTRACT, SYSTEM_JSON
from oppomem.models.semantics import cosine, encode_documents, encode_query
from oppomem.types import CentroidCluster, CentroidMatch, EpisodeTrace, UNKNOWN_OPPONENT_STRATEGY
from oppomem.memory.rps import abstract_opponent_strategy, abstract_response_strategy, sanitize_cluster_texts

from .bank import CentroidMemoryBank


def _cluster_vector(cluster: CentroidCluster, *, use_prototype: bool = False) -> list[float]:
    """Return the prototype when available, otherwise preserve centroid behavior."""
    if use_prototype and cluster.prototype_embedding:
        return cluster.prototype_embedding
    return cluster.centroid

class CentroidRetriever:
    """Cosine top-1 retrieval with optional medoid prototypes.

    Configure ``use_prototype=True`` only for banks materialized with prototype
    embeddings.  Legacy schema-7 banks safely fall back to their centroid.
    """

    def __init__(self, embedder, *, min_similarity: float = .75,
                 use_prototype: bool = False) -> None:  # type: ignore[no-untyped-def]
        self.embedder = embedder
        self.min_similarity = float(min_similarity)
        self.use_prototype = bool(use_prototype)
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ValueError("retrieval_min_similarity must be in [-1,1]")

    def rank(self, query: str, bank: CentroidMemoryBank,
             *, audit: dict[str, Any] | None = None, **_: Any) -> list[CentroidMatch]:
        if not bank.clusters:
            if audit is not None:
                audit.update(reason="empty_bank", selected=False,
                             threshold=self.min_similarity)
            return []
        query_vector = encode_query(self.embedder, [query])[0]
        ranked = sorted(
            ((cosine(query_vector, _cluster_vector(cluster,
                                                   use_prototype=self.use_prototype)), cluster)
             for cluster in bank.clusters.values()),
            key=lambda item: (-item[0], item[1].cluster_id),
        )
        similarity, cluster = ranked[0]
        if audit is not None:
            audit.update(
                reason=("selected" if similarity >= self.min_similarity
                        else "below_similarity_threshold"),
                selected=similarity >= self.min_similarity,
                similarity=float(similarity),
                threshold=self.min_similarity,
                candidate_cluster_id=cluster.cluster_id,
                second_similarity=(float(ranked[1][0]) if len(ranked) > 1 else None),
                # Every cluster's score, so any other threshold is recomputable
                # offline from one run instead of needing a re-run per threshold.
                all_similarities=[[item[1].cluster_id, float(item[0])] for item in ranked],
                prototype_mode=self.use_prototype,
            )
        if similarity < self.min_similarity:
            return []
        return [CentroidMatch(
            cluster_id=cluster.cluster_id,
            opponent_strategy=cluster.opponent_strategy,
            response_strategy=cluster.response_strategy,
            score=similarity,
            metric="cosine",
        )]

class CosineTopKNLIReranker:
    """Dense top-k recall followed by a cross-encoder/NLI top-1 rerank.

    Only the final candidate is returned.  NLI sees the ToM and opponent strategy,
    never the response advice, so stylistically similar advice cannot win retrieval.
    """

    metric = "cosine_topk_nli_rerank"

    def __init__(self, embedder, nli, *, min_similarity: float = .75,
                 min_entailment: float = .0, top_k: int = 8,
                 use_prototype: bool = False) -> None:  # type: ignore[no-untyped-def]
        self.embedder = embedder
        self.nli = nli
        self.min_similarity = float(min_similarity)
        self.min_entailment = float(min_entailment)
        self.top_k = max(1, int(top_k))
        self.use_prototype = bool(use_prototype)
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ValueError("retrieval_min_similarity must be in [-1,1]")
        if not 0.0 <= self.min_entailment <= 1.0:
            raise ValueError("retrieval_min_entailment must be in [0,1]")

    @staticmethod
    def _plain_query(query: str) -> str:
        prefix = "Opponent strategy:"
        value = str(query).strip()
        return value[len(prefix):].strip() if value.startswith(prefix) else value

    def rank(self, query: str, bank: CentroidMemoryBank,
             *, audit: dict[str, Any] | None = None, **_: Any) -> list[CentroidMatch]:
        if not bank.clusters:
            if audit is not None:
                audit.update(reason="empty_bank", selected=False, metric=self.metric,
                             candidates=[])
            return []
        query_vector = encode_query(self.embedder, [query])[0]
        ranked = sorted(
            ((cosine(query_vector, _cluster_vector(cluster,
                                                   use_prototype=self.use_prototype)), cluster)
             for cluster in bank.clusters.values()),
            key=lambda item: (-item[0], item[1].cluster_id),
        )
        recalled = [(score, cluster) for score, cluster in ranked
                    if score >= self.min_similarity][:self.top_k]
        if not recalled:
            if audit is not None:
                audit.update(reason="below_similarity_threshold", selected=False,
                             metric=self.metric,
                             similarity=float(ranked[0][0]),
                             threshold=self.min_similarity,
                             all_similarities=[[c.cluster_id, float(s)] for s, c in ranked])
            return []
        premise = self._plain_query(query)
        pairs = [(premise, cluster.opponent_strategy) for _, cluster in recalled]
        score_many = getattr(self.nli, "score_many", None)
        scores = score_many(pairs) if callable(score_many) else [
            self.nli.score(left, right) for left, right in pairs
        ]
        # Entailment is primary; cosine breaks ties and keeps the reranker conservative.
        choices = sorted(
            zip(recalled, scores),
            key=lambda item: (-float(item[1].entailment), -float(item[0][0]), item[0][1].cluster_id),
        )
        (similarity, cluster), top = choices[0]
        selected = float(top.entailment) >= self.min_entailment
        candidates = [{
            "cluster_id": c.cluster_id,
            "cosine": float(s),
            "entailment": float(n.entailment),
            "neutral": float(n.neutral),
            "contradiction": float(n.contradiction),
        } for (s, c), n in zip(recalled, scores)]
        if audit is not None:
            audit.update(
                reason="selected" if selected else "below_entailment_threshold",
                selected=selected, metric=self.metric, top_k=self.top_k,
                threshold=self.min_similarity, entailment_threshold=self.min_entailment,
                similarity=float(similarity),
                candidate_cluster_id=cluster.cluster_id,
                candidates=candidates,
                all_similarities=[[c.cluster_id, float(s)] for s, c in ranked],
                prototype_mode=self.use_prototype,
            )
        if not selected:
            return []
        return [CentroidMatch(cluster_id=cluster.cluster_id,
                              opponent_strategy=cluster.opponent_strategy,
                              response_strategy=cluster.response_strategy,
                              score=float(top.entailment), metric=self.metric)]

class NLITop1Retriever:
    """Rank a small fixed bank by directional local-NLI entailment."""

    metric = "nli_entailment"

    def __init__(self, nli, *, min_entailment: float = .97) -> None:  # type: ignore[no-untyped-def]
        self.nli = nli
        self.min_entailment = float(min_entailment)
        self.min_similarity = self.min_entailment  # Unknown-turn audit compatibility.
        if not 0.0 <= self.min_entailment <= 1.0:
            raise ValueError("retrieval_min_entailment must be in [0,1]")

    @staticmethod
    def _plain_query(query: str) -> str:
        prefix = "Opponent strategy:"
        value = str(query).strip()
        return value[len(prefix):].strip() if value.startswith(prefix) else value

    def rank(self, query: str, bank: CentroidMemoryBank,
             *, audit: dict[str, Any] | None = None, **_: Any) -> list[CentroidMatch]:
        if not bank.clusters:
            if audit is not None:
                audit.update(reason="empty_bank", selected=False,
                             threshold=self.min_entailment, metric=self.metric,
                             candidates=[])
            return []
        premise = self._plain_query(query)
        ordered = sorted(bank.clusters.values(), key=lambda cluster: cluster.cluster_id)
        pairs = [(premise, cluster.opponent_strategy) for cluster in ordered]
        score_many = getattr(self.nli, "score_many", None)
        scores = score_many(pairs) if callable(score_many) else [
            self.nli.score(left, right) for left, right in pairs
        ]
        candidates = [{
            "cluster_id": cluster.cluster_id,
            "entailment": float(score.entailment),
            "neutral": float(score.neutral),
            "contradiction": float(score.contradiction),
        } for cluster, score in zip(ordered, scores)]
        ranked = sorted(
            zip(ordered, scores),
            key=lambda item: (-float(item[1].entailment), item[0].cluster_id),
        )
        cluster, top = ranked[0]
        value = float(top.entailment)
        selected = value >= self.min_entailment
        if audit is not None:
            audit.update(
                reason="selected" if selected else "below_entailment_threshold",
                selected=selected,
                threshold=self.min_entailment,
                metric=self.metric,
                score=value,
                candidate_cluster_id=cluster.cluster_id,
                candidates=candidates,
            )
        if not selected:
            return []
        return [CentroidMatch(
            cluster_id=cluster.cluster_id,
            opponent_strategy=cluster.opponent_strategy,
            response_strategy=cluster.response_strategy,
            score=value,
            metric=self.metric,
        )]

class CosineTop1NLIGatedRetriever:
    """Select cosine top-1, then require directional NLI entailment.

    NLI is deliberately a gate rather than a reranker: the semantic nearest
    centroid remains the only candidate, preventing a broad entailed policy
    from displacing the closest policy description.
    """

    metric = "cosine_top1_nli_entailment_gate"

    def __init__(self, embedder, nli, *, min_similarity: float = .55,
                 min_entailment: float = .80) -> None:  # type: ignore[no-untyped-def]
        self.embedder = embedder
        self.nli = nli
        self.min_similarity = float(min_similarity)
        self.min_entailment = float(min_entailment)
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ValueError("retrieval_min_similarity must be in [-1,1]")
        if not 0.0 <= self.min_entailment <= 1.0:
            raise ValueError("retrieval_min_entailment must be in [0,1]")

    @staticmethod
    def _plain_query(query: str) -> str:
        prefix = "Opponent strategy:"
        value = str(query).strip()
        return value[len(prefix):].strip() if value.startswith(prefix) else value

    def rank(self, query: str, bank: CentroidMemoryBank,
             *, audit: dict[str, Any] | None = None, **_: Any) -> list[CentroidMatch]:
        if not bank.clusters:
            if audit is not None:
                audit.update(reason="empty_bank", selected=False, metric=self.metric,
                             similarity_threshold=self.min_similarity,
                             entailment_threshold=self.min_entailment)
            return []
        query_vector = encode_query(self.embedder, [query])[0]
        ranked = sorted(
            ((cosine(query_vector, cluster.centroid), cluster)
             for cluster in bank.clusters.values()),
            key=lambda item: (-item[0], item[1].cluster_id),
        )
        similarity, cluster = ranked[0]
        second_similarity = ranked[1][0] if len(ranked) > 1 else None
        nli_score = None
        if similarity >= self.min_similarity:
            nli_score = self.nli.score(
                self._plain_query(query), cluster.opponent_strategy,
            )
        selected = bool(
            nli_score is not None
            and float(nli_score.entailment) >= self.min_entailment
        )
        if audit is not None:
            reason = (
                "below_similarity_threshold" if similarity < self.min_similarity
                else "selected" if selected
                else "below_entailment_threshold"
            )
            audit.update(
                reason=reason,
                selected=selected,
                metric=self.metric,
                candidate_cluster_id=cluster.cluster_id,
                similarity=float(similarity),
                second_similarity=(float(second_similarity)
                                   if second_similarity is not None else None),
                similarity_threshold=self.min_similarity,
                entailment=(float(nli_score.entailment)
                            if nli_score is not None else None),
                neutral=(float(nli_score.neutral) if nli_score is not None else None),
                contradiction=(float(nli_score.contradiction)
                               if nli_score is not None else None),
                entailment_threshold=self.min_entailment,
            )
        if not selected:
            return []
        return [CentroidMatch(
            cluster_id=cluster.cluster_id,
            opponent_strategy=cluster.opponent_strategy,
            response_strategy=cluster.response_strategy,
            score=float(nli_score.entailment),
            metric=self.metric,
        )]
