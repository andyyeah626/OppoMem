"""Extracted learning responsibilities of the working memory implementation."""
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

from .common import MECHANISM, MEMORY_SYSTEM, REFLECTION_SYSTEM, _clean_text, _medoid_from_candidates, _valid_text, cluster_members, mean_embedding, medoid_index, members_payload, retain_closest_members
from .prompts import reflection_prompt, window_prompt
from .windows import normalize_windows
from .bank import CentroidMemoryBank


class CentroidMemoryLearner:
    """Immediately assign every valid window to one existing or new centroid."""

    def __init__(self, bank: CentroidMemoryBank, llm, embedder, nli, game: str,
                 scope: str, *, similarity_threshold: float = .82,
                 contradiction_threshold: float = .70,
                 max_centroid_size: int | None = None,
                 max_cluster_members: int | None = None,
                 fixed_policy: bool = False,
                 summarizer_llm=None,
                 abstract_rps: bool = False) -> None:  # type: ignore[no-untyped-def]
        self.bank, self.llm, self.embedder, self.nli = bank, llm, embedder, nli
        # The two summary stages are the only calls whose output never becomes a
        # game action, so they are the only ones a cheaper model may serve.  Both
        # live here rather than in the agent: `learn` (memory_windows) and
        # `_reflect` (memory_reflection) are the complete set.  Defaulting to
        # `llm` keeps every existing caller byte-identical.
        self.summarizer_llm = summarizer_llm if summarizer_llm is not None else llm
        self.game, self.scope = game, scope
        self.similarity_threshold = float(similarity_threshold)
        self.contradiction_threshold = float(contradiction_threshold)
        self.max_centroid_size = (
            int(max_centroid_size) if max_centroid_size is not None else None
        )
        # Member mode: keep an explicit set of at most `max_cluster_members` ToM texts
        # per cluster, evicting the member farthest from the set's centroid on overflow.
        # It replaces (never combines with) `max_centroid_size`, whose rule is the
        # opposite -- reject a full cluster and open a new one.
        self.max_cluster_members = (
            int(max_cluster_members) if max_cluster_members is not None else None
        )
        self.store_members = self.max_cluster_members is not None
        self.fixed_policy = bool(fixed_policy)
        self.abstract_rps = bool(
            abstract_rps and game in {"open_spiel_roshambo", "repeated_rps"}
        )
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("merge_similarity_threshold must be in [-1,1]")
        if not 0.0 <= self.contradiction_threshold <= 1.0:
            raise ValueError("nli_contradiction_threshold must be in [0,1]")
        if self.max_centroid_size is not None and self.max_centroid_size < 1:
            raise ValueError("max_centroid_size must be >= 1")
        if self.max_cluster_members is not None and self.max_cluster_members < 1:
            raise ValueError("max_cluster_members must be >= 1")

    @staticmethod
    def _key(text: str) -> str:
        return "Opponent strategy: " + text

    def _nli_contradictions(self, opponent: str,
                            candidates: list[tuple[float, dict]]) -> list[float]:
        left = self._key(opponent)
        pairs = []
        for _, candidate in candidates:
            right = self._key(candidate["opponent_strategy"])
            pairs.extend(((left, right), (right, left)))
        score_many = getattr(self.nli, "score_many", None)
        scores = score_many(pairs) if callable(score_many) else [
            self.nli.score(a, b) for a, b in pairs
        ]
        return [max(scores[index].contradiction, scores[index + 1].contradiction)
                for index in range(0, len(scores), 2)]

    def _cluster_id(self, opponent: str, ordinal: int) -> str:
        digest = hashlib.sha256(
            json.dumps([opponent, ordinal], ensure_ascii=False).encode()
        ).hexdigest()[:16]
        return "tom_" + digest

    def _reflect(self, trace: EpisodeTrace, window: dict, canonical: str,
                 previous_response: str | None) -> tuple[str | None, dict]:
        raw = self.summarizer_llm.complete_json(
            stage="memory_reflection", system=REFLECTION_SYSTEM,
            prompt=reflection_prompt(
                trace, window, canonical,
                previous_response_strategy=previous_response,
                abstract_actions=self.abstract_rps,
            ),
            max_tokens=300,
        )
        response = _clean_text(raw.get("response_strategy"))
        if self.abstract_rps:
            return abstract_response_strategy(response), raw
        return (response if _valid_text(response) else None), raw

    def _window_token_budget(self) -> int:
        """The whole output allowance this client is willing to spend on one call.

        `memory_windows` segments an entire online ToM chain in one shot, and the
        summarizer is a reasoning model whose hidden reasoning bills against
        `max_tokens`.  Measured on a 50-round IPD episode (2026-09-17): 9,591 reasoning
        tokens on a 20,186-character prompt, against a 4,000-token floor.  The floor
        therefore fails, and `complete_json`'s ladder needs three attempts -- 4k, 8k,
        16k -- before it reaches a budget that fits, discarding ~12,000 billed reasoning
        tokens and roughly a minute of backoff per episode.  That is per episode, and it
        is not an outage that retrying can ride out: it is a floor set below the cost of
        the call.

        Starting at the ceiling skips only the attempts that were certain to fail: the
        ladder caps at `MAX_TOKEN_BUDGET` anyway, so this spends nothing the ladder would
        not have spent on a genuine failure.  The number is read off the client rather
        than written down here so a provider whose ceiling moves takes this with it; the
        fallback is the floor `complete_json` would have applied on its own, which keeps
        a client without the attribute at its previous behaviour rather than unbounded.
        """
        ceiling = getattr(self.summarizer_llm, "MAX_TOKEN_BUDGET", None)
        if not isinstance(ceiling, int) or ceiling <= 0:
            return 600
        return ceiling

    def learn(self, trace: EpisodeTrace) -> list[CentroidCluster]:
        raw = self.summarizer_llm.complete_json(
            stage="memory_windows", system=MEMORY_SYSTEM,
            prompt=window_prompt(trace, fixed_policy=self.fixed_policy,
                                 abstract_actions=self.abstract_rps),
            max_tokens=self._window_token_budget(),
        )
        windows, normalization = normalize_windows(
            raw, trace.turns, fixed_policy=self.fixed_policy,
            tom_states=trace.tom_chain, abstract_actions=self.abstract_rps,
        )
        operations = []
        reflections = []
        changed = []
        # Working copies make every touched cluster atomic for this episode.  They
        # also let later windows match a provisional cluster created earlier in the
        # same segmentation pass.
        proposals: dict[str, dict[str, Any]] = {
            cluster_id: {
                "cluster_id": cluster_id,
                "opponent_strategy": cluster.opponent_strategy,
                "previous_response_strategy": cluster.response_strategy,
                "centroid": np.asarray(cluster.centroid, dtype=np.float32),
                "centroid_size": cluster.centroid_size,
                "base_cluster": cluster,
                "windows": [],
                "members": (cluster_members(cluster) if self.store_members else []),
                "prototype_candidates": ([(cluster.prototype_strategy,
                                           np.asarray(cluster.prototype_embedding,
                                                     dtype=np.float32))]
                                         if cluster.prototype_strategy
                                         and cluster.prototype_embedding else []),
            }
            for cluster_id, cluster in self.bank.clusters.items()
        }
        next_ordinal = len(proposals)
        for window in windows:
            opponent = window["opponent_strategy"]
            vector = encode_documents(self.embedder, [self._key(opponent)])[0]
            candidates = sorted(
                ((cosine(vector, proposal["centroid"]), proposal)
                 for proposal in proposals.values()),
                key=lambda item: item[0], reverse=True,
            )
            eligible = [item for item in candidates
                        if item[0] >= self.similarity_threshold]
            contradictions = self._nli_contradictions(opponent, eligible)
            selected: dict[str, Any] | None = None
            selected_similarity = None
            selected_contradiction = None
            nli_candidates = []
            for (similarity, proposal), contradiction in zip(eligible, contradictions):
                nli_candidates.append({
                    "cluster_id": proposal["cluster_id"],
                    "cosine": float(similarity),
                    "contradiction": float(contradiction),
                })
                if contradiction >= self.contradiction_threshold:
                    operations.append({"operation": "nli_rejected",
                                       "cluster_id": proposal["cluster_id"],
                                       "cosine": float(similarity),
                                       "contradiction": float(contradiction),
                                       "round_ranges": window["round_ranges"]})
                    continue
                if (not self.store_members and self.max_centroid_size is not None
                        and int(proposal["centroid_size"]) >= self.max_centroid_size):
                    operations.append({
                        "operation": "size_cap_rejected",
                        "cluster_id": proposal["cluster_id"],
                        "cosine": float(similarity),
                        "centroid_size": int(proposal["centroid_size"]),
                        "round_ranges": window["round_ranges"],
                    })
                    continue
                selected = proposal
                selected_similarity = float(similarity)
                selected_contradiction = float(contradiction)
                break
            if selected is None:
                cluster_id = self._cluster_id(opponent, next_ordinal)
                while cluster_id in proposals:
                    next_ordinal += 1
                    cluster_id = self._cluster_id(opponent, next_ordinal)
                next_ordinal += 1
                selected = {
                    "cluster_id": cluster_id,
                    "opponent_strategy": opponent,
                    "previous_response_strategy": None,
                    "centroid": np.asarray(vector, dtype=np.float32),
                    "centroid_size": 0,
                    "base_cluster": None,
                    "windows": [],
                    "members": [],
                    "prototype_candidates": [],
                }
                proposals[cluster_id] = selected
                assignment = "new"
            else:
                assignment = "existing"
            if self.store_members:
                # The centroid is exactly the mean of the retained members, so it is
                # recomputed instead of carried as a running mean: once the farthest
                # member is evicted the old running mean no longer describes the set.
                selected["members"].append((opponent, np.asarray(vector, dtype=np.float32)))
                selected["members"] = retain_closest_members(
                    selected["members"], self.max_cluster_members)
                selected["centroid"] = mean_embedding(
                    [member for _, member in selected["members"]])
                selected["centroid_size"] = len(selected["members"])
                medoid = medoid_index(selected["members"], selected["centroid"])
                selected["opponent_strategy"] = selected["members"][medoid][0]
                selected["prototype"] = selected["members"][medoid]
            else:
                old_size = int(selected["centroid_size"])
                selected["centroid"] = (
                    (selected["centroid"] * old_size + vector) / (old_size + 1)
                    if old_size else np.asarray(vector, dtype=np.float32)
                )
                selected["centroid_size"] = old_size + 1
                selected["prototype_candidates"].append(
                    (opponent, np.asarray(vector, dtype=np.float32)))
            selected["windows"].append(window)
            operations.append({
                "operation": "assigned",
                "assignment": assignment,
                "cluster_id": selected["cluster_id"],
                "cosine": selected_similarity,
                "contradiction": selected_contradiction,
                "round_ranges": window["round_ranges"],
                "nli_candidates": nli_candidates,
            })

        # Reflect after all assignments.  Overlap is retained across clusters and
        # deduplicated within a cluster, so each touched cluster gets one call.
        for proposal in proposals.values():
            assigned_windows = proposal["windows"]
            if not assigned_windows:
                continue
            rounds = sorted({round_index for window in assigned_windows
                             for round_index in window["rounds"]})
            combined_window = {
                "rounds": rounds,
                "round_ranges": [interval for window in assigned_windows
                                 for interval in window["round_ranges"]],
            }
            response, raw_reflection = self._reflect(
                trace, combined_window, proposal["opponent_strategy"],
                proposal["previous_response_strategy"],
            )
            reflection_audit = {
                "cluster_id": proposal["cluster_id"],
                "segment_count": len(assigned_windows),
                "round_ranges": combined_window["round_ranges"],
                "rounds": rounds,
                "canonical_opponent_strategy": proposal["opponent_strategy"],
                "previous_response_strategy": proposal["previous_response_strategy"],
                "raw_response": raw_reflection,
            }
            reflections.append(reflection_audit)
            if response is None:
                operations.append({
                    "operation": "reflection_rejected",
                    "cluster_id": proposal["cluster_id"],
                    "segment_count": len(assigned_windows),
                    "round_ranges": combined_window["round_ranges"],
                })
                continue
            base = proposal["base_cluster"]
            if self.store_members:
                prototype_strategy, prototype_vector = proposal["prototype"]
                prototype_embedding = np.asarray(
                    prototype_vector, dtype=np.float32).tolist()
            else:
                prototype_strategy, prototype_embedding = _medoid_from_candidates(
                    proposal["prototype_candidates"], proposal["centroid"]
                )
            member_rows = (members_payload(proposal["members"])
                           if self.store_members else None)
            if base is None:
                committed = CentroidCluster(
                    cluster_id=proposal["cluster_id"],
                    opponent_strategy=proposal["opponent_strategy"],
                    response_strategy=response,
                    centroid=proposal["centroid"].tolist(),
                    centroid_size=int(proposal["centroid_size"]),
                    prototype_strategy=prototype_strategy,
                    prototype_embedding=prototype_embedding,
                    members=member_rows,
                )
                self.bank.clusters[committed.cluster_id] = committed
                operation = "created"
            else:
                base.centroid = proposal["centroid"].tolist()
                base.centroid_size = int(proposal["centroid_size"])
                base.response_strategy = response
                if prototype_strategy is not None:
                    base.prototype_strategy = prototype_strategy
                    base.prototype_embedding = prototype_embedding
                if self.store_members:
                    # The canonical ToM tracks the medoid member, so the NLI gate, the
                    # stored description, and the retrieval prototype describe the same
                    # member instead of three different versions of the cluster.
                    base.opponent_strategy = proposal["opponent_strategy"]
                    base.members = member_rows
                committed = base
                operation = "merged"
            operations.append({
                "operation": operation,
                "cluster_id": committed.cluster_id,
                "segment_count": len(assigned_windows),
                "round_ranges": combined_window["round_ranges"],
            })
            changed.append(committed)
        self.bank.metadata = {
            **self.bank.metadata,
            "mechanism": MECHANISM,
            "game": self.game,
            "scope": self.scope,
            "granularity": ("episode_policy" if self.fixed_policy else "semantic_tom_window"),
            "pipeline": "tom_semantic_segments_cosine_bidirectional_nli_then_reflection",
            "merge_llm_confirmation": False,
            "centroid_size_unit": ("retained_members" if self.store_members
                                   else "semantic_segments"),
            "max_cluster_members": self.max_cluster_members,
            "fixed_policy_protocol": self.fixed_policy,
            "embedding_model": getattr(self.embedder, "model_name", type(self.embedder).__name__),
            "merge_similarity_threshold": self.similarity_threshold,
            "nli_contradiction_threshold": self.contradiction_threshold,
            "retrieval_min_similarity": self.bank.metadata.get(
                "retrieval_min_similarity", .75
            ),
        }
        if self.abstract_rps:
            for cluster in self.bank.clusters.values():
                cluster.opponent_strategy = abstract_opponent_strategy(cluster.opponent_strategy)
                cluster.response_strategy = abstract_response_strategy(cluster.response_strategy)
                if cluster.prototype_strategy:
                    cluster.prototype_strategy = abstract_opponent_strategy(cluster.prototype_strategy)
                for member in cluster.members or []:
                    if member.get("opponent_strategy"):
                        member["opponent_strategy"] = abstract_opponent_strategy(
                            member["opponent_strategy"]
                        )
        self.bank.save()
        trace.learning_audit["centroid_memory"] = {
            "raw_response": raw,
            "parsed_windows": windows,
            "normalization": normalization,
            "operations": operations,
            "reflections": reflections,
            "cluster_count": len(self.bank.clusters),
        }
        return changed
