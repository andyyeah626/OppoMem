"""Extracted common responsibilities of the working memory implementation."""
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



MECHANISM = "tom_centroid_v1"

SCHEMA_VERSION = 7

MEMORY_SYSTEM = OPPONENT_MODEL_CONTRACT + "\n\n" + SYSTEM_JSON

REFLECTION_SYSTEM = RESPONSE_CONTRACT + "\n\n" + SYSTEM_JSON

_PLACEHOLDERS = {
    "english core behavioral family",
    "concise english core behavioral family",
    "concise canonical tom",
    "actual strategy description",
    "opponent strategy",
    "response strategy",
    "concise opponent strategy",
    "concise response strategy",
}

def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()

def _valid_text(value: object) -> bool:
    text = _clean_text(value)
    return (bool(text) and text.casefold() not in _PLACEHOLDERS
            and text.casefold() != UNKNOWN_OPPONENT_STRATEGY.casefold())

def _medoid_from_candidates(candidates: list[tuple[str, np.ndarray]],
                            centroid: np.ndarray) -> tuple[str | None, list[float] | None]:
    """Choose the real segment nearest the accumulated centroid."""
    if not candidates:
        return None, None
    index = max(range(len(candidates)),
                key=lambda i: cosine(candidates[i][1], centroid))
    text, vector = candidates[index]
    return text, np.asarray(vector, dtype=np.float32).tolist()

def cluster_members(cluster: CentroidCluster) -> list[tuple[str, np.ndarray]]:
    """The retained member ToM set, or the stored prototype as a one-member fallback.

    Legacy banks never store members; falling back to `prototype_strategy` /
    `prototype_embedding` lets a capped bank be seeded from a prototype bank without a
    separate migration step, and leaves a bank with neither field empty.
    """
    rows: list[tuple[str, np.ndarray]] = []
    for row in (cluster.members or []):
        if not isinstance(row, dict):
            continue
        text = _clean_text(row.get("opponent_strategy"))
        embedding = row.get("embedding")
        if text and embedding:
            rows.append((text, np.asarray(embedding, dtype=np.float32)))
    if not rows and cluster.prototype_strategy and cluster.prototype_embedding:
        rows.append((cluster.prototype_strategy,
                     np.asarray(cluster.prototype_embedding, dtype=np.float32)))
    return rows

def mean_embedding(vectors: list[np.ndarray]) -> np.ndarray:
    stack = np.stack([np.asarray(vector, dtype=np.float32) for vector in vectors])
    return stack.mean(axis=0)

def retain_closest_members(members: list[tuple[str, np.ndarray]],
                           cap: int | None) -> list[tuple[str, np.ndarray]]:
    """Keep the `cap` members closest to the mean of the set.

    "Drop the member farthest from the centroid" applied one insertion at a time is the
    same operation as keeping the `cap` nearest, and computing it on the whole set keeps
    the result independent of insertion order.
    """
    if cap is None or len(members) <= cap:
        return list(members)
    center = mean_embedding([vector for _, vector in members])
    order = sorted(range(len(members)),
                   key=lambda index: -cosine(members[index][1], center))
    keep = sorted(order[:cap])
    return [members[index] for index in keep]

def medoid_index(members: list[tuple[str, np.ndarray]], center: np.ndarray) -> int:
    return max(range(len(members)),
               key=lambda index: cosine(members[index][1], center))

def members_payload(members: list[tuple[str, np.ndarray]]) -> list[dict[str, Any]]:
    return [{"opponent_strategy": text,
             "embedding": np.asarray(vector, dtype=np.float32).tolist()}
            for text, vector in members]

def _has_target_action(value: object) -> bool:
    text = str(value or "").strip()
    if not text or text == "Game ended. Own terminal payoff only.":
        return False
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text != "[]"
    if isinstance(parsed, dict) and isinstance(parsed.get("target_events"), list):
        return bool(parsed["target_events"])
    if isinstance(parsed, list):
        return bool(parsed)
    return True
