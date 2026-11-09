"""Extracted bank responsibilities of the working memory implementation."""
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

from .common import MECHANISM, SCHEMA_VERSION


class CentroidMemoryBank:
    schema_version = SCHEMA_VERSION

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.clusters: dict[str, CentroidCluster] = {}
        self.metadata: dict[str, Any] = {}

    def load(self) -> "CentroidMemoryBank":
        if not self.path.exists():
            return self
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != self.schema_version:
            raise ValueError(
                f"Memory bank uses schema {raw.get('schema_version')}; "
                f"{MECHANISM} requires a fresh schema-{self.schema_version} bank"
            )
        metadata = raw.get("metadata") or {}
        if metadata.get("mechanism") not in (MECHANISM, "situation_centroid_v1"):
            raise ValueError("Memory bank belongs to a different memory mechanism")
        if (metadata.get("mechanism") == "situation_centroid_v1" and
                metadata.get("retrieval_prefix") != "Observed situation: "):
            raise ValueError("Situation memory is missing its retrieval prefix")
        self.metadata = metadata
        self.clusters = {
            row["cluster_id"]: CentroidCluster.from_dict(row)
            for row in raw.get("clusters", [])
        }
        return self

    def save(self) -> None:
        if self.metadata.get("abstract_rps_memory"):
            # Validate all advice before changing any cluster or publishing the bank.
            for cluster in self.clusters.values():
                abstract_response_strategy(cluster.response_strategy)
            for cluster in self.clusters.values():
                sanitize_cluster_texts(cluster)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "metadata": self.metadata,
            "clusters": [cluster.to_dict() for cluster in self.clusters.values()],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
