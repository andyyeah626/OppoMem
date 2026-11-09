"""Explicit, validated settings shared by the CLI and experiment pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

GAME_IDS = {"fixed_rps": "open_spiel_roshambo", "rps": "open_spiel_roshambo",
            "ipd": "axelrod_ipd", "hanabi": "adaptive_hanabi",
            "deliberation": "llm_deliberation"}


@dataclass(frozen=True)
class MemoryConfig:
    memory_mechanism: str = "tom_centroid_v1"
    memory_scope: str = "extracted-core-v1"
    protocol_id: str = "extracted-core-v1"
    tom_prompt_variant: str = "A"
    embedding_mode: str = "none"
    retrieval_backend: str = "cosine_topk_nli_rerank"
    retrieval_min_similarity: float = 0.55
    retrieval_min_entailment: float = 0.0
    retrieval_top_k: int = 8
    use_prototype: bool = True
    merge_similarity_threshold: float = 0.82
    nli_contradiction_threshold: float = 0.70
    max_centroid_size: int | None = None
    max_cluster_members: int | None = None
    fixed_policy_protocol: bool = False
    abstract_rps_memory: bool = False
    injection_min_margin: float = 0.0
    injection_tom_overlap_min: float = 0.0

    def __post_init__(self):
        if self.memory_mechanism != "tom_centroid_v1":
            raise ValueError("only schema-7 tom_centroid_v1 memory is supported")
        if self.retrieval_backend not in {"cosine", "cosine_topk_nli_rerank", "cosine_top1_nli_gate", "nli_entailment_top1"}:
            raise ValueError("unknown retrieval_backend")
        if self.embedding_mode not in {"none", "e5", "bge"}:
            raise ValueError("unknown embedding_mode")
        if self.tom_prompt_variant not in {"A", "B", "C", "NL"}:
            raise ValueError("unknown tom_prompt_variant")
        for name in ("retrieval_top_k", "max_centroid_size", "max_cluster_members"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("retrieval_min_similarity", "merge_similarity_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError(f"{name} must be in [-1, 1]")
        for name in ("retrieval_min_entailment", "nli_contradiction_threshold", "injection_tom_overlap_min"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.injection_min_margin) or self.injection_min_margin < 0:
            raise ValueError("injection_min_margin must be finite and nonnegative")


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "offline"
    model: str = "deterministic-smoke"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    reasoning_effort: str = "default"
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli: str = "cross-encoder/nli-deberta-v3-xsmall"
    cache_dir: str | None = None
    allow_download: bool = False
    read_only_response_cache: str | None = None

    def __post_init__(self):
        if self.provider not in {"offline", "openai", "deepseek"}:
            raise ValueError("provider must be offline, openai, or deepseek")
        if not self.model or not self.api_key_env:
            raise ValueError("model and api_key_env are required")
        if self.read_only_response_cache is not None and not isinstance(self.read_only_response_cache, str):
            raise ValueError("read_only_response_cache must be a path string or null")


@dataclass(frozen=True)
class ExperimentConfig:
    game: str = "fixed_rps"
    opponents: tuple[str, ...] = ("rockbot",)
    horizon: int = 6
    batches: int = 2
    episodes_per_batch: int = 2
    test_episodes: int = 2
    evaluation_workers: int = 1
    seed: int = 20261103
    history_limit: int | None = None
    injection_dropout: float = 0.5
    partner: str = "rule"
    game_dir: str | None = None
    prompt_source: str | None = None
    frozen_profile: str | None = None
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    models: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        if self.game not in GAME_IDS:
            raise ValueError(f"unsupported game: {self.game}")
        if self.partner not in {"rule", "llm"}:
            raise ValueError("partner must be rule or llm")
        if self.game == "deliberation" and not self.game_dir:
            raise ValueError("deliberation requires game_dir")
        if not self.opponents or any(not isinstance(x, str) or not x for x in self.opponents):
            raise ValueError("at least one opponent ID is required")
        if len(set(self.opponents)) != len(self.opponents):
            raise ValueError("opponent IDs must be distinct")
        for name in ("horizon", "batches", "episodes_per_batch", "test_episodes", "evaluation_workers"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.horizon < 2:
            raise ValueError("horizon must be at least two for memory construction")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.history_limit is not None and (type(self.history_limit) is not int or self.history_limit < 1):
            raise ValueError("history_limit must be a positive integer or null")
        if not math.isfinite(self.injection_dropout) or not 0 <= self.injection_dropout <= 1:
            raise ValueError("injection_dropout must be in [0, 1]")
        if self.models.provider == "offline" and self.game not in {"fixed_rps", "ipd"}:
            raise ValueError("offline smoke supports fixed_rps and the real IPD adapter")

    @property
    def game_id(self):
        return GAME_IDS[self.game]

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str | Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["memory"] = MemoryConfig(**raw.get("memory", {}))
        raw["models"] = ModelConfig(**raw.get("models", {}))
        raw["opponents"] = tuple(raw.get("opponents", ("rockbot",)))
        return cls(**raw)
