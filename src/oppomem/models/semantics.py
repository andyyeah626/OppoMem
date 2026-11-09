from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class RetrievalEmbedder:
    """Add retrieval-specific query/document prefixes to an existing encoder.

    The base encoder remains untouched.  This makes E5/BGE-style encoders usable
    without baking a model-specific convention into the memory or the game code.
    ``mode='none'`` is deliberately byte-compatible with the existing MiniLM path.
    """

    _PREFIXES = {
        "none": ("", ""),
        "e5": ("query: ", "passage: "),
        # BGE's recommended instruction is useful for short queries; documents
        # intentionally receive no instruction.
        "bge": ("Represent this sentence for searching relevant passages: ", ""),
    }

    def __init__(self, base: Embedder, mode: str = "none") -> None:
        mode = str(mode or "none").casefold()
        if mode not in self._PREFIXES:
            raise ValueError(f"Unknown retrieval embedding mode {mode!r}")
        self.base = base
        self.mode = mode
        self.model_name = getattr(base, "model_name", type(base).__name__)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.base.encode(texts)

    def encode_query(self, texts: Sequence[str]) -> np.ndarray:
        prefix, _ = self._PREFIXES[self.mode]
        return self.base.encode([prefix + str(text) for text in texts])

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        _, prefix = self._PREFIXES[self.mode]
        return self.base.encode([prefix + str(text) for text in texts])


def encode_query(embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
    method = getattr(embedder, "encode_query", None)
    return method(texts) if callable(method) else embedder.encode(texts)


def encode_documents(embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
    method = getattr(embedder, "encode_documents", None)
    return method(texts) if callable(method) else embedder.encode(texts)


@dataclass(frozen=True, slots=True)
class NLIScores:
    entailment: float
    neutral: float
    contradiction: float


class NLIScorer(Protocol):
    def score(self, premise: str, hypothesis: str) -> NLIScores: ...


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.shape != b.shape or not a.size:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


class MiniLMEmbedder:
    """Transformer encoder; preserve mean pooling by default for existing banks."""

    def __init__(self, model_name: str, *, cache_dir: str | None = None,
                 local_files_only: bool = True, pooling: str = "mean") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be mean or cls")
        self.pooling = pooling
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=local_files_only
        ).eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        batch = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            token_embeddings = self.model(**batch).last_hidden_state
        if self.pooling == "cls":
            pooled = token_embeddings[:, 0]
        else:
            mask = batch["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)


class DebertaNLIScorer:
    """Directional NLI scores.

    Callers that need equivalence must score both ``(A, B)`` and ``(B, A)``
    explicitly.  Keeping this layer directional prevents an asymmetric relation
    from being averaged away before retrieval or clustering can inspect it.
    """

    def __init__(self, model_name: str, *, cache_dir: str | None = None, local_files_only: bool = True) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=local_files_only
        ).eval()
        self.labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}

    def _one_way(self, premise: str, hypothesis: str) -> dict[str, float]:
        torch = self._torch
        inputs = self.tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            probs = torch.softmax(self.model(**inputs).logits[0], dim=-1).cpu().tolist()
        result = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        for index, probability in enumerate(probs):
            label = self.labels.get(index, "")
            for target in result:
                if target in label:
                    result[target] = float(probability)
        if sum(result.values()) < 0.99:
            # The downloaded cross-encoder uses the conventional C/E/N order.
            result = {"contradiction": float(probs[0]), "entailment": float(probs[1]), "neutral": float(probs[2])}
        return result

    def _batch_one_way(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[dict[str, float]]:
        torch = self._torch
        inputs = self.tokenizer(
            list(premises), list(hypotheses), padding=True, truncation=True, return_tensors="pt"
        )
        with torch.inference_mode():
            rows = torch.softmax(self.model(**inputs).logits, dim=-1).cpu().tolist()
        outputs = []
        for probs in rows:
            result = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
            for index, probability in enumerate(probs):
                label = self.labels.get(index, "")
                for target in result:
                    if target in label:
                        result[target] = float(probability)
            if sum(result.values()) < 0.99:
                result = {
                    "contradiction": float(probs[0]),
                    "entailment": float(probs[1]),
                    "neutral": float(probs[2]),
                }
            outputs.append(result)
        return outputs

    def score(self, premise: str, hypothesis: str) -> NLIScores:
        return NLIScores(**self._one_way(premise, hypothesis))

    def score_many(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        if not pairs:
            return []
        premises = [pair[0] for pair in pairs]
        hypotheses = [pair[1] for pair in pairs]
        return [NLIScores(**row) for row in self._batch_one_way(premises, hypotheses)]


def weighted_centroid(old: Sequence[float], old_count: int, new: Sequence[float], new_count: int) -> list[float]:
    old_v = np.asarray(old, dtype=np.float32)
    new_v = np.asarray(new, dtype=np.float32)
    merged = (old_v * old_count + new_v * new_count) / max(1, old_count + new_count)
    norm = float(np.linalg.norm(merged))
    if math.isclose(norm, 0.0):
        return merged.tolist()
    return (merged / norm).tolist()
