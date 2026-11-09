"""Deterministic model doubles for wiring checks, never for reported results."""
import numpy as np
from oppomem.models.semantics import NLIScores

KEY = "The opponent maintains a stable behavioral preference across rounds."


class OfflineEncoder:
    model_name = "offline-constant-vector"

    def encode(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class OfflineNLI:
    model_name = "offline-constant-nli"

    def score(self, premise, hypothesis):
        return NLIScores(0.99, 0.01, 0.0)

    def score_many(self, pairs):
        return [self.score(*pair) for pair in pairs]


class OfflineLLM:
    def __init__(self, config):
        self.config = config
        self.requests_made = 0
        self.events = []

    def complete_json(self, *, stage, system, prompt, max_tokens=None):
        self.requests_made += 1
        if stage == "tom":
            return {"opponent_strategy": KEY}
        if stage == "decision":
            return {"action": "Cooperate" if self.config.game == "ipd" else "Paper",
                    "response_intent": "Exercise the shared decision path"}
        if stage == "memory_windows":
            return {"windows": [{"opponent_strategy": KEY,
                                  "round_ranges": [[1, self.config.horizon - 1]]}]}
        if stage == "memory_reflection":
            return {"response_strategy": "Adapt to the observed stable preference while monitoring for changes."}
        raise ValueError(f"offline model does not implement stage {stage}")
