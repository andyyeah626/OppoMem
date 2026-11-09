from __future__ import annotations
from typing import Any, Protocol

class JsonLLM(Protocol):
    def complete_json(self, *, stage: str, system: str, prompt: str, max_tokens: int | None = None) -> dict[str, Any]: ...
