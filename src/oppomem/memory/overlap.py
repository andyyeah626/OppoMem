"""Lightweight keyword overlap for live ToM vs cluster strategy gating."""
from __future__ import annotations

import re

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "has", "have", "been",
    "after", "when", "only", "into", "does", "not", "our", "target", "self",
    "than", "will", "would", "could", "should", "about", "their", "there",
    "what", "which", "while", "where", "other", "more", "most", "some",
}


def strategy_keyword_overlap(left: str, right: str) -> float:
    """Jaccard overlap on content words (0..1), for live ToM vs cluster text."""

    def tokens(text: str) -> set[str]:
        words = re.findall(r"[a-z]{3,}", str(text).casefold())
        return {w for w in words if w not in _STOP}

    a, b = tokens(left), tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
