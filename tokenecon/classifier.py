"""Heuristic request-difficulty scoring (paper, section 3.1).

Deliberately simple and deliberately replaceable: scores surface signals —
request length, analytical keywords, question marks, multi-step phrasing —
into a 0..1 difficulty number. Teams should start here and graduate to a
learned classifier once they have labeled traffic.
"""

from __future__ import annotations

import re

_ANALYTICAL_KEYWORDS = frozenset(
    {
        "compare", "contrast", "design", "prove", "proof", "analyze",
        "analyse", "evaluation", "evaluate", "synthesize", "architect",
        "derive", "derivation", "optimize", "trade-off", "tradeoff",
    }
)
_MULTI_STEP_PATTERNS = (
    r"\bstep by step\b",
    r"\bfirst\b.*\bthen\b",
    r"\bmulti-?step\b",
)


def difficulty_score(request: str) -> float:
    """Score request difficulty in [0, 1] from surface signals."""
    text = request.lower()
    score = 0.15  # floor: every request costs attention
    words = len(text.split())
    score += min(words / 400.0, 0.25)
    hits = sum(1 for kw in _ANALYTICAL_KEYWORDS if kw in text)
    score += min(hits * 0.12, 0.36)
    score += min(text.count("?") * 0.05, 0.10)
    if any(re.search(p, text) for p in _MULTI_STEP_PATTERNS):
        score += 0.15
    return round(min(max(score, 0.0), 1.0), 3)
