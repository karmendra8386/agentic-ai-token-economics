"""Deterministic mock model for demos and tests.

Stands in for live provider calls so routing, escalation, receipts, and
guardrails can be exercised with no API keys. Confidence is a deterministic
function of tier capability vs. request difficulty — replace with a real
signal (logprobs, verifier, sampling agreement) in production.
"""

from __future__ import annotations

from typing import Tuple

from .classifier import difficulty_score
from .pricing import ModelTier


class MockModel:
    """model_fn(request, tier) -> (answer, confidence)."""

    def __call__(self, request: str, tier: ModelTier) -> Tuple[str, float]:
        difficulty = difficulty_score(request)
        confidence = tier.capability - 0.25 * difficulty + 0.15
        confidence = round(min(max(confidence, 0.0), 1.0), 3)
        return f"[{tier.name}] answer to: {request[:60]}", confidence
