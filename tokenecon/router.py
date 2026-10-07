"""Tiered routing: classify, select, escalate, budget, receipt (paper, section 3).

Design summary
--------------
1. Score the request's difficulty (0..1).
2. Select the cheapest tier whose capability clears difficulty + safety margin;
   fall back to the strongest tier — the router never refuses work.
3. Run the model; if confidence is below the quality threshold, escalate to
   the next tier (cascade). Confidence may come from logprobs, a verifier, or
   sampling agreement — the router only needs a number and a threshold.
4. Guardrails (budgets degrade, never fail):
   - Before spending: if even the cheapest first step exceeds the remaining
     budget, serve a cached answer for $0 and mark the run degraded.
   - Before escalating: if the next tier up would break the budget, keep the
     current tier's answer and mark the run degraded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .classifier import difficulty_score
from .pricing import ModelTier
from .receipts import StepReceipt, TaskReceipt

#: model_fn(request, tier) -> (answer, confidence in [0, 1])
ModelFn = Callable[[str, ModelTier], Tuple[str, float]]

_SYSTEM_PROMPT_TOKENS = 200  # rough allowance for system instructions


@dataclass
class TieredRouter:
    tiers: List[ModelTier]
    safety_margin: float = 0.10
    quality_threshold: float = 0.70
    budget: Optional[float] = None  # USD ceiling per task; None = unbounded

    def __post_init__(self) -> None:
        if len(self.tiers) < 2:
            raise ValueError("need at least two tiers to route between")
        # Ascending by price; capability is expected to rise with price.
        self.tiers = sorted(self.tiers, key=lambda t: (t.input_per_mtok, t.capability))
        self._cache: Dict[str, str] = {}

    # -- policy ---------------------------------------------------------

    def select_tier(self, difficulty: float) -> ModelTier:
        """Cheapest tier whose capability clears difficulty + margin."""
        bar = min(difficulty + self.safety_margin, 1.0)
        for tier in self.tiers:
            if tier.capability >= bar:
                return tier
        return self.tiers[-1]

    def _next_tier(self, tier: ModelTier) -> Optional[ModelTier]:
        idx = self.tiers.index(tier)
        return self.tiers[idx + 1] if idx + 1 < len(self.tiers) else None

    # -- token estimation (heuristic; ~4 chars/token for English) --------

    def _estimate_io(self, request: str, tier: ModelTier, step: int) -> Tuple[int, int]:
        base_in = len(request) // 4 + _SYSTEM_PROMPT_TOKENS
        # Context accumulates across steps: each prior step appends its
        # output plus retrieval-ish overhead (paper, section 2.1).
        input_tokens = base_in + step * 700
        output_tokens = {"small": 120, "medium": 220}.get(tier.name, 340)
        return input_tokens, output_tokens

    # -- run ------------------------------------------------------------

    def run(self, request: str, model_fn: ModelFn) -> TaskReceipt:
        difficulty = difficulty_score(request)
        receipt = TaskReceipt(request=request, difficulty=difficulty)
        spent = 0.0

        # Guardrail 1 — before spending: cheapest first step over budget
        # and a cached answer exists -> serve it for $0, degraded.
        cheapest = self.tiers[0]
        in_tok, out_tok = self._estimate_io(request, cheapest, 0)
        if (
            self.budget is not None
            and cheapest.call_cost(in_tok, out_tok) > self.budget
            and request in self._cache
        ):
            receipt.answer = self._cache[request]
            receipt.outcome = "degraded_cached"
            receipt.steps.append(
                StepReceipt(
                    tier="cache", input_tokens=0, output_tokens=0,
                    cost=0.0, confidence=1.0,
                    note="budget exhausted; served cached answer",
                )
            )
            return receipt

        tier = self.select_tier(difficulty)
        answer: Optional[str] = None
        step = 0
        while True:
            in_tok, out_tok = self._estimate_io(request, tier, step)
            step_cost = tier.call_cost(in_tok, out_tok)
            answer, confidence = model_fn(request, tier)
            spent += step_cost
            receipt.steps.append(
                StepReceipt(
                    tier=tier.name, input_tokens=in_tok,
                    output_tokens=out_tok, cost=step_cost,
                    confidence=round(confidence, 3),
                )
            )
            if confidence >= self.quality_threshold:
                break
            nxt = self._next_tier(tier)
            if nxt is None:
                break
            # Guardrail 2 — before escalating: next tier would break the
            # budget -> keep this answer, mark degraded.
            n_in, n_out = self._estimate_io(request, nxt, step + 1)
            if self.budget is not None and spent + nxt.call_cost(n_in, n_out) > self.budget:
                receipt.outcome = "degraded"
                receipt.steps[-1].note = "escalation skipped: budget"
                break
            tier, step = nxt, step + 1

        receipt.answer = answer
        receipt.total_cost = spent
        if receipt.outcome == "ok":
            self._cache[request] = answer or ""
        return receipt

    # -- forecasting (no model calls) ------------------------------------

    def forecast(self, request: str) -> Dict[str, float]:
        """Predicted tier and cost for a request without running any model."""
        difficulty = difficulty_score(request)
        tier = self.select_tier(difficulty)
        in_tok, out_tok = self._estimate_io(request, tier, 0)
        return {
            "difficulty": difficulty,
            "tier": tier.name,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "est_cost": tier.call_cost(in_tok, out_tok),
        }
