"""Cost tracking for Strands Agents SDK runs.

Reads the token usage the Strands agent already records in
``agent.event_loop_metrics`` and prices it with tokenecon tiers, producing a
``TaskReceipt`` — the same receipt type the tiered router emits. No estimation,
no token-counting heuristics: the numbers come from the provider.

This module duck-types the agent (it only reads
``agent.event_loop_metrics.agent_invocations[-1].cycles``), so ``strands-agents``
is never imported and never needs to be installed to use tokenecon itself.
Import this module only when you have a Strands agent to track::

    from tokenecon.integrations.strands import CostTracker

    tracker = CostTracker(tier="medium")  # or pass a tokenecon.ModelTier
    agent("Summarize our Q3 cloud spend.")
    receipt = tracker.receipt(agent, "Summarize our Q3 cloud spend.")
    print(receipt.pretty())
    # request    : Summarize our Q3 cloud spend.
    # difficulty : 0.213
    # outcome    : ok
    #   step 1: tier=medium in=1840 out=312 cost=$0.003088 conf=1.00 event-loop cycle 1
    #   step 2: tier=medium in=2210 out=148 cost=$0.002802 conf=1.00 event-loop cycle 2
    # total_cost : $0.005890

Each event-loop cycle (one model call) becomes one receipt step. Confidence is
1.0 on every step: with provider-reported usage there is no uncertainty in the
cost figure, unlike the router's estimated confidence signal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ..classifier import difficulty_score
from ..cost_model import CallRecord
from ..pricing import DEFAULT_TIERS, ModelTier
from ..receipts import StepReceipt, TaskReceipt


def _resolve_tier(tier: Union[str, ModelTier]) -> ModelTier:
    """Accept a tier name ('small' | 'medium' | 'large') or a ModelTier."""
    if isinstance(tier, ModelTier):
        return tier
    for candidate in DEFAULT_TIERS:
        if candidate.name == tier:
            return candidate
    raise ValueError(
        f"unknown tier {tier!r}; choose from "
        f"{[t.name for t in DEFAULT_TIERS]} or pass a tokenecon.ModelTier"
    )


def _latest_cycles(agent: Any) -> List[Any]:
    """Event-loop cycles of the agent's most recent invocation."""
    metrics = getattr(agent, "event_loop_metrics", None)
    invocations = getattr(metrics, "agent_invocations", None)
    if not invocations:
        raise ValueError(
            "no agent invocations recorded yet — call agent(...) first, "
            "then ask the tracker for a receipt"
        )
    return list(invocations[-1].cycles)


class CostTracker:
    """Prices Strands agent runs with tokenecon tiers.

    Args:
        tier: Tier name ('small', 'medium', 'large') or a ``ModelTier`` with
            your provider's actual prices. Defaults to 'medium'.
    """

    def __init__(self, tier: Union[str, ModelTier] = "medium") -> None:
        self.tier = _resolve_tier(tier)

    def receipt(
        self, agent: Any, request: str, answer: Optional[str] = None
    ) -> TaskReceipt:
        """Build a ``TaskReceipt`` from the agent's latest invocation.

        Args:
            agent: A Strands ``Agent`` (or any object exposing
                ``event_loop_metrics.agent_invocations[-1].cycles`` with
                per-cycle ``usage`` dicts carrying ``inputTokens`` /
                ``outputTokens``).
            request: The task text, used for the difficulty score.
            answer: Optional final answer text to attach to the receipt.

        Raises:
            ValueError: If the agent has no recorded invocations yet.
        """
        steps: List[StepReceipt] = []
        total_cost = 0.0
        for i, cycle in enumerate(_latest_cycles(agent), 1):
            usage: Dict[str, Any] = getattr(cycle, "usage", None) or {}
            record = CallRecord(
                tier=self.tier,
                prompt_tokens=int(usage.get("inputTokens") or 0),
                completion_tokens=int(usage.get("outputTokens") or 0),
            )
            cost = record.cost()
            total_cost += cost
            steps.append(
                StepReceipt(
                    tier=self.tier.name,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cost=cost,
                    confidence=1.0,
                    note=f"event-loop cycle {i}",
                )
            )
        return TaskReceipt(
            request=request,
            difficulty=difficulty_score(request),
            steps=steps,
            answer=answer,
            total_cost=total_cost,
            outcome="ok",
        )


def track(agent: Any, tier: Union[str, ModelTier] = "medium") -> CostTracker:
    """Convenience constructor: ``track(agent)`` returns a ``CostTracker``."""
    return CostTracker(tier=tier)


__all__ = ["CostTracker", "track"]
