"""Per-task cost decomposition across the four token streams.

Streams (paper, section 2.1): prompt tokens, completion tokens, tool-call
overhead, and retrieval tokens. For a task making N calls::

    C_task = sum_i( t_in_i * p_in_m(i) + t_out_i * p_out_m(i) )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .pricing import ModelTier


@dataclass
class CallRecord:
    """One priced model call inside a task."""

    tier: ModelTier
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_tokens: int = 0
    retrieval_tokens: int = 0

    @property
    def input_tokens(self) -> int:
        return self.prompt_tokens + self.tool_tokens + self.retrieval_tokens

    @property
    def output_tokens(self) -> int:
        return self.completion_tokens

    def cost(self) -> float:
        return self.tier.call_cost(self.input_tokens, self.output_tokens)


def task_cost(calls: Iterable[CallRecord]) -> float:
    """Total USD cost of a task from its per-call records."""
    return sum(call.cost() for call in calls)


def tiering_ratio(planning_fraction: float, price_ratio: float) -> float:
    """C_tiered / C_all_strong ≈ α + (1 − α) / r (paper, section 2.3).

    With planning fraction α ≈ 0.1 and price ratio r ≈ 30, tiered cost is
    roughly 13% of the all-strong baseline.
    """
    if price_ratio <= 0:
        raise ValueError("price_ratio must be positive")
    if not 0.0 <= planning_fraction <= 1.0:
        raise ValueError("planning_fraction must be in [0, 1]")
    return planning_fraction + (1.0 - planning_fraction) / price_ratio
