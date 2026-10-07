"""Model tier definitions and price handling.

A tier is a priced capability band: input/output price per million tokens
plus a capability score in [0, 1]. Prices here are ILLUSTRATIVE placeholders —
substitute your provider's actual price list before making decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModelTier:
    name: str
    input_per_mtok: float   # USD per million input tokens
    output_per_mtok: float  # USD per million output tokens
    capability: float       # 0..1, higher = more capable

    def __post_init__(self) -> None:
        if not 0.0 <= self.capability <= 1.0:
            raise ValueError("capability must be in [0, 1]")
        if self.input_per_mtok < 0 or self.output_per_mtok < 0:
            raise ValueError("prices must be non-negative")

    def call_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Cost in USD of one call with the given token counts."""
        return (
            input_tokens / 1_000_000 * self.input_per_mtok
            + output_tokens / 1_000_000 * self.output_per_mtok
        )


#: Illustrative three-tier ladder (price ratio large/small = 30, as in the paper).
DEFAULT_TIERS: Tuple[ModelTier, ...] = (
    ModelTier("small", input_per_mtok=0.20, output_per_mtok=0.80, capability=0.35),
    ModelTier("medium", input_per_mtok=1.00, output_per_mtok=4.00, capability=0.65),
    ModelTier("large", input_per_mtok=6.00, output_per_mtok=24.00, capability=0.95),
)
