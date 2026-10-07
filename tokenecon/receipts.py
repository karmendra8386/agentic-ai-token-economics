"""Per-step and per-task cost receipts (paper, section 4.2).

The receipt is the unit of cost observability: tier used, tokens, cost, and
confidence per step, plus totals. Accumulated over time, receipts become the
dataset from which the routing policy improves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepReceipt:
    tier: str
    input_tokens: int
    output_tokens: int
    cost: float
    confidence: float
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskReceipt:
    request: str
    difficulty: float
    steps: List[StepReceipt] = field(default_factory=list)
    answer: Optional[str] = None
    total_cost: float = 0.0
    outcome: str = "ok"  # ok | degraded | degraded_cached

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request,
            "difficulty": self.difficulty,
            "outcome": self.outcome,
            "total_cost": round(self.total_cost, 6),
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
        }

    def pretty(self) -> str:
        lines = [
            f"request    : {self.request[:80]}",
            f"difficulty : {self.difficulty}",
            f"outcome    : {self.outcome}",
        ]
        for i, s in enumerate(self.steps, 1):
            lines.append(
                f"  step {i}: tier={s.tier} in={s.input_tokens} "
                f"out={s.output_tokens} cost=${s.cost:.6f} "
                f"conf={s.confidence:.2f} {s.note}".rstrip()
            )
        lines.append(f"total_cost : ${self.total_cost:.6f}")
        return "\n".join(lines)
