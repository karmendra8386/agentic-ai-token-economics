"""tokenecon: token-economics cost model and tiered routing for agentic AI."""

from .classifier import difficulty_score
from .cost_model import CallRecord, task_cost, tiering_ratio
from .mock import MockModel
from .pricing import DEFAULT_TIERS, ModelTier
from .receipts import StepReceipt, TaskReceipt
from .router import TieredRouter

__version__ = "0.1.0"
__all__ = [
    "ModelTier",
    "DEFAULT_TIERS",
    "CallRecord",
    "task_cost",
    "tiering_ratio",
    "difficulty_score",
    "TieredRouter",
    "StepReceipt",
    "TaskReceipt",
    "MockModel",
]
