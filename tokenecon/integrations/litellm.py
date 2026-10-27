"""Cost tracking for LiteLLM calls.

Registers a LiteLLM ``CustomLogger`` that prices every successful call with
tokenecon tiers and accumulates the results as ``StepReceipt``\\ s — the same
receipt types the tiered router and the Strands integration emit. No
estimation, no token-counting heuristics: the numbers come from the provider
usage LiteLLM attaches to each response.

Import this module only when you route calls through LiteLLM::

    import litellm
    from tokenecon.integrations.litellm import CostTracker

    tracker = CostTracker(tier="medium")
    litellm.callbacks = [tracker]
    litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarize our Q3 cloud spend."}],
    )
    receipt = tracker.receipt("Summarize our Q3 cloud spend.")
    print(receipt.pretty())
    # request    : Summarize our Q3 cloud spend.
    # difficulty : 0.163
    # outcome    : ok
    #   step 1: tier=medium in=1840 out=312 cost=$0.003088 conf=1.00 litellm call 1 (gpt-4o-mini)
    # total_cost : $0.003088

Each successful call becomes one receipt step. Confidence is 1.0 on every
step: with provider-reported usage there is no uncertainty in the cost figure,
unlike the router's estimated confidence signal.

Pricing uses one tier for every call by default. Pass ``model_tiers`` to price
specific models differently — keys match the model name exactly, or the part
after the last ``/`` (so ``"gpt-4o"`` also covers ``"openai/gpt-4o"``)::

    tracker = CostTracker(
        tier="small",
        model_tiers={"gpt-4o": "large", "claude-opus-4-1": "large"},
    )

Only successful calls are priced (a failed call has no usage to price);
``log_failure_event`` keeps LiteLLM's default no-op. ``litellm`` is an
optional dependency: this module imports without it, but instantiating
``CostTracker`` raises ``ImportError`` until litellm is installed.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from ..classifier import difficulty_score
from ..cost_model import CallRecord
from ..pricing import DEFAULT_TIERS, ModelTier
from ..receipts import StepReceipt, TaskReceipt

try:  # Optional dependency — needed to use the integration, not to import it.
    from litellm.integrations.custom_logger import (
        CustomLogger as _CustomLogger,
    )
except ImportError:  # pragma: no cover - exercised only without litellm
    _CustomLogger = None

_Base = _CustomLogger if _CustomLogger is not None else object


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


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_count(container: Any, *names: str) -> Optional[int]:
    """First present token count in a usage object or dict, else None."""
    for name in names:
        if isinstance(container, dict):
            value = container.get(name)
        else:
            value = getattr(container, name, None)
        if value is not None:
            return _int_or_zero(value)
    return None


def _extract(
    kwargs: Dict[str, Any], response_obj: Any
) -> Tuple[Optional[str], int, int, bool]:
    """Pull (model, prompt_tokens, completion_tokens, usage_found) from an event.

    ``response_obj`` is normally a LiteLLM ``ModelResponse`` with ``.model``
    and ``.usage``; dict-shaped responses are handled too. Falls back to the
    ``model`` LiteLLM puts in the event kwargs.
    """
    if isinstance(response_obj, dict):
        model = response_obj.get("model")
        usage = response_obj.get("usage")
    else:
        model = getattr(response_obj, "model", None)
        usage = getattr(response_obj, "usage", None)
    if not model:
        model = kwargs.get("model")
    prompt = _read_count(usage, "prompt_tokens", "input_tokens")
    completion = _read_count(usage, "completion_tokens", "output_tokens")
    found = prompt is not None or completion is not None
    return model, prompt or 0, completion or 0, found


class CostTracker(_Base):  # type: ignore[valid-type,misc]
    """LiteLLM callback that prices every successful call with tokenecon.

    Register it like any LiteLLM custom logger::

        import litellm
        litellm.callbacks = [CostTracker(tier="medium")]

    Args:
        tier: Tier name ('small', 'medium', 'large') or a ``ModelTier`` with
            your provider's actual prices, used for every model without a
            ``model_tiers`` entry. Defaults to 'medium'.
        model_tiers: Optional mapping of model name to tier (name or
            ``ModelTier``), matched exactly or on the part after the last
            ``/`` in the model string.
    """

    def __init__(
        self,
        tier: Union[str, ModelTier] = "medium",
        model_tiers: Optional[Dict[str, Union[str, ModelTier]]] = None,
    ) -> None:
        if _CustomLogger is None:
            raise ImportError(
                "tokenecon's LiteLLM integration requires the 'litellm' "
                "package: pip install litellm"
            )
        self.tier = _resolve_tier(tier)
        self.model_tiers = {
            name: _resolve_tier(t) for name, t in (model_tiers or {}).items()
        }
        self._steps: List[StepReceipt] = []
        self._lock = threading.Lock()

    # -- LiteLLM CustomLogger hooks --------------------------------------

    def log_success_event(
        self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self._record(kwargs or {}, response_obj)

    async def async_log_success_event(
        self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self._record(kwargs or {}, response_obj)

    # -- tokenecon side ---------------------------------------------------

    def _tier_for(self, model: Optional[str]) -> ModelTier:
        if model:
            if model in self.model_tiers:
                return self.model_tiers[model]
            tail = model.rsplit("/", 1)[-1]
            if tail in self.model_tiers:
                return self.model_tiers[tail]
        return self.tier

    def _record(self, kwargs: Dict[str, Any], response_obj: Any) -> None:
        model, prompt_tokens, completion_tokens, found = _extract(
            kwargs, response_obj
        )
        tier = self._tier_for(model)
        record = CallRecord(
            tier=tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        with self._lock:
            note = f"litellm call {len(self._steps) + 1}"
            if model:
                note += f" ({model})"
            if not found:
                note += "; no usage reported"
            self._steps.append(
                StepReceipt(
                    tier=tier.name,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cost=record.cost(),
                    confidence=1.0,
                    note=note,
                )
            )

    @property
    def steps(self) -> List[StepReceipt]:
        """Snapshot of the steps recorded so far, in call order."""
        with self._lock:
            return list(self._steps)

    @property
    def total_cost(self) -> float:
        """Total USD cost of all calls recorded so far."""
        with self._lock:
            return sum(step.cost for step in self._steps)

    def receipt(
        self, request: str, answer: Optional[str] = None
    ) -> TaskReceipt:
        """Build a ``TaskReceipt`` from every call recorded so far.

        Args:
            request: The task text, used for the difficulty score.
            answer: Optional final answer text to attach to the receipt.
        """
        steps = self.steps
        return TaskReceipt(
            request=request,
            difficulty=difficulty_score(request),
            steps=steps,
            answer=answer,
            total_cost=sum(step.cost for step in steps),
            outcome="ok",
        )

    def reset(self) -> None:
        """Forget every recorded call (e.g. between tasks)."""
        with self._lock:
            self._steps.clear()


def track(
    tier: Union[str, ModelTier] = "medium",
    model_tiers: Optional[Dict[str, Union[str, ModelTier]]] = None,
) -> CostTracker:
    """Convenience constructor: ``track()`` returns a ``CostTracker``."""
    return CostTracker(tier=tier, model_tiers=model_tiers)


__all__ = ["CostTracker", "track"]
