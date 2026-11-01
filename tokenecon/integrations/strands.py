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
    # difficulty : 0.163
    # outcome    : ok
    #   step 1: tier=medium in=1840 out=312 cost=$0.003088 conf=1.00 event-loop cycle 1
    #   step 2: tier=medium in=2210 out=148 cost=$0.002802 conf=1.00 event-loop cycle 2
    # total_cost : $0.005890

Each event-loop cycle (one model call) becomes one receipt step. Confidence is
1.0 on every step: with provider-reported usage there is no uncertainty in the
cost figure, unlike the router's estimated confidence signal.

For live control instead of after-the-fact accounting, attach
``CostReceiptHook`` to the agent: it prices each model call as it completes
and can cancel the next call once a budget is spent — see the class docstring.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Union

from ..classifier import difficulty_score
from ..cost_model import CallRecord
from ..pricing import DEFAULT_TIERS, ModelTier
from ..receipts import StepReceipt, TaskReceipt

try:  # Optional dependency — needed for CostReceiptHook, not for CostTracker.
    from strands.hooks import (
        AfterModelCallEvent as _AfterModelCallEvent,
    )
    from strands.hooks import (
        BeforeModelCallEvent as _BeforeModelCallEvent,
    )
except ImportError:  # pragma: no cover - exercised only without strands-agents
    _AfterModelCallEvent = None
    _BeforeModelCallEvent = None


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


def _message_usage(message: Any) -> Optional[Dict[str, Any]]:
    """Provider usage dict from a model-call message's metadata, if present.

    The Strands event loop attaches the call's provider-reported usage to the
    assistant message as ``message["metadata"]["usage"]`` before firing
    ``AfterModelCallEvent`` — that dict is the per-call source of truth this
    hook prices. Returns None when the message carries no usage (e.g. the
    synthetic message produced when a hook cancels a call).
    """
    if isinstance(message, dict):
        metadata = message.get("metadata")
    else:
        metadata = getattr(message, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    usage = metadata.get("usage")
    return usage if isinstance(usage, dict) else None


def _agent_model_id(agent: Any) -> Optional[str]:
    """Best-effort model id from ``agent.model``'s config, else None."""
    model = getattr(agent, "model", None)
    get_config = getattr(model, "get_config", None)
    if not callable(get_config):
        return None
    try:
        config = get_config()
    except Exception:
        return None
    if isinstance(config, dict):
        model_id = config.get("model_id")
    else:
        model_id = getattr(config, "model_id", None)
    return model_id if isinstance(model_id, str) and model_id else None


class CostReceiptHook:
    """Strands hook that prices model calls live and enforces a budget.

    Attach it like any Strands hook provider::

        from strands import Agent
        from tokenecon.integrations.strands import CostReceiptHook

        hook = CostReceiptHook(tier="medium", budget=0.004)  # 0.4¢ per run
        agent = Agent(hooks=[hook])
        agent("Summarize our Q3 cloud spend.")
        print(hook.receipt("Summarize our Q3 cloud spend.").pretty())
        # request    : Summarize our Q3 cloud spend.
        # difficulty : 0.163
        # outcome    : degraded
        #   step 1: tier=medium in=1840 out=312 cost=$0.003088 conf=1.00 model call 1 (stub-model-1)
        #   step 2: tier=medium in=2210 out=148 cost=$0.002802 conf=1.00 model call 2 (stub-model-1)
        # total_cost : $0.005890

    That output is from a verified run against a scripted stub model (id
    ``stub-model-1``, usage 1,840/312 then 2,210/148 tokens — no AWS calls);
    with a real provider each step's note carries that provider's model id.
    The agent's third model call never happened in that run: after step 2 the
    accumulated spend ($0.005890) had crossed the $0.004 budget, so the hook
    cancelled the next call via ``BeforeModelCallEvent.cancel`` and the run
    ended with the cancellation message as its final answer. A receipt whose
    run was cut short by the budget guard reports outcome ``"degraded"``.

    Pricing source: the provider-reported usage the event loop attaches to
    each completed call's message metadata — the same numbers that later land
    in ``agent.event_loop_metrics``. No estimation, no token counting. Only
    completed calls are priced: a failed call fires ``AfterModelCallEvent``
    with an exception and no usage, and a cancelled call never reaches the
    provider; neither becomes a step. If a hook retry discards a response,
    its call is still priced — the provider charged for it.

    Args:
        tier: Tier name ('small', 'medium', 'large') or a ``ModelTier`` with
            your provider's actual prices, used for every model without a
            ``model_tiers`` entry. Defaults to 'medium'.
        budget: Optional per-run budget in USD. Once accumulated priced
            spend reaches it, the next model call is cancelled before it
            happens. ``None`` (default) means receipts only, no enforcement.
        model_tiers: Optional mapping of model id to tier (name or
            ``ModelTier``), matched exactly or on the part after the last
            ``/`` in the model id.
        on_step: Optional callable invoked with each ``StepReceipt`` as it
            is priced, e.g. to print the receipt live as the run proceeds.
    """

    def __init__(
        self,
        tier: Union[str, ModelTier] = "medium",
        budget: Optional[float] = None,
        model_tiers: Optional[Dict[str, Union[str, ModelTier]]] = None,
        on_step: Optional[Callable[[StepReceipt], None]] = None,
    ) -> None:
        if _BeforeModelCallEvent is None or _AfterModelCallEvent is None:
            raise ImportError(
                "tokenecon's Strands CostReceiptHook requires the "
                "'strands-agents' package: pip install strands-agents"
            )
        if budget is not None and budget < 0:
            raise ValueError("budget must be non-negative or None")
        self.tier = _resolve_tier(tier)
        self.budget = budget
        self.model_tiers = {
            name: _resolve_tier(t) for name, t in (model_tiers or {}).items()
        }
        self.on_step = on_step
        self._steps: List[StepReceipt] = []
        self._lock = threading.Lock()
        self._pending_model_id: Optional[str] = None
        self._budget_cancelled = False

    # -- Strands HookProvider protocol ------------------------------------

    def register_hooks(self, registry: Any, **kwargs: Any) -> None:
        """Register the model-call callbacks with the agent's hook registry."""
        registry.add_callback(_BeforeModelCallEvent, self._before_model_call)
        registry.add_callback(_AfterModelCallEvent, self._after_model_call)

    # -- Hook callbacks -----------------------------------------------------

    def _before_model_call(self, event: Any) -> None:
        self._pending_model_id = _agent_model_id(getattr(event, "agent", None))
        if self.budget is None:
            return
        spent = self.total_cost
        if spent >= self.budget:
            self._budget_cancelled = True
            event.cancel = (
                f"tokenecon: budget of ${self.budget:.6f} exhausted "
                f"(${spent:.6f} spent); model call cancelled by CostReceiptHook"
            )

    def _after_model_call(self, event: Any) -> None:
        if getattr(event, "exception", None) is not None:
            return  # Failed call: no usage to price.
        stop_response = getattr(event, "stop_response", None)
        if stop_response is None:
            return
        usage = _message_usage(getattr(stop_response, "message", None))
        if usage is None:
            return  # No provider usage reported (e.g. a cancelled call).
        tier = self._tier_for(self._pending_model_id)
        record = CallRecord(
            tier=tier,
            prompt_tokens=int(usage.get("inputTokens") or 0),
            completion_tokens=int(usage.get("outputTokens") or 0),
        )
        with self._lock:
            note = f"model call {len(self._steps) + 1}"
            if self._pending_model_id:
                note += f" ({self._pending_model_id})"
            step = StepReceipt(
                tier=tier.name,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                cost=record.cost(),
                confidence=1.0,
                note=note,
            )
            self._steps.append(step)
        if self.on_step is not None:
            self.on_step(step)

    # -- tokenecon side -----------------------------------------------------

    def _tier_for(self, model_id: Optional[str]) -> ModelTier:
        if model_id:
            if model_id in self.model_tiers:
                return self.model_tiers[model_id]
            tail = model_id.rsplit("/", 1)[-1]
            if tail in self.model_tiers:
                return self.model_tiers[tail]
        return self.tier

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

    @property
    def budget_exceeded(self) -> bool:
        """True once the budget guard has cancelled a model call."""
        return self._budget_cancelled

    def receipt(
        self, request: str, answer: Optional[str] = None
    ) -> TaskReceipt:
        """Build a ``TaskReceipt`` from every call recorded so far.

        The outcome is ``"degraded"`` if the budget guard cancelled a call
        during the run, else ``"ok"``.

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
            outcome="degraded" if self._budget_cancelled else "ok",
        )

    def reset(self) -> None:
        """Forget every recorded call (e.g. between tasks)."""
        with self._lock:
            self._steps.clear()
            self._pending_model_id = None
            self._budget_cancelled = False


__all__ = ["CostTracker", "CostReceiptHook", "track"]
