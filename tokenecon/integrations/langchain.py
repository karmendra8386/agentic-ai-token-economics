"""Cost tracking for LangChain and LangGraph runs.

Registers a LangChain ``BaseCallbackHandler`` that prices every completed
model call with tokenecon tiers and accumulates the results as
``StepReceipt``\\ s — the same receipt types the tiered router and the other
integrations emit. No estimation, no token-counting heuristics: the numbers
come from the provider usage the framework attaches to each run.

Handlers attach wherever LangChain accepts callbacks — a model, a chain, an
agent, or a compiled LangGraph graph::

    from langchain_openai import ChatOpenAI
    from tokenecon.integrations.langchain import CostTracker

    tracker = CostTracker(tier="medium")
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[tracker])
    llm.invoke("Summarize our Q3 cloud spend.")
    receipt = tracker.receipt("Summarize our Q3 cloud spend.")
    print(receipt.pretty())
    # request    : Summarize our Q3 cloud spend.
    # difficulty : 0.162
    # outcome    : ok
    #   step 1: tier=medium in=1840 out=312 cost=$0.003088 conf=1.00 langchain call 1 (gpt-4o-mini)
    # total_cost : $0.003088

Chat models and plain LLMs both finish through ``on_llm_end`` — for chat
models the handler reads ``message.usage_metadata`` off each returned
``ChatGeneration``; for LLMs it reads ``llm_output["token_usage"]``. Either
way, each finished run becomes one receipt step. Confidence is 1.0 on every
step: with provider-reported usage there is no uncertainty in the cost figure,
unlike the router's estimated confidence signal.

Pricing uses one tier for every call by default. Pass ``model_tiers`` to price
specific models differently — keys match the model name exactly, or the part
after the last ``/``::

    tracker = CostTracker(tier="small", model_tiers={"gpt-4o": "large"})

Only successful calls are priced (a failed call has no usage to price), and
the handler never raises into the chain: a run whose usage cannot be found
records a zero-token step instead of breaking the application. ``langchain``
/ ``langchain-core`` is an optional dependency: this module imports without
it, but instantiating ``CostTracker`` raises ``ImportError`` until one of
them is installed.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from ..classifier import difficulty_score
from ..cost_model import CallRecord
from ..pricing import DEFAULT_TIERS, ModelTier
from ..receipts import StepReceipt, TaskReceipt

try:  # Optional dependency — needed to use the integration, not to import it.
    from langchain_core.callbacks import BaseCallbackHandler as _BaseHandler
except ImportError:  # pragma: no cover - legacy langchain layout
    try:
        from langchain.callbacks.base import BaseCallbackHandler as _BaseHandler
    except ImportError:  # pragma: no cover - exercised with neither installed
        _BaseHandler = None

_Base = _BaseHandler if _BaseHandler is not None else object


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


def _read_usage(container: Any) -> Optional[Tuple[int, int]]:
    """(prompt_tokens, completion_tokens) from a usage dict/object, or None.

    Accepts both naming conventions — OpenAI-style ``prompt_tokens`` /
    ``completion_tokens`` and LangChain's ``input_tokens`` / ``output_tokens``.
    """
    if container is None:
        return None
    if isinstance(container, dict):
        prompt = container.get("prompt_tokens", container.get("input_tokens"))
        completion = container.get(
            "completion_tokens", container.get("output_tokens")
        )
    else:
        prompt = getattr(container, "prompt_tokens", None)
        if prompt is None:
            prompt = getattr(container, "input_tokens", None)
        completion = getattr(container, "completion_tokens", None)
        if completion is None:
            completion = getattr(container, "output_tokens", None)
    if prompt is None and completion is None:
        return None
    return _int_or_zero(prompt), _int_or_zero(completion)


def _nested_usage(mapping: Any) -> Optional[Tuple[int, int]]:
    """Usage from a metadata dict: nested under a key, or the dict itself."""
    if not isinstance(mapping, dict):
        return None
    for key in ("token_usage", "usage"):
        found = _read_usage(mapping.get(key))
        if found is not None:
            return found
    return _read_usage(mapping)


def _extract(response: Any) -> Tuple[Optional[str], int, int, bool]:
    """Pull (model, prompt_tokens, completion_tokens, usage_found) from a result.

    Prefers ``llm_output`` (the classic LLM path); otherwise sums the
    generations (the chat-model path carries usage on each message). A run
    with several generations (``n > 1``) is billed for all of them, so the
    generations are summed into the single step for the run.
    """
    model: Optional[str] = None
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        model = llm_output.get("model_name") or llm_output.get("model")
        found = _nested_usage(llm_output)
        if found is not None:
            return model, found[0], found[1], True

    prompt_total = 0
    completion_total = 0
    found_any = False
    for generation_list in getattr(response, "generations", None) or []:
        for generation in generation_list or []:
            usage = None
            message = getattr(generation, "message", None)
            if message is not None:
                usage = _read_usage(getattr(message, "usage_metadata", None))
                metadata = getattr(message, "response_metadata", None)
                if isinstance(metadata, dict):
                    if not model:
                        model = metadata.get("model_name") or metadata.get(
                            "model"
                        )
                    if usage is None:
                        usage = _nested_usage(metadata)
            if usage is None:
                usage = _nested_usage(
                    getattr(generation, "generation_info", None)
                )
            if usage is not None:
                prompt_total += usage[0]
                completion_total += usage[1]
                found_any = True
    return model, prompt_total, completion_total, found_any


class CostTracker(_Base):  # type: ignore[valid-type,misc]
    """LangChain callback handler that prices model calls with tokenecon.

    Attach it anywhere LangChain accepts ``callbacks`` — a chat model, an
    LLM, a chain, an agent, or a compiled LangGraph graph.

    Args:
        tier: Tier name ('small', 'medium', 'large') or a ``ModelTier`` with
            your provider's actual prices, used for every model without a
            ``model_tiers`` entry. Defaults to 'medium'.
        model_tiers: Optional mapping of model name to tier (name or
            ``ModelTier``), matched exactly or on the part after the last
            ``/`` in the model string.
    """

    # A cost observer must never break the application it observes.
    raise_error = False

    def __init__(
        self,
        tier: Union[str, ModelTier] = "medium",
        model_tiers: Optional[Dict[str, Union[str, ModelTier]]] = None,
    ) -> None:
        if _BaseHandler is None:
            raise ImportError(
                "tokenecon's LangChain integration requires 'langchain-core' "
                "(or 'langchain'): pip install langchain-core"
            )
        self.tier = _resolve_tier(tier)
        self.model_tiers = {
            name: _resolve_tier(t) for name, t in (model_tiers or {}).items()
        }
        self._steps: List[StepReceipt] = []
        self._lock = threading.Lock()

    # -- LangChain callback hook ------------------------------------------

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Price a finished LLM or chat-model run (both end here)."""
        model, prompt_tokens, completion_tokens, found = _extract(response)
        tier = self._tier_for(model)
        record = CallRecord(
            tier=tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        with self._lock:
            note = f"langchain call {len(self._steps) + 1}"
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

    # -- tokenecon side ---------------------------------------------------

    def _tier_for(self, model: Optional[str]) -> ModelTier:
        if model:
            if model in self.model_tiers:
                return self.model_tiers[model]
            tail = model.rsplit("/", 1)[-1]
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
