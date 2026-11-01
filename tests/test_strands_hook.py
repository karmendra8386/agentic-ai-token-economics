"""Tests for the Strands CostReceiptHook.

The hook binds the ``strands.hooks`` event classes at import time, so these
tests install a lightweight fake ``strands`` package in ``sys.modules``
before importing the integration — the suite then runs identically with or
without strands-agents installed, and the fake events exercise the exact
same code path as real Strands hook events (the event loop attaches each
call's provider usage to the assistant message as
``message["metadata"]["usage"]`` before firing ``AfterModelCallEvent``).
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from tokenecon import ModelTier


class _FakeBeforeModelCallEvent:
    def __init__(self, agent=None):
        self.agent = agent
        self.invocation_state = {}
        self.projected_input_tokens = None
        self.cancel = False


class _FakeAfterModelCallEvent:
    def __init__(self, agent=None, stop_response=None, exception=None):
        self.agent = agent
        self.invocation_state = {}
        self.stop_response = stop_response
        self.exception = exception
        self.retry = False


class _FakeHookRegistry:
    def __init__(self):
        self.callbacks = {}

    def add_callback(self, event_type, callback):
        self.callbacks.setdefault(event_type, []).append(callback)


def _install_fake_strands():
    strands = types.ModuleType("strands")
    hooks = types.ModuleType("strands.hooks")
    hooks.BeforeModelCallEvent = _FakeBeforeModelCallEvent
    hooks.AfterModelCallEvent = _FakeAfterModelCallEvent
    hooks.HookRegistry = _FakeHookRegistry
    strands.hooks = hooks
    return {"strands": strands, "strands.hooks": hooks}


def _import_integration(module_map):
    """Import tokenecon.integrations.strands bound to the given sys.modules."""
    keys = ("strands", "strands.hooks")
    saved = {key: sys.modules.get(key) for key in keys}
    for key in keys:
        if module_map is None:
            sys.modules[key] = None  # makes `import strands...` raise
        else:
            sys.modules[key] = module_map[key]
    sys.modules.pop("tokenecon.integrations.strands", None)
    try:
        return importlib.import_module("tokenecon.integrations.strands")
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


integration = _import_integration(_install_fake_strands())
CostReceiptHook = integration.CostReceiptHook


def _fake_agent(model_id="stub-model-1"):
    """Minimal stand-in for a Strands Agent (the hook reads agent.model)."""
    model = SimpleNamespace(get_config=lambda: {"model_id": model_id})
    return SimpleNamespace(model=model)


def _registered(hook):
    registry = _FakeHookRegistry()
    hook.register_hooks(registry)
    return registry


def _fire(registry, event):
    for callback in registry.callbacks.get(type(event), []):
        callback(event)


def _model_call(registry, agent, input_tokens, output_tokens):
    """Drive one completed model call (before + after) through the hook."""
    _fire(registry, _FakeBeforeModelCallEvent(agent=agent))
    message = {
        "role": "assistant",
        "content": [{"text": "ok"}],
        "metadata": {
            "usage": {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "totalTokens": input_tokens + output_tokens,
            },
            "metrics": {"latencyMs": 1},
        },
    }
    stop = SimpleNamespace(message=message, stop_reason="end_turn")
    _fire(registry, _FakeAfterModelCallEvent(agent=agent, stop_response=stop))


def test_registers_both_model_call_events():
    registry = _registered(CostReceiptHook())
    assert len(registry.callbacks[_FakeBeforeModelCallEvent]) == 1
    assert len(registry.callbacks[_FakeAfterModelCallEvent]) == 1


def test_prices_each_completed_call():
    hook = CostReceiptHook(tier="medium")
    registry = _registered(hook)
    agent = _fake_agent()
    _model_call(registry, agent, 1840, 312)
    _model_call(registry, agent, 2210, 148)
    _model_call(registry, agent, 2400, 200)
    # medium: $1.00 / $4.00 per Mtok
    assert [s.cost for s in hook.steps] == [
        pytest.approx(0.003088),
        pytest.approx(0.002802),
        pytest.approx(0.0032),
    ]
    assert hook.total_cost == pytest.approx(0.00909)
    assert hook.steps[0].tier == "medium"
    assert hook.steps[0].note == "model call 1 (stub-model-1)"
    assert hook.steps[2].input_tokens == 2400


def test_receipt_from_hook_steps():
    hook = CostReceiptHook(tier="medium")
    registry = _registered(hook)
    _model_call(registry, _fake_agent(), 1840, 312)
    receipt = hook.receipt("Summarize our Q3 cloud spend.")
    assert receipt.outcome == "ok"
    assert receipt.total_cost == pytest.approx(0.003088)
    assert "total_cost : $0.003088" in receipt.pretty()


def test_budget_cancel_fires_exactly_at_cap():
    # Budget equals the first call's cost to the micro-dollar.
    hook = CostReceiptHook(tier="medium", budget=0.003088)
    registry = _registered(hook)
    agent = _fake_agent()
    before1 = _FakeBeforeModelCallEvent(agent=agent)
    _fire(registry, before1)
    assert before1.cancel is False  # nothing spent yet
    _model_call(registry, agent, 1840, 312)  # spends exactly the budget
    before2 = _FakeBeforeModelCallEvent(agent=agent)
    _fire(registry, before2)
    assert isinstance(before2.cancel, str)
    assert "budget" in before2.cancel
    assert hook.budget_exceeded is True
    assert hook.receipt("task").outcome == "degraded"


def test_budget_not_reached_no_cancel():
    hook = CostReceiptHook(tier="medium", budget=0.01)
    registry = _registered(hook)
    agent = _fake_agent()
    _model_call(registry, agent, 1840, 312)
    before = _FakeBeforeModelCallEvent(agent=agent)
    _fire(registry, before)
    assert before.cancel is False
    assert hook.budget_exceeded is False


def test_no_budget_never_cancels():
    hook = CostReceiptHook(tier="medium")
    registry = _registered(hook)
    agent = _fake_agent()
    _model_call(registry, agent, 10_000_000, 10_000_000)
    before = _FakeBeforeModelCallEvent(agent=agent)
    _fire(registry, before)
    assert before.cancel is False


def test_failed_call_not_priced():
    hook = CostReceiptHook()
    registry = _registered(hook)
    agent = _fake_agent()
    _fire(registry, _FakeBeforeModelCallEvent(agent=agent))
    _fire(
        registry,
        _FakeAfterModelCallEvent(agent=agent, exception=RuntimeError("boom")),
    )
    assert hook.steps == []
    assert hook.total_cost == 0.0


def test_call_without_usage_not_priced():
    # The shape of the synthetic message a cancelled call produces: no
    # metadata, hence no provider usage — nothing to price.
    hook = CostReceiptHook()
    registry = _registered(hook)
    agent = _fake_agent()
    stop = SimpleNamespace(
        message={"role": "assistant", "content": [{"text": "cancelled"}]},
        stop_reason="end_turn",
    )
    _fire(registry, _FakeAfterModelCallEvent(agent=agent, stop_response=stop))
    assert hook.steps == []


def test_receipt_before_any_events_is_empty():
    hook = CostReceiptHook()
    assert hook.steps == []
    assert hook.total_cost == 0.0
    receipt = hook.receipt("nothing ran yet")
    assert receipt.steps == []
    assert receipt.total_cost == 0.0
    assert receipt.outcome == "ok"


def test_agent_without_metrics_still_priced():
    # The hook never reads event_loop_metrics; an agent exposing only a
    # model prices fine (the no-invocations state cannot crash it).
    hook = CostReceiptHook()
    registry = _registered(hook)
    agent = SimpleNamespace(model=SimpleNamespace(get_config=lambda: {}))
    _model_call(registry, agent, 1000, 500)
    assert hook.total_cost == pytest.approx(0.001 + 0.002)
    assert hook.steps[0].note == "model call 1"


def test_model_tiers_match_exact_and_tail():
    hook = CostReceiptHook(
        tier="medium", model_tiers={"stub-model-1": "large"}
    )
    registry = _registered(hook)
    _model_call(registry, _fake_agent("stub-model-1"), 1840, 312)
    _model_call(registry, _fake_agent("acme/stub-model-1"), 1840, 312)
    _model_call(registry, _fake_agent("other-model"), 1840, 312)
    costs = [s.cost for s in hook.steps]
    assert costs[0] == pytest.approx(0.018528)  # large: $6 / $24 per Mtok
    assert costs[1] == pytest.approx(0.018528)  # matched on the tail
    assert costs[2] == pytest.approx(0.003088)  # default tier


def test_custom_model_tier_instance():
    custom = ModelTier("custom", input_per_mtok=2.0, output_per_mtok=8.0, capability=0.5)
    hook = CostReceiptHook(tier=custom)
    registry = _registered(hook)
    _model_call(registry, _fake_agent(), 500_000, 0)
    assert hook.steps[0].cost == pytest.approx(1.0)
    assert hook.steps[0].tier == "custom"


def test_on_step_callback_invoked():
    seen = []
    hook = CostReceiptHook(on_step=seen.append)
    registry = _registered(hook)
    _model_call(registry, _fake_agent(), 1840, 312)
    _model_call(registry, _fake_agent(), 2210, 148)
    assert [s.cost for s in seen] == [
        pytest.approx(0.003088),
        pytest.approx(0.002802),
    ]


def test_reset_clears_steps_and_budget_flag():
    hook = CostReceiptHook(tier="medium", budget=0.003088)
    registry = _registered(hook)
    agent = _fake_agent()
    _model_call(registry, agent, 1840, 312)
    _fire(registry, _FakeBeforeModelCallEvent(agent=agent))
    assert hook.budget_exceeded is True
    hook.reset()
    assert hook.steps == []
    assert hook.total_cost == 0.0
    assert hook.budget_exceeded is False
    assert hook.receipt("task").outcome == "ok"


def test_unknown_tier_and_negative_budget_raise():
    with pytest.raises(ValueError, match="unknown tier"):
        CostReceiptHook(tier="ultra")
    with pytest.raises(ValueError, match="budget"):
        CostReceiptHook(budget=-1.0)


def test_missing_strands_raises_import_error():
    bare = _import_integration(None)
    try:
        with pytest.raises(ImportError, match="strands-agents"):
            bare.CostReceiptHook()
        # The after-the-fact tracker in the same module is unaffected.
        assert bare.CostTracker is not None
    finally:
        # Leave the shared module bound to the fakes for other test files.
        _import_integration(_install_fake_strands())
