"""Tests for the LiteLLM cost-tracking integration.

The integration binds LiteLLM's ``CustomLogger`` at import time, so these
tests install a lightweight fake ``litellm`` package in ``sys.modules``
before importing it — the suite then runs identically with or without
litellm installed, and the fake event objects exercise the exact same code
path as real LiteLLM success events.
"""

import asyncio
import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from tokenecon import DEFAULT_TIERS, ModelTier


class _FakeCustomLogger:
    """Stand-in for litellm.integrations.custom_logger.CustomLogger."""

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        pass


def _import_integration():
    """Import tokenecon.integrations.litellm bound to the fake CustomLogger."""
    litellm = types.ModuleType("litellm")
    integrations = types.ModuleType("litellm.integrations")
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")
    custom_logger.CustomLogger = _FakeCustomLogger
    integrations.custom_logger = custom_logger
    litellm.integrations = integrations

    keys = ("litellm", "litellm.integrations", "litellm.integrations.custom_logger")
    saved = {key: sys.modules.get(key) for key in keys}
    sys.modules.update(
        {
            "litellm": litellm,
            "litellm.integrations": integrations,
            "litellm.integrations.custom_logger": custom_logger,
        }
    )
    sys.modules.pop("tokenecon.integrations.litellm", None)
    try:
        return importlib.import_module("tokenecon.integrations.litellm")
    finally:
        # The imported module keeps its reference to the fake base class;
        # sys.modules goes back to whatever it was so nothing else is affected.
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


integration = _import_integration()
CostTracker = integration.CostTracker
track = integration.track


def _success(tracker, model="gpt-4o-mini", prompt=0, completion=0, usage=None):
    """Fire a fake LiteLLM success event at the tracker."""
    if usage is None:
        usage = SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion
        )
    response = SimpleNamespace(model=model, usage=usage)
    tracker.log_success_event({"model": model}, response, None, None)


def test_tracker_is_a_custom_logger():
    assert issubclass(CostTracker, _FakeCustomLogger)
    assert isinstance(CostTracker(), _FakeCustomLogger)


def test_success_event_prices_call():
    tracker = CostTracker(tier="small")
    _success(tracker, prompt=1_000_000, completion=500_000)
    # small: $0.20 / $0.80 per Mtok
    assert len(tracker.steps) == 1
    step = tracker.steps[0]
    assert step.cost == pytest.approx(0.20 + 0.40)
    assert step.tier == "small"
    assert step.input_tokens == 1_000_000
    assert step.output_tokens == 500_000
    assert step.confidence == 1.0
    assert "gpt-4o-mini" in step.note
    assert tracker.total_cost == pytest.approx(0.60)


def test_async_success_event_prices_call():
    tracker = CostTracker(tier="medium")
    response = SimpleNamespace(
        model="gpt-4o",
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=500),
    )
    asyncio.run(
        tracker.async_log_success_event({"model": "gpt-4o"}, response, None, None)
    )
    # medium: $1.00 / $4.00 per Mtok
    assert tracker.total_cost == pytest.approx(0.001 + 0.002)


def test_dict_shaped_response_and_usage():
    tracker = CostTracker(tier="medium")
    response = {
        "model": "claude-sonnet-4-5",
        "usage": {"prompt_tokens": 2000, "completion_tokens": 1000},
    }
    tracker.log_success_event({}, response, None, None)
    assert tracker.steps[0].input_tokens == 2000
    assert tracker.steps[0].output_tokens == 1000
    assert tracker.steps[0].cost == pytest.approx(0.002 + 0.004)
    assert "claude-sonnet-4-5" in tracker.steps[0].note


def test_model_falls_back_to_kwargs():
    tracker = CostTracker()
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    )
    tracker.log_success_event({"model": "gpt-4o-mini"}, response, None, None)
    assert "gpt-4o-mini" in tracker.steps[0].note


def test_model_tiers_route_by_model():
    tracker = CostTracker(
        tier="small", model_tiers={"gpt-4o": "large", "gpt-4o-mini": "medium"}
    )
    _success(tracker, model="gpt-4o", prompt=1_000_000, completion=0)
    _success(tracker, model="openai/gpt-4o", prompt=1_000_000, completion=0)
    _success(tracker, model="gpt-4o-mini", prompt=1_000_000, completion=0)
    _success(tracker, model="llama-3", prompt=1_000_000, completion=0)
    # large $6.00 (exact + after-slash match), medium $1.00, small $0.20
    costs = [step.cost for step in tracker.steps]
    assert costs == pytest.approx([6.00, 6.00, 1.00, 0.20])
    assert [step.tier for step in tracker.steps] == [
        "large",
        "large",
        "medium",
        "small",
    ]


def test_model_tier_accepts_model_tier_instance():
    custom = ModelTier("custom", input_per_mtok=2.0, output_per_mtok=8.0, capability=0.5)
    tracker = CostTracker(tier=custom)
    _success(tracker, prompt=500_000, completion=0)
    assert tracker.steps[0].cost == pytest.approx(1.0)
    assert tracker.steps[0].tier == "custom"


def test_unknown_tier_name_raises():
    with pytest.raises(ValueError, match="unknown tier"):
        CostTracker(tier="ultra")
    with pytest.raises(ValueError, match="unknown tier"):
        CostTracker(model_tiers={"gpt-4o": "ultra"})


def test_missing_usage_records_zero_step():
    tracker = CostTracker()
    tracker.log_success_event({"model": "gpt-4o"}, SimpleNamespace(model="gpt-4o"), None, None)
    tracker.log_success_event({}, None, None, None)
    assert len(tracker.steps) == 2
    assert tracker.total_cost == pytest.approx(0.0)
    assert "no usage reported" in tracker.steps[0].note


def test_receipt_aggregates_calls():
    tracker = CostTracker(tier="medium")
    _success(tracker, prompt=1000, completion=500)
    _success(tracker, prompt=2000, completion=250)
    receipt = tracker.receipt("hello", answer="hi")
    assert len(receipt.steps) == 2
    assert receipt.total_cost == pytest.approx(0.003 + 0.003)
    assert receipt.total_cost == pytest.approx(tracker.total_cost)
    assert receipt.answer == "hi"
    assert receipt.outcome == "ok"
    text = receipt.pretty()
    assert "total_cost" in text
    assert "litellm call 1" in text


def test_reset_clears_steps():
    tracker = CostTracker()
    _success(tracker, prompt=100, completion=50)
    tracker.reset()
    assert tracker.steps == []
    assert tracker.total_cost == pytest.approx(0.0)


def test_track_convenience_constructor():
    tracker = track(tier="large")
    assert isinstance(tracker, CostTracker)
    assert tracker.tier == DEFAULT_TIERS[2]
