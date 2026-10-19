"""Tests for the LangChain / LangGraph cost-tracking integration.

The integration binds LangChain's ``BaseCallbackHandler`` at import time, so
these tests install a lightweight fake ``langchain_core`` package in
``sys.modules`` before importing it — the suite then runs identically with
or without langchain installed, and fake ``LLMResult`` objects exercise the
exact same code path as real callback events.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from tokenecon import DEFAULT_TIERS, ModelTier


class _FakeBaseCallbackHandler:
    """Stand-in for langchain_core.callbacks.BaseCallbackHandler."""


def _import_integration():
    """Import tokenecon.integrations.langchain bound to the fake handler."""
    core = types.ModuleType("langchain_core")
    callbacks = types.ModuleType("langchain_core.callbacks")
    callbacks.BaseCallbackHandler = _FakeBaseCallbackHandler
    core.callbacks = callbacks

    keys = ("langchain_core", "langchain_core.callbacks")
    saved = {key: sys.modules.get(key) for key in keys}
    sys.modules.update(
        {"langchain_core": core, "langchain_core.callbacks": callbacks}
    )
    sys.modules.pop("tokenecon.integrations.langchain", None)
    try:
        return importlib.import_module("tokenecon.integrations.langchain")
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


def _llm_result(prompt, completion, model="gpt-4o"):
    """Fake LLMResult on the classic path: usage in llm_output."""
    return SimpleNamespace(
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
            "model_name": model,
        },
        generations=[[]],
    )


def _chat_result(prompt, completion, model="gpt-4o-mini"):
    """Fake LLMResult on the chat path: usage on each message."""
    message = SimpleNamespace(
        usage_metadata={
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
        },
        response_metadata={"model_name": model},
    )
    generation = SimpleNamespace(message=message, generation_info=None)
    return SimpleNamespace(llm_output=None, generations=[[generation]])


def test_tracker_is_a_base_callback_handler():
    assert issubclass(CostTracker, _FakeBaseCallbackHandler)
    assert isinstance(CostTracker(), _FakeBaseCallbackHandler)
    assert CostTracker.raise_error is False


def test_llm_output_path_prices_call():
    tracker = CostTracker(tier="small")
    tracker.on_llm_end(_llm_result(1_000_000, 500_000), run_id="run-1")
    # small: $0.20 / $0.80 per Mtok
    assert len(tracker.steps) == 1
    step = tracker.steps[0]
    assert step.cost == pytest.approx(0.20 + 0.40)
    assert step.tier == "small"
    assert step.input_tokens == 1_000_000
    assert step.output_tokens == 500_000
    assert step.confidence == 1.0
    assert "gpt-4o" in step.note


def test_chat_message_usage_metadata_path():
    tracker = CostTracker(tier="medium")
    tracker.on_llm_end(_chat_result(1000, 500), run_id="run-2")
    # medium: $1.00 / $4.00 per Mtok
    step = tracker.steps[0]
    assert step.input_tokens == 1000
    assert step.output_tokens == 500
    assert step.cost == pytest.approx(0.001 + 0.002)
    assert "gpt-4o-mini" in step.note


def test_multiple_generations_are_summed():
    tracker = CostTracker(tier="medium")
    result = _chat_result(100, 50)
    twin = SimpleNamespace(
        message=SimpleNamespace(
            usage_metadata={"input_tokens": 100, "output_tokens": 50},
            response_metadata={},
        ),
        generation_info=None,
    )
    result.generations[0].append(twin)
    tracker.on_llm_end(result)
    assert tracker.steps[0].input_tokens == 200
    assert tracker.steps[0].output_tokens == 100


def test_response_metadata_token_usage_fallback():
    tracker = CostTracker(tier="medium")
    message = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "token_usage": {"prompt_tokens": 300, "completion_tokens": 100},
            "model_name": "claude-sonnet-4-5",
        },
    )
    result = SimpleNamespace(
        llm_output={},
        generations=[[SimpleNamespace(message=message, generation_info=None)]],
    )
    tracker.on_llm_end(result)
    step = tracker.steps[0]
    assert step.input_tokens == 300
    assert step.output_tokens == 100
    assert "claude-sonnet-4-5" in step.note


def test_model_tiers_route_by_model():
    tracker = CostTracker(tier="small", model_tiers={"gpt-4o": "large"})
    tracker.on_llm_end(_llm_result(1_000_000, 0, model="gpt-4o"))
    tracker.on_llm_end(_llm_result(1_000_000, 0, model="llama-3"))
    assert [step.cost for step in tracker.steps] == pytest.approx([6.00, 0.20])
    assert [step.tier for step in tracker.steps] == ["large", "small"]


def test_model_tier_accepts_model_tier_instance():
    custom = ModelTier("custom", input_per_mtok=2.0, output_per_mtok=8.0, capability=0.5)
    tracker = CostTracker(tier=custom)
    tracker.on_llm_end(_llm_result(500_000, 0))
    assert tracker.steps[0].cost == pytest.approx(1.0)
    assert tracker.steps[0].tier == "custom"


def test_unknown_tier_name_raises():
    with pytest.raises(ValueError, match="unknown tier"):
        CostTracker(tier="ultra")


def test_missing_usage_records_zero_step():
    tracker = CostTracker()
    tracker.on_llm_end(SimpleNamespace(llm_output=None, generations=[]))
    tracker.on_llm_end(
        SimpleNamespace(llm_output={}, generations=[[SimpleNamespace()]])
    )
    assert len(tracker.steps) == 2
    assert tracker.total_cost == pytest.approx(0.0)
    assert "no usage reported" in tracker.steps[0].note


def test_receipt_aggregates_calls():
    tracker = CostTracker(tier="medium")
    tracker.on_llm_end(_llm_result(1000, 500))
    tracker.on_llm_end(_chat_result(2000, 250))
    receipt = tracker.receipt("hello", answer="hi")
    assert len(receipt.steps) == 2
    assert receipt.total_cost == pytest.approx(0.003 + 0.003)
    assert receipt.total_cost == pytest.approx(tracker.total_cost)
    assert receipt.answer == "hi"
    assert receipt.outcome == "ok"
    text = receipt.pretty()
    assert "total_cost" in text
    assert "langchain call 1" in text


def test_reset_clears_steps():
    tracker = CostTracker()
    tracker.on_llm_end(_llm_result(100, 50))
    tracker.reset()
    assert tracker.steps == []
    assert tracker.total_cost == pytest.approx(0.0)


def test_track_convenience_constructor():
    tracker = track(tier="large")
    assert isinstance(tracker, CostTracker)
    assert tracker.tier == DEFAULT_TIERS[2]
