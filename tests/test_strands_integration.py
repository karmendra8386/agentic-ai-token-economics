"""Tests for the Strands cost-tracking integration.

Strands is NOT installed here by design: the integration duck-types the agent
(only ``agent.event_loop_metrics.agent_invocations[-1].cycles`` is read), so
fakes built from SimpleNamespace exercise the exact same code path.
"""

from types import SimpleNamespace

import pytest

from tokenecon import DEFAULT_TIERS, ModelTier
from tokenecon.integrations.strands import CostTracker, track


def _fake_agent(cycles):
    """Minimal stand-in for a Strands Agent with recorded invocations."""
    invocation = SimpleNamespace(
        cycles=[SimpleNamespace(usage=dict(u)) for u in cycles]
    )
    return SimpleNamespace(
        event_loop_metrics=SimpleNamespace(agent_invocations=[invocation])
    )


def test_receipt_prices_each_cycle():
    agent = _fake_agent(
        [
            {"inputTokens": 1_000_000, "outputTokens": 500_000},
            {"inputTokens": 250_000, "outputTokens": 250_000},
        ]
    )
    receipt = CostTracker(tier="small").receipt(agent, "hello")
    # small: $0.20 / $0.80 per Mtok
    assert receipt.steps[0].cost == pytest.approx(0.20 + 0.40)
    assert receipt.steps[1].cost == pytest.approx(0.05 + 0.20)
    assert receipt.total_cost == pytest.approx(0.85)
    assert receipt.steps[0].tier == "small"
    assert receipt.steps[0].input_tokens == 1_000_000
    assert receipt.steps[0].output_tokens == 500_000
    assert receipt.outcome == "ok"


def test_tier_accepts_model_tier_instance():
    custom = ModelTier("custom", input_per_mtok=2.0, output_per_mtok=8.0, capability=0.5)
    agent = _fake_agent([{"inputTokens": 500_000, "outputTokens": 0}])
    receipt = CostTracker(tier=custom).receipt(agent, "hello")
    assert receipt.steps[0].cost == pytest.approx(1.0)
    assert receipt.steps[0].tier == "custom"


def test_unknown_tier_name_raises():
    with pytest.raises(ValueError, match="unknown tier"):
        CostTracker(tier="ultra")


def test_no_invocations_raises():
    agent = SimpleNamespace(
        event_loop_metrics=SimpleNamespace(agent_invocations=[])
    )
    with pytest.raises(ValueError, match="no agent invocations"):
        CostTracker().receipt(agent, "hello")


def test_missing_usage_defaults_to_zero():
    agent = _fake_agent([{}])
    receipt = CostTracker().receipt(agent, "hello")
    assert receipt.total_cost == pytest.approx(0.0)
    assert len(receipt.steps) == 1


def test_track_convenience_constructor():
    tracker = track(_fake_agent([]), tier="large")
    assert isinstance(tracker, CostTracker)
    assert tracker.tier == DEFAULT_TIERS[2]


def test_pretty_output_contains_total():
    agent = _fake_agent([{"inputTokens": 1000, "outputTokens": 500}])
    text = CostTracker(tier="medium").receipt(agent, "hello").pretty()
    assert "total_cost" in text
    assert "event-loop cycle 1" in text
