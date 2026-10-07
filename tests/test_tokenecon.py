"""Tests for tokenecon: cost math, routing policy, guardrails."""

import pytest

from tokenecon import (
    DEFAULT_TIERS,
    CallRecord,
    MockModel,
    ModelTier,
    TieredRouter,
    difficulty_score,
    task_cost,
    tiering_ratio,
)


def test_tier_validation():
    with pytest.raises(ValueError):
        ModelTier("x", 1.0, 2.0, capability=1.5)
    with pytest.raises(ValueError):
        ModelTier("x", -1.0, 2.0, capability=0.5)


def test_call_cost_math():
    tier = ModelTier("t", input_per_mtok=1.0, output_per_mtok=4.0, capability=0.5)
    assert tier.call_cost(1_000_000, 1_000_000) == pytest.approx(5.0)
    assert tier.call_cost(500_000, 250_000) == pytest.approx(1.5)


def test_four_stream_cost():
    tier = DEFAULT_TIERS[0]
    call = CallRecord(
        tier=tier, prompt_tokens=1000, completion_tokens=200,
        tool_tokens=300, retrieval_tokens=8000,
    )
    assert call.input_tokens == 9300
    expected = 9300 / 1e6 * 0.20 + 200 / 1e6 * 0.80
    assert call.cost() == pytest.approx(expected)
    assert task_cost([call, call]) == pytest.approx(2 * expected)


def test_tiering_ratio():
    # Paper's headline numbers: alpha=0.1, r=30 -> ~13%.
    assert tiering_ratio(0.1, 30) == pytest.approx(0.13, abs=0.01)
    assert tiering_ratio(0.0, 10) == pytest.approx(0.10)
    assert tiering_ratio(1.0, 30) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        tiering_ratio(0.5, 0)


def test_difficulty_ordering():
    easy = difficulty_score("What is the capital of France?")
    hard = difficulty_score(
        "Design a fault-tolerant orchestration layer. Prove it handles partial "
        "failures. Show your reasoning step by step."
    )
    assert 0.0 <= easy <= 1.0
    assert easy < hard


def test_select_tier_cheapest_capable():
    router = TieredRouter(tiers=list(DEFAULT_TIERS))
    assert router.select_tier(0.05).name == "small"
    assert router.select_tier(0.99).name == "large"
    mid = router.select_tier(0.45)
    assert mid.name in {"small", "medium"}


def test_router_needs_two_tiers():
    with pytest.raises(ValueError):
        TieredRouter(tiers=[DEFAULT_TIERS[0]])


def test_easy_task_stays_cheap():
    router = TieredRouter(tiers=list(DEFAULT_TIERS))
    receipt = router.run("What is 2 + 2?", MockModel())
    assert receipt.outcome == "ok"
    assert receipt.steps[0].tier == "small"
    assert receipt.total_cost > 0


def test_hard_task_escalates():
    # Medium-difficulty task: selected tier can't clear the quality bar,
    # so the router cascades up exactly one tier.
    router = TieredRouter(tiers=list(DEFAULT_TIERS))
    receipt = router.run(
        "Compare Kubernetes and ECS for our batch workload and recommend "
        "one, with trade-offs.",
        MockModel(),
    )
    tiers_used = [s.tier for s in receipt.steps]
    assert tiers_used[0] == "medium"
    assert len(tiers_used) > 1  # cascade: cheap first, escalate on doubt
    assert receipt.outcome == "ok"


def test_hardest_task_goes_straight_to_large():
    router = TieredRouter(tiers=list(DEFAULT_TIERS))
    receipt = router.run(
        "Design a fault-tolerant agent orchestration layer. Prove it handles "
        "partial tool failures. Show your reasoning step by step.",
        MockModel(),
    )
    assert receipt.steps[0].tier == "large"
    assert receipt.outcome == "ok"


def test_budget_guardrail_degrades_not_fails():
    router = TieredRouter(tiers=list(DEFAULT_TIERS), budget=1e-9)
    receipt = router.run(
        "Design a fault-tolerant agent orchestration layer with proofs.",
        MockModel(),
    )
    # Near-zero budget: first run can't even afford step one and has no
    # cache -> router still completes via the escalation guardrail path.
    assert receipt.outcome in {"ok", "degraded", "degraded_cached"}
    assert receipt.answer is not None


def test_cached_answer_on_second_run():
    router = TieredRouter(tiers=list(DEFAULT_TIERS))
    req = "What is the capital of France?"
    first = router.run(req, MockModel())
    assert first.outcome == "ok"  # runs, then caches the answer
    router.budget = 1e-9  # now nothing is affordable...
    second = router.run(req, MockModel())
    assert second.outcome == "degraded_cached"  # ...so the cache serves it
    assert second.total_cost == 0.0
    assert second.answer == first.answer


def test_forecast_needs_no_model():
    router = TieredRouter(tiers=list(DEFAULT_TIERS))
    f = router.forecast("What is the capital of France?")
    assert f["tier"] == "small"
    assert f["est_cost"] > 0
    assert set(f) == {"difficulty", "tier", "input_tokens", "output_tokens", "est_cost"}


def test_receipt_serializes():
    router = TieredRouter(tiers=list(DEFAULT_TIERS))
    receipt = router.run("What is 2 + 2?", MockModel())
    d = receipt.to_dict()
    assert d["outcome"] == "ok"
    assert len(d["steps"]) >= 1
    assert "tier" in d["steps"][0]
