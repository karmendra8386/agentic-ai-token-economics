"""Command-line interface: demo the router, forecast spend, compare baselines."""

from __future__ import annotations

import argparse
import json
import sys

from .mock import MockModel
from .pricing import DEFAULT_TIERS, ModelTier
from .router import TieredRouter

DEMO_TASKS = [
    "What is the capital of France?",
    "Summarize these three paragraphs about our Q3 cloud spend in two sentences.",
    "Compare Kubernetes and ECS for our batch workload and recommend one, with trade-offs.",
    "Debug this multi-step data pipeline failure. First reproduce the error, then trace it "
    "through three services, then propose a fix.",
    "Design a fault-tolerant agent orchestration layer. Prove it handles partial tool "
    "failures. Show your reasoning step by step.",
]


def _tiers_from_json(path: str):
    with open(path) as fh:
        data = json.load(fh)
    return [ModelTier(**t) for t in data["tiers"]]


def cmd_demo(args: argparse.Namespace) -> int:
    tiers = _tiers_from_json(args.tiers) if args.tiers else list(DEFAULT_TIERS)
    router = TieredRouter(tiers=tiers, budget=args.budget)
    model = MockModel()
    total = 0.0
    for task in DEMO_TASKS:
        receipt = router.run(task, model)
        total += receipt.total_cost
        print(receipt.pretty())
        print("-" * 60)
    # Baseline: every task served by the strongest tier, one step each.
    strongest = max(tiers, key=lambda t: t.capability)
    baseline = sum(
        strongest.call_cost(len(t) // 4 + 200, 340) for t in DEMO_TASKS
    )
    print(f"router total   : ${total:.6f}")
    print(f"all-large total: ${baseline:.6f}")
    print(f"ratio          : {total / baseline:.2%} of baseline")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    tiers = _tiers_from_json(args.tiers) if args.tiers else list(DEFAULT_TIERS)
    router = TieredRouter(tiers=tiers)
    with open(args.requests) as fh:
        requests = [line.strip() for line in fh if line.strip()]
    total = 0.0
    print(f"{'difficulty':>10} {'tier':>8} {'est_cost':>12}  request")
    for req in requests:
        f = router.forecast(req)
        total += f["est_cost"]
        print(f"{f['difficulty']:>10.3f} {f['tier']:>8} ${f['est_cost']:>10.6f}  {req[:60]}")
    print(f"\nforecast total for {len(requests)} requests: ${total:.6f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tokenecon",
        description="Token-economics cost model and tiered routing for agentic AI.",
    )
    p.add_argument("--tiers", help="JSON file with custom tier definitions")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run 5 sample tasks through the router")
    d.add_argument("--budget", type=float, default=None, help="per-task budget in USD")
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("estimate", help="forecast cost for a list of requests")
    e.add_argument("requests", help="text file, one request per line")
    e.set_defaults(func=cmd_estimate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
