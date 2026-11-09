"""Reproduce the tokenecon demo.

1. Runs the shipped CLI demo (`python -m tokenecon.cli demo`): five sample
   tasks through the tiered router, per-task receipts, and the router total
   vs the all-large baseline.
2. Re-runs the same five tasks in-process, totals the real receipts, and
   prices a 'cost / correct' badge from them with tokenecon.badge.

Writes cli-demo-output.txt, badge-output.txt and cost-per-correct.svg next
to this script. The CLI demo uses the repo's deterministic MockModel (no
API keys): it exercises the routing, escalation, budget and receipt
machinery. 'Solved' means the router finished with outcome 'ok'.

    python demo/run_demo.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def main():
    p = subprocess.run([sys.executable, "-m", "tokenecon.cli", "demo"],
                       cwd=ROOT, capture_output=True, text=True)
    (HERE / "cli-demo-output.txt").write_text(p.stdout)
    print(p.stdout)

    from tokenecon import DEFAULT_TIERS, MockModel, TieredRouter
    from tokenecon.badge import cost_per_correct, endpoint_json, render_svg
    from tokenecon.cli import DEMO_TASKS

    router = TieredRouter(tiers=list(DEFAULT_TIERS), budget=0.05)
    model = MockModel()
    total, solved, lines = 0.0, 0, []
    for task in DEMO_TASKS:
        r = router.run(task, model)
        total += r.total_cost
        solved += 1 if r.outcome == "ok" else 0
        lines.append(f"{r.outcome:>8}  ${r.total_cost:.6f}  {task[:52]}")
    results = {"model": "tiered-router (mock model)",
               "tasks_solved": solved, "tasks_total": len(DEMO_TASKS),
               "total_cost_usd": round(total, 6)}
    cpc = cost_per_correct(results)
    lines.append(f"\nresults: {json.dumps(results)}")
    lines.append(f">> cost per correct = ${total:.6f} / {solved} solved "
                 f"= ${cpc:.6f} per solved task")
    lines.append(f"badge endpoint JSON: {json.dumps(endpoint_json(results))}")
    (HERE / "cost-per-correct.svg").write_text(render_svg(
        "cost / correct", endpoint_json(results)["message"],
        "brightgreen" if cpc < 0.01 else "green"))
    lines.append("wrote cost-per-correct.svg")
    out = "\n".join(lines) + "\n"
    (HERE / "badge-output.txt").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
