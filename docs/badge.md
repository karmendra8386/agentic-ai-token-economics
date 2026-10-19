# Cost-per-correct badges

Accuracy leaderboards hide the number production teams actually live with:
what each *correct* answer costs. A tokenecon badge puts that number —
**cost per correct = total spend (USD) ÷ tasks solved** — on your README,
computed from your own run:

![cost per correct](badge-sample.svg)

<sub>Sample: Gemini 2.5 Flash on the [Cost Per Correct](https://www.kaggle.com/benchmarks/karmendra8386/cost-per-correct)
benchmark (5/5 tasks, 2026-10-11). See [About the sample](#about-the-sample).</sub>

## Generate your badge

### 1. Describe your run as JSON

Summary form (recommended for graded runs):

```json
{
  "model": "gemini-2.5-flash",
  "tasks_solved": 5,
  "tasks_total": 5,
  "total_cost_usd": 0.000061
}
```

Or point the tool at tokenecon receipts: a JSON list of `TaskReceipt`
dicts (the shape of `TaskReceipt.to_dict()`), optionally wrapped as
`{"model": "...", "receipts": [...]}`. Total cost is summed and tasks
with `outcome == "ok"` count as solved. Receipts carry no grading, so
for a benchmark where you score answers, use the summary form with your
own solved count.

Coming from a routed run, the summary is three lines:

```python
import json
from tokenecon import TieredRouter  # or your own run harness

receipts = [...]  # TaskReceipt objects from your run
summary = {
    "model": "my-agent-stack",
    "tasks_solved": sum(1 for r in receipts if r.outcome == "ok"),
    "tasks_total": len(receipts),
    "total_cost_usd": round(sum(r.total_cost for r in receipts), 6),
}
json.dump(summary, open("results.json", "w"))
```

### 2. Render

```bash
# Static SVG file (default)
python -m tokenecon.badge results.json --format svg -o docs/cost-per-correct.svg

# shields.io endpoint JSON (print, or write with -o)
python -m tokenecon.badge results.json --format json

# README embed snippet for a hosted badge
python -m tokenecon.badge results.json --format markdown --url https://example.com/cost-per-correct.svg
```

## Embed it

**Static file** — commit the SVG next to your README and reference it:

```markdown
![cost per correct](docs/cost-per-correct.svg)
```

**Dynamic** — serve the endpoint JSON from anywhere (a tiny web app, a
GitHub Action artifact host, your own endpoint) and let shields.io
render it, so the badge updates when the number does:

```bash
python -m tokenecon.badge results.json --format json -o badge.json
# badge.json: {"schemaVersion": 1, "label": "cost / correct",
#              "message": "$0.42", "color": "yellowgreen"}
```

```markdown
![cost per correct](https://img.shields.io/endpoint?url=https://example.com/badge.json)
```

## Reading the badge

The value formats the way tokenecon receipts do: two decimals once the
cost is real money, six decimals below a cent. Colors grade the number
so a reader gets the shape at a glance:

| Cost per correct | Color |
|---|---|
| < $0.01 | bright green |
| < $0.10 | green |
| < $0.50 | yellow-green |
| < $1.00 | yellow |
| < $5.00 | orange |
| ≥ $5.00 | red |
| 0 solved | grey — "no data" |

Cost per correct without an accuracy floor is misleading — the cheapest
run can post the lowest number by failing most tasks. Publish the badge
next to your pass rate, not instead of it.

## About the sample

The sample badge above (`docs/badge-sample.svg`) is generated from the
first hosted run of the
[Cost Per Correct benchmark](https://www.kaggle.com/benchmarks/karmendra8386/cost-per-correct)
(2026-10-11): Gemini 2.5 Flash passed all 5 easy-tier tasks (leaderboard
100.00). Solved/total come from that run's results record
(`hosted-results-2026-10-11.csv` in the benchmark's dataset deposit,
DOI [10.5281/zenodo.23439614](https://doi.org/10.5281/zenodo.23439614)).

Kaggle reports pass/fail only, so no dollar figure is attributed to the
Kaggle run itself. The $0.000012 is a **tokenecon estimate** of that
run's spend: the 5 hosted task prompts priced with tokenecon at Gemini
2.5 Flash list prices ($0.30 / 1M input tokens, $2.50 / 1M output
tokens), tokens estimated at ~4 characters per token on the prompts,
with each task's expected answer as its output. Treat it as the
methodology demonstration it is; for your own runs, feed the badge your
measured spend.
