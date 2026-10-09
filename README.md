# tokenecon — Token-Economics for Agentic AI

[![PyPI version](https://img.shields.io/pypi/v/tokenecon.svg)](https://pypi.org/project/tokenecon/)
[![Python versions](https://img.shields.io/pypi/pyversions/tokenecon.svg)](https://pypi.org/project/tokenecon/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A dependency-free Python implementation of the token-economics cost model and
tiered model routing design from the paper
[*Token-Economics for Agentic AI: A Cost Model and Tiered Routing Reference*](https://zenodo.org/records/23195511)
(K. Pandey, Oct 2026).

**The problem it solves:** a single model call has a predictable price; a looping
agent does not. Every tool call, observation, retry, and retrieved document adds
another round-trip through a priced model, and teams usually discover the total
from the cloud bill instead of the architecture. This package gives you two things:

1. **A cost model** — decompose per-task spend into prompt, completion, tool-call,
   and retrieval tokens, priced per model tier, so you can forecast and budget
   before deployment instead of reconciling after it.
2. **A tiered router** — classify each request by difficulty, send it to the
   cheapest tier capable of handling it, escalate on low confidence, and enforce
   per-task budgets that degrade gracefully instead of failing.

## Install

```bash
pip install tokenecon
```

No dependencies. Python 3.9+.

From source: `git clone https://github.com/karmendra8386/agentic-ai-token-economics.git && cd agentic-ai-token-economics && pip install .`

## Quickstart

```python
from tokenecon import TieredRouter, DEFAULT_TIERS, MockModel

router = TieredRouter(tiers=list(DEFAULT_TIERS), budget=0.05)  # 5¢ per task
receipt = router.run("Summarize our Q3 cloud spend in two sentences.", MockModel())
print(receipt.pretty())
# request    : Summarize our Q3 cloud spend in two sentences.
# difficulty : 0.213
# outcome    : ok
#   step 1: tier=small in=210 out=120 cost=$0.000138 conf=0.70
# total_cost : $0.000138
```

Every run produces a **receipt**: tier used, tokens, cost, and confidence per step.
Receipts are the unit of cost observability — accumulate them and they become the
dataset your routing policy improves from.

## Track a real Strands agent

The router above forecasts cost. To measure a live agent, point the
`CostTracker` at any [Strands](https://github.com/strands-agents/harness-sdk)
agent — each event-loop cycle becomes a priced receipt step, using the token
counts the provider already reported (no estimation, no heuristics):

```python
from tokenecon.integrations.strands import CostTracker

tracker = CostTracker(tier="medium")  # or pass a tokenecon.ModelTier
agent("Summarize our Q3 cloud spend.")
print(tracker.receipt(agent, "Summarize our Q3 cloud spend.").pretty())
# request    : Summarize our Q3 cloud spend.
# difficulty : 0.213
# outcome    : ok
#   step 1: tier=medium in=1840 out=312 cost=$0.003088 conf=1.00 event-loop cycle 1
#   step 2: tier=medium in=2210 out=148 cost=$0.002802 conf=1.00 event-loop cycle 2
# total_cost : $0.005890
```

The integration is optional and dependency-free: it only reads the agent's
`event_loop_metrics`, so `strands-agents` is never imported by tokenecon itself.

## CLI

```bash
# Run 5 sample tasks through the router and compare against the all-large baseline
tokenecon demo

# Same, under a 1¢ per-task budget (watch guardrails kick in)
tokenecon demo --budget 0.01

# Forecast spend for a list of requests (one per line, no model calls)
tokenecon estimate requests.txt

# Bring your own tiers (JSON: {"tiers": [{"name": ..., "input_per_mtok": ...,
#   "output_per_mtok": ..., "capability": ...}]})
tokenecon demo --tiers my-tiers.json
```

## How the router works

1. **Classify** — each request gets a difficulty score (0–1) from surface signals:
   length, analytical keywords ("compare", "design", "prove"), question marks,
   multi-step phrasing. Simple and replaceable; graduate to a learned classifier
   with labeled traffic.
2. **Select** — cheapest tier whose capability clears difficulty + safety margin.
   Falls back to the strongest tier; the router never refuses work.
3. **Escalate** — every response carries a confidence signal; below the quality
   threshold, the request cascades to the next tier (try cheap, escalate on doubt).
4. **Budget** — two guardrails, both degrade instead of failing:
   - *Before spending:* if the cheapest first step exceeds the budget, serve a
     cached answer for $0.
   - *Before escalating:* if the next tier up would break the budget, keep the
     current answer and mark the run degraded.

The tiering arithmetic: with planning fraction α ≈ 0.1 and price ratio r ≈ 30
between tiers, tiered cost is roughly 13% of the all-strong baseline —
`tokenecon.tiering_ratio(0.1, 30)`.

## Honest limitations

- **Pricing is illustrative.** Default tiers are placeholders. Substitute your
  provider's actual price list before making decisions.
- **Token counting is heuristic** (~4 chars/token for English). Production use
  needs the provider's tokenizer or usage API.
- **Confidence comes from the mock model** in demos. Production needs a real
  signal: log-probabilities, a verifier model, or sampling agreement.
- **Single-request scope.** Multi-turn compounding (tool-call overhead, retrieval
  accumulation) is modeled in the cost equations but not executed by the router.
- See the paper (§6) for the full limitations discussion.

## Cite

If you use this in your work, please cite the paper:

> K. Pandey, "Token-Economics for Agentic AI: A Cost Model and Tiered Routing
> Reference," Zenodo, DOI [10.5281/zenodo.23195511](https://zenodo.org/records/23195511), Oct. 2026.

A `CITATION.cff` is included for automated citation tooling.

## License

MIT — see [LICENSE](LICENSE).
