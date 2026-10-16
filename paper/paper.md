---
title: "tokenecon: Token-economics cost accounting and tiered routing for LLM agents"
tags:
  - python
  - llm
  - ai-agents
  - cost-accounting
  - model-routing
authors:
  - name: "Karmendra Pandey"
    orcid: "0009-0000-3461-9889"
    affiliation: 1
affiliations:
  - name: "Independent researcher"
    index: 1
date: 16 October 2026
bibliography: paper.bib
---

# Summary

Software agents built on large language models (LLMs) do not make one model
call per task. They loop: they plan, call tools, read the results, retry, and
retrieve documents, and every step is another paid round-trip through a model.
The price of a single call is easy to look up, but the cost of a whole task is
usually discovered afterwards, from the cloud bill. `tokenecon` is a small,
dependency-free Python library that makes the cost of an agent task visible
*before* and *while* it runs. It splits a task's spending into four token
streams — prompt tokens, completion tokens, tool-call overhead, and retrieval
tokens — and prices each stream against configurable model tiers, so a
developer can forecast what a task will cost and set a per-task budget in
advance. It also provides a tiered router that sends each request to the
cheapest model tier likely to handle it, escalates when confidence is low, and
enforces budgets by degrading gracefully rather than failing. Every run,
whether forecast through the router or measured on a live agent, produces a
per-step "receipt" — tier used, tokens, cost, and confidence — that serves as
the unit of cost observability. A `CostTracker` integration additionally
measures agents built with the Strands Agents framework using the token counts
the provider already reports, with no estimation involved.

# Statement of need

Practitioners deploying LLM agents currently lack tooling for the most basic
economic questions about their systems: What will this task cost? Which parts
of the trajectory drive the spend? What happens to the bill if traffic grows
tenfold, or if a cheaper model tier handles the easy steps? Per-call price
lists do not answer these questions, because agent cost is a property of the
whole multi-step trajectory, not of any single call. In practice, teams
reconcile spend after the fact from invoices, and budgets, where they exist at
all, are enforced by killing runs rather than by design.

`tokenecon` addresses this gap at two points in the workflow. At design time,
its cost model lets developers decompose and forecast per-task spend across
the four token streams, priced per tier, so cost becomes an architectural
input instead of an operational surprise. At run time, the tiered router
enforces per-task budgets with guardrails that degrade service instead of
refusing it, and the receipt mechanism makes every step's spend attributable
after the fact. The library is the reference implementation of the cost model
presented in the companion paper [@pandey_tecon_2026], and the software itself
is archived and citable [@pandey_tokenecon_2026]. It is aimed at researchers
and engineers building or evaluating agentic systems who need cost to be a
first-class, measurable quantity in their experiments and deployments.

# State of the field

Cost-aware model selection is an active area. FrugalGPT [@chen_frugalgpt_2023]
showed that cascading LLM calls — trying a cheaper model first and escalating
only when necessary — can substantially reduce cost while maintaining or
improving quality. RouteLLM [@ong_routellm_2024] learned routers from human
preference data that assign queries to a strong or a weak model, cutting cost
while largely preserving response quality. These systems focus on the
per-query routing decision: which model should answer *this* request.

`tokenecon` occupies a different niche and is not a competing router. Its
bundled router is deliberately simple — a heuristic difficulty classifier with
a documented upgrade path to a learned one — because the library's
contribution is the accounting layer around routing rather than the routing
policy itself: per-task cost decomposition over multi-step agent
trajectories, budgets that bind to tasks rather than to individual calls, and
receipts that make spend auditable step by step. Systems such as FrugalGPT and
RouteLLM assume a notion of per-query cost; `tokenecon` supplies the
bookkeeping that makes per-*task* cost well defined for looping agents, and
its router exists to exercise that bookkeeping end to end.

# Software design

`tokenecon` is a dependency-free Python package (Python 3.9+) installable
from PyPI (`pip install tokenecon`). Its modules mirror the stages of the
cost model:

- **Cost model** (`cost_model`) decomposes per-task spend into prompt,
  completion, tool-call, and retrieval token streams, priced per model tier,
  and provides the tiering-ratio arithmetic used to compare tiered execution
  against an all-strong-model baseline.
- **Pricing** (`pricing`) defines model tiers — a name, per-million-token
  input and output prices, and a capability level — with placeholder defaults
  that users replace with their provider's actual price list.
- **Classifier** (`classifier`) scores request difficulty (0–1) from surface
  signals such as length, analytical keywords, and multi-step phrasing. It is
  intentionally simple and replaceable, so users can substitute a learned
  classifier trained on their own traffic.
- **Router** (`router`) implements the `TieredRouter`: classify each request,
  select the cheapest tier whose capability clears the difficulty plus a
  safety margin, execute, and escalate to the next tier when the response
  confidence falls below a threshold. Two budget guardrails degrade instead of
  failing: if the cheapest first step would exceed the remaining budget, a
  cached answer is served at zero model cost; if escalation would break the
  budget, the current answer is kept and the run is marked degraded.
- **Receipts** (`receipts`) record every step — tier, input and output
  tokens, cost, confidence — and render a human-readable summary. Receipts
  accumulate into the dataset from which routing policies can be evaluated and
  improved.
- **Command-line interface** (`cli`) offers a `demo` command that runs sample
  tasks through the router against an all-large baseline, and an `estimate`
  command that forecasts spend for a file of requests without making any
  model calls. Custom tiers can be supplied as JSON.
- **Strands Agents integration** (`integrations.strands`) provides a
  `CostTracker` that converts each event-loop cycle of a running Strands agent
  into a priced receipt step using the provider-reported token counts. The
  integration only reads the agent's `event_loop_metrics`; `tokenecon` never
  imports the framework, so the core package stays dependency-free.

Two design choices are worth noting. First, forecasting and measurement share
one receipt format, so a task's predicted and actual spend are directly
comparable. Second, the limitations are documented rather than hidden: default
prices are illustrative, token counting is heuristic where provider counts are
unavailable, and the confidence signal is expected to come from the caller's
model in production use.

# Research impact statement

`tokenecon` was released in October 2026 (version 0.2.0 on PyPI) and is at an
early stage. Its immediate significance is that it makes the cost model of the
companion paper executable: researchers studying the economics of agentic
systems can run, inspect, and cite a concrete implementation rather than
re-deriving the decomposition from the text. The release is archived on Zenodo
with a persistent DOI [@pandey_tokenecon_2026], and the Strands Agents
integration means the accounting applies to agents built with a widely used
open-source framework, not only to the bundled router. No adoption or usage
metrics are claimed at this stage; the aim of this submission is to place the
software on record so that cost-per-task can be reported, compared, and cited
alongside accuracy in future agent evaluations.

# Acknowledgements

The author thanks the maintainers of the Strands Agents framework, whose
event-loop metrics surface the `CostTracker` integration builds on.

# References
