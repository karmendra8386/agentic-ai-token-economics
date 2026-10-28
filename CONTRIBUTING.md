# Contributing to tokenecon

Thanks for your interest in improving tokenecon. Contributions of all sizes
are welcome: bug reports, documentation fixes, new model-tier presets, and
improvements to the cost model, router, receipts, or the Strands integration.

## Ground rules

- **Dependency-free core.** tokenecon's core package intentionally has no
  runtime dependencies. Pull requests that add a hard dependency to the core
  will not be accepted; optional integrations must follow the existing pattern
  in `tokenecon/integrations/` (read the host framework's data, never import
  tokenecon *into* a required dependency, and keep the integration importable
  without the host framework installed).
- **Python 3.9+.** Code must run on every supported version (see the PyPI
  classifiers in `pyproject.toml`).
- **Honest numbers.** Pricing defaults in tokenecon are illustrative
  placeholders. Do not present default tier prices as any provider's real
  price list, in code, docs, or examples.

## Reporting bugs and requesting features

Open an issue at
https://github.com/karmendra8386/agentic-ai-token-economics/issues with:

- what you ran (library call or `tokenecon` CLI command) and what you expected,
- what actually happened, including the full traceback for errors,
- your Python version and tokenecon version (`pip show tokenecon`).

For cost-model questions (for example, how a four-stream cost decomposition
should treat a new workload shape), open an issue describing the workload
before writing code — the model in the companion paper is the design
reference, and changes to the equations should be discussed first.

## Development setup

```bash
git clone https://github.com/karmendra8386/agentic-ai-token-economics.git
cd agentic-ai-token-economics
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest
```

## Running the tests

```bash
python -m pytest tests/ -q
```

All tests run offline: the suite uses the built-in mock model and fake
Strands metric objects, so no API keys, network access, or live agent
deployment are needed. New behaviour must come with tests in the same style.
The full suite (58 tests at v0.2.3) should pass before you open a pull
request.

## Pull requests

1. Fork the repository and create a topic branch from `main`.
2. Keep the change focused; unrelated refactors belong in separate PRs.
3. Add or update tests for behaviour changes, and update the README if the
   public API, CLI, or documented behaviour changes.
4. Make sure `python -m pytest tests/ -q` passes.
5. In the PR description, state the problem, the change, and how you tested
   it. For cost-model or routing-policy changes, include a worked numeric
   example showing the before/after cost or routing outcome.

Maintainer review is by the project author; expect questions on design fit
with the cost model before merge. Response times are best-effort — this is a
single-maintainer project.

## Support and governance

- Support is via GitHub issues only; there is no commercial support offering.
- The project author is the maintainer and final decision-maker on scope,
  the cost model's equations, and releases.
- Releases are cut from tagged versions (e.g. `v0.2.0`) and published to PyPI
  via the repository's trusted-publisher workflow; each release is archived
  on Zenodo (software DOI in `CITATION.cff`).

## License

By contributing, you agree that your contributions are licensed under the
MIT License (see `LICENSE`).
