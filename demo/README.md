# tokenecon demo

`demo.gif` (top of the repo README) renders two live runs of the shipped
code made on Nov 9, 2026. The full record is here:

| File | What it is |
|---|---|
| `cli-demo-output.txt` | `tokenecon demo`: five sample tasks through the tiered router. Every task returns a receipt (tier, tokens, cost, confidence per step). Router total **$0.029307** vs **$0.047454** for serving everything on the large tier — **61.76% of baseline**. |
| `badge-output.txt` | The same five tasks re-run in-process; their real receipt total feeds `tokenecon.badge`: **cost per correct = $0.005861 per solved task** (5/5 solved). |
| `cost-per-correct.svg` | The badge rendered from those receipts. |
| `run_demo.py` | Re-runs both and rewrites the outputs. |
| `transcript.txt` | Every command run for the demo, with full output and exit codes. |

Honesty note: the CLI demo uses the repo's deterministic `MockModel` (no
API keys), so it demonstrates the routing, escalation, budget and receipt
machinery — not a live model's accuracy. Real-model cost-per-correct
numbers are published separately as the Cost Per Correct v2 dataset
(65 tasks, Kaggle / Hugging Face).
