"""Embeddable "cost per correct" badges: SVG + shields.io endpoint JSON.

Usage:
    python -m tokenecon.badge results.json --format svg -o badge.svg
    python -m tokenecon.badge results.json --format json
    python -m tokenecon.badge results.json --format markdown --url https://example.com/badge.svg

Input JSON accepts two shapes:
  1. A result summary::
        {"model": "gemini-2.5-flash", "tasks_solved": 5,
         "tasks_total": 5, "total_cost_usd": 0.000057}
  2. A list of tokenecon TaskReceipt dicts (as produced by
     ``TaskReceipt.to_dict()``), optionally wrapped as
     ``{"model": "...", "receipts": [...]}``. Tasks solved are the
     receipts whose outcome is "ok"; total cost is summed.
     Receipts carry no grading information, so for graded benchmark
     runs the summary form is the honest input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape as xml_escape

LABEL = "cost / correct"
DEFAULT_OUTPUT = "cost-per-correct.svg"

#: shields.io-style named colors -> hex.
COLORS: Dict[str, str] = {
    "brightgreen": "#4c1",
    "green": "#97ca00",
    "yellowgreen": "#a4a61d",
    "yellow": "#dfb317",
    "orange": "#fe7d37",
    "red": "#e05d44",
    "lightgrey": "#9f9f9f",
}


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return float(value)


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if int(value) != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return int(value)


def _aggregate_receipts(receipts: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0.0
    solved = 0
    for i, r in enumerate(receipts):
        if not isinstance(r, dict):
            raise ValueError(f"receipt #{i} is not an object: {r!r}")
        total += _require_number(r.get("total_cost"), f"receipts[{i}].total_cost")
        if r.get("outcome", "ok") == "ok":
            solved += 1
    return {"tasks_solved": solved, "tasks_total": len(receipts), "total_cost_usd": total}


def load_results(path: str) -> Dict[str, Any]:
    """Load a results file (summary or tokenecon receipts) into a result dict."""
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, list):
        agg = _aggregate_receipts(data)
        return {"model": None, **agg}
    if isinstance(data, dict):
        if "receipts" in data:
            agg = _aggregate_receipts(data["receipts"])
            return {"model": data.get("model"), **agg}
        solved = _require_int(data.get("tasks_solved"), "tasks_solved")
        total = _require_int(data.get("tasks_total"), "tasks_total")
        if solved > total:
            raise ValueError(f"tasks_solved ({solved}) > tasks_total ({total})")
        cost = _require_number(data.get("total_cost_usd"), "total_cost_usd")
        return {
            "model": data.get("model"),
            "tasks_solved": solved,
            "tasks_total": total,
            "total_cost_usd": cost,
        }
    raise ValueError(f"unsupported results format: expected object or list, got {type(data)}")


def cost_per_correct(results: Dict[str, Any]) -> Optional[float]:
    """Cost per solved task in USD, or None when nothing was solved."""
    if results["tasks_solved"] == 0:
        return None
    return results["total_cost_usd"] / results["tasks_solved"]


def format_cost(value: Optional[float]) -> str:
    """Human badge text for a cost-per-correct value.

    dollars convention follows tokenecon: cents at 2dp for anything that
    reads as real money, 6dp for sub-cent values (as in receipts).
    """
    if value is None:
        return "no data"
    if value == 0:
        return "$0.00"
    if value >= 0.01:
        return f"${value:,.2f}"
    return "$" + f"{value:.6f}".rstrip("0").rstrip(".")


def badge_color(value: Optional[float]) -> str:
    """Named shields color for a cost-per-correct value (USD per solved task)."""
    if value is None:
        return "lightgrey"
    if value < 0.01:
        return "brightgreen"
    if value < 0.10:
        return "green"
    if value < 0.50:
        return "yellowgreen"
    if value < 1.00:
        return "yellow"
    if value < 5.00:
        return "orange"
    return "red"


def endpoint_json(results: Dict[str, Any], label: str = LABEL) -> Dict[str, Any]:
    """shields.io endpoint JSON for this result set."""
    value = cost_per_correct(results)
    return {
        "schemaVersion": 1,
        "label": label,
        "message": format_cost(value),
        "color": badge_color(value),
    }


def _text_width(text: str) -> int:
    # Approximation of 11px Verdana used by shields.io badges.
    return round(sum(6.2 if c not in "ijl.,:;!1'" else 3.1 for c in text))


def _num(value: float) -> str:
    return f"{value:g}"


def render_svg(label: str, message: str, color: str) -> str:
    """Self-contained shields-style flat SVG badge."""
    hex_color = COLORS[color]
    label = xml_escape(label)
    message = xml_escape(message)
    lw = _text_width(label) + 20
    rw = _text_width(message) + 20
    w = lw + rw
    lx = _num(lw / 2)
    rx = _num(lw + rw / 2)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="20" role="img" aria-label="{label}: {message}">
<title>{label}: {message}</title>
<linearGradient id="s" x2="0" y2="1"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient>
<clipPath id="r"><rect width="{w}" height="20" rx="3" fill="#fff"/></clipPath>
<g clip-path="url(#r)">
<rect width="{lw}" height="20" fill="#555"/>
<rect x="{lw}" width="{rw}" height="20" fill="{hex_color}"/>
<rect width="{w}" height="20" fill="url(#s)"/>
</g>
<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
<text aria-hidden="true" x="{lx}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
<text x="{lx}" y="14">{label}</text>
<text aria-hidden="true" x="{rx}" y="15" fill="#010101" fill-opacity=".3">{message}</text>
<text x="{rx}" y="14">{message}</text>
</g>
</svg>
"""


def badge_svg(results: Dict[str, Any], label: str = LABEL) -> str:
    """Render the full badge for a loaded result dict."""
    value = cost_per_correct(results)
    return render_svg(label, format_cost(value), badge_color(value))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tokenecon.badge",
        description="Render an embeddable 'cost per correct' badge from tokenecon results.",
    )
    p.add_argument("input", help="JSON results file (summary or tokenecon receipts)")
    p.add_argument(
        "--format",
        choices=["svg", "json", "markdown"],
        default="svg",
        help="svg: write a badge file (default); json: shields.io endpoint JSON on "
        "stdout; markdown: print a README embed snippet",
    )
    p.add_argument("-o", "--output", help="output file path (default: cost-per-correct.svg)")
    p.add_argument(
        "--url",
        help="badge URL for the markdown snippet (defaults to the output path)",
    )
    p.add_argument("--label", default=LABEL, help="badge label (default: 'cost / correct')")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    results = load_results(args.input)
    if args.format == "json":
        payload = json.dumps(endpoint_json(results, args.label), indent=2)
        if args.output:
            with open(args.output, "w") as fh:
                fh.write(payload + "\n")
        else:
            print(payload)
    elif args.format == "markdown":
        target = args.url or args.output or DEFAULT_OUTPUT
        print(f"![cost per correct]({target})")
    else:
        out = args.output or DEFAULT_OUTPUT
        with open(out, "w") as fh:
            fh.write(badge_svg(results, args.label) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
