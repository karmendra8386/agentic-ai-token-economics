"""Tests for tokenecon.badge: cost-per-correct math, SVG/JSON badge output."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from tokenecon import TaskReceipt
from tokenecon.badge import (
    badge_color,
    badge_svg,
    cost_per_correct,
    endpoint_json,
    format_cost,
    load_results,
    main,
)

SUMMARY = {
    "model": "gemini-2.5-flash",
    "tasks_solved": 5,
    "tasks_total": 5,
    "total_cost_usd": 2.10,
}


def _write(tmp_path, data, name="results.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def test_load_summary_and_cost_per_correct(tmp_path):
    results = load_results(_write(tmp_path, SUMMARY))
    assert results["model"] == "gemini-2.5-flash"
    assert cost_per_correct(results) == pytest.approx(0.42)


def test_load_receipts_aggregates(tmp_path):
    receipts = [
        TaskReceipt(request="a", difficulty=0.1, total_cost=0.30, outcome="ok").to_dict(),
        TaskReceipt(request="b", difficulty=0.5, total_cost=0.50, outcome="ok").to_dict(),
        TaskReceipt(request="c", difficulty=0.9, total_cost=0.70, outcome="degraded").to_dict(),
    ]
    results = load_results(_write(tmp_path, receipts))
    assert results["tasks_total"] == 3
    assert results["tasks_solved"] == 2
    assert cost_per_correct(results) == pytest.approx(1.50 / 2)


def test_load_receipts_wrapped_with_model(tmp_path):
    data = {"model": "m", "receipts": [
        TaskReceipt(request="a", difficulty=0.1, total_cost=0.25, outcome="ok").to_dict()
    ]}
    results = load_results(_write(tmp_path, data))
    assert results["model"] == "m"
    assert cost_per_correct(results) == pytest.approx(0.25)


def test_invalid_results_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_results(_write(tmp_path, {"tasks_solved": 6, "tasks_total": 5,
                                       "total_cost_usd": 1.0}))
    with pytest.raises(ValueError):
        load_results(_write(tmp_path, {"tasks_solved": 1, "tasks_total": 5,
                                       "total_cost_usd": -1.0}))
    with pytest.raises(ValueError):
        load_results(_write(tmp_path, {"tasks_solved": 1, "tasks_total": 5}))


def test_zero_solved_is_no_data(tmp_path):
    results = load_results(_write(tmp_path, {
        "tasks_solved": 0, "tasks_total": 5, "total_cost_usd": 1.0}))
    assert cost_per_correct(results) is None
    assert endpoint_json(results)["message"] == "no data"
    assert endpoint_json(results)["color"] == "lightgrey"


def test_format_cost():
    assert format_cost(None) == "no data"
    assert format_cost(0.0) == "$0.00"
    assert format_cost(0.42) == "$0.42"
    assert format_cost(1.5) == "$1.50"
    assert format_cost(1234.567) == "$1,234.57"
    assert format_cost(0.0000113) == "$0.000011"


def test_badge_color_scale():
    assert badge_color(None) == "lightgrey"
    assert badge_color(0.005) == "brightgreen"
    assert badge_color(0.05) == "green"
    assert badge_color(0.20) == "yellowgreen"
    assert badge_color(0.75) == "yellow"
    assert badge_color(2.00) == "orange"
    assert badge_color(9.99) == "red"


def test_endpoint_json_shape():
    payload = endpoint_json(SUMMARY)
    assert payload == {
        "schemaVersion": 1,
        "label": "cost / correct",
        "message": "$0.42",
        "color": "yellowgreen",
    }


def test_svg_is_self_contained_and_parseable():
    svg = badge_svg(SUMMARY)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert root.get("height") == "20"
    assert "cost / correct" in svg
    assert "$0.42" in svg
    assert "#a4a61d" in svg  # yellowgreen fill for $0.42
    assert "http" not in svg.replace('xmlns="http://www.w3.org/2000/svg"', "")


def test_cli_json_and_markdown_modes(tmp_path, capsys):
    path = _write(tmp_path, SUMMARY)
    assert main([path, "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["message"] == "$0.42"

    assert main([path, "--format", "markdown",
                 "--url", "https://example.com/badge.svg"]) == 0
    assert capsys.readouterr().out.strip() == \
        "![cost per correct](https://example.com/badge.svg)"


def test_cli_svg_mode_writes_file(tmp_path):
    path = _write(tmp_path, SUMMARY)
    out = tmp_path / "b.svg"
    assert main([path, "--format", "svg", "-o", str(out)]) == 0
    assert "$0.42" in out.read_text()


def test_module_entrypoint_help():
    proc = subprocess.run(
        [sys.executable, "-m", "tokenecon.badge", "--help"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "cost per correct" in proc.stdout
