"""Phase 4: report data, HTML, Markdown, JSON, and feedback."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from markdown_it import MarkdownIt

from tests.fixtures.agents import AGENTS
from tests.helpers import load_fixture
from tracedoc.analyze import analyze
from tracedoc.config import Config
from tracedoc.report import all_findings, render_html, render_markdown, write_reports
from tracedoc.report.data import SEVERITIES, aggregate, build_rows, fill_report
from tracedoc.report.html import TEMPLATES, embedded_data
from tracedoc.schema import Report

SNAPSHOT = Path(__file__).parent / "snapshots" / "report_sections.json"
UPDATE = os.environ.get("UPDATE_SNAPSHOTS") == "1"


@pytest.fixture(scope="module")
def analyzed() -> tuple[Report, list]:
    runs = [load_fixture(n) for n in AGENTS]
    report = analyze(runs, Config())
    return report, runs


@pytest.fixture(scope="module")
def rows(analyzed) -> dict[str, Any]:
    report, runs = analyzed
    return build_rows(runs, all_findings(report), report.signals, text_limit=200)


def _sections(report: Report, rows: dict[str, Any]) -> dict[str, Any]:
    """The data behind each of the 14 report sections, in a stable, comparable form."""
    agg = aggregate(rows)
    run_index = {r["id"]: i for i, r in enumerate(rows["runs"])}
    first_run = rows["runs"][0]["id"]
    return {
        "01_scorecard": agg["scorecard"],
        "02_top_fixes": [
            [g.check_id, g.representative.severity, g.runs_affected, g.node, g.key]
            for g in report.groups[:3]
        ],
        "03_failure_frequency": agg["failure_frequency"],
        "04_failures_by_task": agg["failures_by_task"],
        "05_status_by_task": agg["status_by_task"],
        "06_flow": agg["flow"],
        "07_handoffs": agg["handoffs"],
        "08_cost_by_node": agg["cost_by_node"],
        "09_context_growth": {
            **agg["context_growth"],
            "worst_run": run_index.get(agg["context_growth"]["worst_run"]),
        },
        "10_steps_per_run": agg["steps_per_run"],
        "11_latency_by_node": [[n, len(q), c] for n, q, c in agg["latency_by_node"]],
        "12_replay": [
            [
                s["kind"],
                s["name"],
                s["node"],
                s["tin"],
                s["tout"],
                s["err"],
                s["in"][:60],
                s["out"][:60],
            ]
            for s in rows["steps"]
            if s["run"] == first_run and s["kind"] != "node"
        ],
        "13_signals": sorted([s["check"], run_index[s["run"]], s["key"]] for s in rows["signals"]),
        "14_findings": sorted(
            [f["check"], f["severity"], run_index[f["run"]], f["node"], f["key"], f["unconfirmed"]]
            for f in rows["findings"]
        ),
    }


def _snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT.read_text()) if SNAPSHOT.is_file() else {}


SECTIONS = [f"{i:02d}" for i in range(1, 15)]


@pytest.mark.parametrize("section", SECTIONS)
def test_section_snapshot(section: str, analyzed, rows) -> None:
    report, _ = analyzed
    current = json.loads(json.dumps(_sections(report, rows)))
    key = next(k for k in current if k.startswith(section))
    if UPDATE:
        data = _snapshot()
        data[key] = current[key]
        SNAPSHOT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    assert key in _snapshot(), f"no snapshot for {key}; run with UPDATE_SNAPSHOTS=1"
    assert current[key] == _snapshot()[key]


def test_snapshot_covers_every_section() -> None:
    assert sorted(k[:2] for k in _snapshot()) == SECTIONS


def test_filters_narrow_consistently(rows) -> None:
    full = aggregate(rows)
    research = aggregate(rows, task="research")
    assert research["scorecard"]["runs"] == 4
    assert set(research["status_by_task"]) == {"research"}
    high = aggregate(rows, sev="high")
    assert set(high["scorecard"]["findings_by_severity"]) == set(SEVERITIES)
    assert high["scorecard"]["findings_by_severity"]["medium"] == 0
    writer = aggregate(rows, node="writer")
    assert {r["node"] for r in writer["cost_by_node"]} == {"writer"}
    assert writer["scorecard"]["runs"] < full["scorecard"]["runs"]


# ---- JS mirrors Python ----------------------------------------------------------

NODE = shutil.which("node")
FILTERS = [
    {},
    {"task": "research"},
    {"task": "writing"},
    {"node": "writer"},
    {"node": "tools"},
    {"sev": "high"},
    {"sev": "low", "task": "research"},
    {"node": "supervisor", "sev": "medium"},
]


def _close(a: Any, b: Any, path: str = "") -> None:
    if isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, abs=1e-3), path
    elif isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _close(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_javascript_aggregate_matches_python(rows, tmp_path: Path) -> None:
    data = tmp_path / "rows.json"
    data.write_text(json.dumps({"rows": rows, "filters": FILTERS}))
    script = (
        f"const {{aggregate}} = require({json.dumps(str(TEMPLATES / 'app.js'))});"
        f"const d = JSON.parse(require('fs').readFileSync({json.dumps(str(data))}, 'utf8'));"
        "const run = (f) => aggregate(d.rows, f.task ?? null, f.node ?? null, f.sev ?? null);"
        "console.log(JSON.stringify(d.filters.map(run)));"
    )
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True)
    js_results = json.loads(out.stdout)
    for f, js in zip(FILTERS, js_results, strict=True):
        py = json.loads(json.dumps(aggregate(rows, **f)))
        _close(js, py, str(f))


# ---- HTML -------------------------------------------------------------------------


def test_html_is_self_contained(analyzed) -> None:
    report, runs = analyzed
    html = render_html(report, runs, all_findings(report))
    markup = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.S)  # code strings aside
    assert not re.search(r"<(script|link|img|iframe)[^>]+(src|href)=", markup)
    assert len(re.findall(r"<script\b", markup)) == 0
    assert "@import" not in html
    assert not re.search(r"url\(\s*['\"]?https?:", html)
    assert "echarts" in html and "Apache" in html
    for section in (
        "scorecard",
        "top-fixes",
        "c-frequency",
        "c-bytask",
        "c-status",
        "c-flow",
        "c-handoffs",
        "c-cost",
        "c-context",
        "c-steps",
        "c-latency",
        "replay",
        "signals-body",
        "findings-body",
    ):
        assert f'id="{section}"' in html, section


def test_embedded_data_round_trips_and_cannot_close_script(analyzed) -> None:
    report, runs = analyzed
    evil = copy.deepcopy(runs)
    evil[0].input_preview = "</script><script>alert(1)</script><!--"
    html = render_html(report, evil, all_findings(report))
    match = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S
    )
    assert match
    data = json.loads(match.group(1))
    assert data["rows"]["runs"][0]["input"].startswith("</script>")
    assert html.count("<script") == 3


def _big_runs(n: int) -> list:
    base = [load_fixture(name) for name in AGENTS]
    runs = []
    for i in range(n):
        run = copy.deepcopy(base[i % len(base)])
        run.id = f"{run.id}-{i}"
        for s in run.steps:
            s.run_id = run.id
            if s.kind in ("llm", "tool"):
                s.output = "x" * 20_000
        runs.append(run)
    return runs


def test_100_runs_under_5_mb() -> None:
    runs = _big_runs(100)
    report = analyze(runs, Config())
    html = render_html(report, runs, all_findings(report))
    size = len(html.encode())
    assert size < 5_000_000, size
    assert embedded_data(report, runs, all_findings(report))["text_limit"] >= 80


# ---- Markdown, JSON, files ----------------------------------------------------------


def test_write_reports_outputs_validate(analyzed, tmp_path: Path) -> None:
    report, runs = analyzed
    files = write_reports(report.model_copy(deep=True), runs, tmp_path, html=True)
    assert files.html == tmp_path / "report.html"
    assert files.summary == tmp_path / "summary.pdf" and files.summary.is_file()
    assert files.details == tmp_path / "details.pdf" and files.details.is_file()
    parsed = Report.model_validate_json((tmp_path / "report.json").read_text())
    assert parsed.distributions["flow"]["nodes"]
    assert parsed.node_stats["writer"]["visits"] == 5  # pipeline 1, healthy 1, supervisor 3
    md = (tmp_path / "report.md").read_text()
    tokens = MarkdownIt("commonmark").enable("table").parse(md)
    headings = [
        (t.tag, tokens[i + 1].content) for i, t in enumerate(tokens) if t.type == "heading_open"
    ]
    assert headings[:3] == [
        ("h1", "Agent trace diagnosis"),
        ("h2", "Scorecard"),
        ("h2", "Top three fixes"),
    ]
    assert [tag for tag, _ in headings].count("h3") == 3
    assert sum(t.type == "table_open" for t in tokens) == 1
    assert sum(t.type == "tr_open" for t in tokens) == 6  # header + 5 metrics
    links = [
        c.attrs.get("href")
        for t in tokens
        if t.children
        for c in t.children
        if c.type == "link_open"
    ]
    assert links == ["report.html"]


def test_markdown_without_findings() -> None:
    report = Report(
        scorecard={
            "runs": 0,
            "completion_rate": 0.0,
            "median_steps": 0,
            "total_cost": None,
            "median_latency_s": None,
            "findings_by_severity": {"high": 0, "medium": 0, "low": 0},
        }
    )
    assert "No problems found." in render_markdown(report)


def test_fill_report_matches_aggregate(analyzed, rows) -> None:
    report, _ = analyzed
    filled = fill_report(report.model_copy(deep=True), rows)
    assert filled.distributions["handoffs"] == aggregate(rows)["handoffs"]
