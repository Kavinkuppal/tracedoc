"""Report outputs: HTML (interactive, offline), Markdown, and JSON."""

from __future__ import annotations

from pathlib import Path

from ..schema import Finding, Report, Run
from .data import build_rows, fill_report
from .html import render_html
from .json_out import render_json
from .markdown import render_markdown


def all_findings(report: Report) -> list[Finding]:
    return [f for g in report.groups for f in g.findings]


def write_reports(report: Report, runs: list[Run], out: Path) -> Path:
    """Fill node stats and distributions, then write report.{html,md,json} to `out`."""
    findings = all_findings(report)
    fill_report(report, build_rows(runs, findings, report.signals, text_limit=80))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(render_json(report), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    html_path = out / "report.html"
    html_path.write_text(render_html(report, runs, findings), encoding="utf-8")
    return html_path


__all__ = ["all_findings", "render_html", "render_json", "render_markdown", "write_reports"]
