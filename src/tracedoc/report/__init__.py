"""Report outputs: summary.pdf and details.pdf (the report), report.json (for scripts and CI),
and, on request, the older interactive HTML report with its Markdown summary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schema import Finding, Report, Run
from .data import build_rows, fill_report
from .html import render_html
from .json_out import render_json
from .markdown import render_markdown
from .pdf import write_pdfs
from .view import ReportView, build_view


@dataclass(frozen=True)
class ReportFiles:
    summary: Path
    details: Path
    json: Path
    view: ReportView
    """What the PDFs show (verdict, fixes, ...), for the command line's own summary."""
    html: Path | None = None
    markdown: Path | None = None


def all_findings(report: Report) -> list[Finding]:
    return [f for g in report.groups for f in g.findings]


def write_reports(
    report: Report,
    runs: list[Run],
    out: Path,
    *,
    html: bool = False,
    name: str | None = None,
    config: Any = None,
) -> ReportFiles:
    """Write summary.pdf, details.pdf and report.json to `out`; with html=True, also
    report.html and report.md. `name` overrides the system name taken from the traces."""
    findings = all_findings(report)
    fill_report(report, build_rows(runs, findings, report.signals, text_limit=80))
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    json_path.write_text(render_json(report), encoding="utf-8")
    view = build_view(report, runs, name=name, config=config)
    summary, details = write_pdfs(view, out)
    html_path = md_path = None
    if html:
        md_path = out / "report.md"
        md_path.write_text(render_markdown(report), encoding="utf-8")
        html_path = out / "report.html"
        html_path.write_text(render_html(report, runs, findings), encoding="utf-8")
    return ReportFiles(summary, details, json_path, view, html_path, md_path)


__all__ = [
    "ReportFiles",
    "all_findings",
    "render_html",
    "render_json",
    "render_markdown",
    "write_reports",
]
