"""The PDF report: summary.pdf (two pages) and details.pdf (the evidence), drawn with ReportLab."""

from __future__ import annotations

from pathlib import Path

from ..view import ReportView
from .details import write_details
from .summary import write_summary


def write_pdfs(view: ReportView, out: Path) -> tuple[Path, Path]:
    """Write details.pdf first (summary.pdf cites its page numbers), then summary.pdf."""
    out.mkdir(parents=True, exist_ok=True)
    details, summary = out / "details.pdf", out / "summary.pdf"
    write_details(view, details)
    write_summary(view, summary)
    return summary, details


__all__ = ["write_pdfs"]
