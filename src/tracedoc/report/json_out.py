"""report.json: the full Report model, for scripting and CI."""

from __future__ import annotations

from ..schema import Report


def render_json(report: Report) -> str:
    return report.model_dump_json(indent=2)
