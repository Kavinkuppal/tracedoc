"""Render the self-contained, offline HTML report."""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ..__about__ import NAME, __version__
from ..schema import Finding, Report, Run
from .data import build_rows, payload

HERE = Path(__file__).parent
TEMPLATES = HERE / "templates"
VENDOR = HERE / "vendor"
TARGET_DATA_BYTES = 3_500_000
"""Embedded data budget. With ECharts (~1.1 MB) the page stays under 5 MB."""
TEXT_LIMITS = (600, 300, 150, 80)


@cache
def _asset(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _script_safe(code: str) -> str:
    """Inline JavaScript must not contain `</script` (it would close the tag)."""
    return re.sub(r"</(script)", r"<\\/\1", code, flags=re.IGNORECASE)


def _json_safe(data: Any) -> str:
    """JSON for a <script> block: escaping every `<` rules out `</script>` and `<!--` tricks."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")


def embedded_data(report: Report, runs: list[Run], findings: list[Finding]) -> dict[str, Any]:
    """Shrink inputs and outputs until the data fits the size budget."""
    for limit in TEXT_LIMITS:
        rows = build_rows(runs, findings, report.signals, limit)
        data = payload(report, rows)
        if len(json.dumps(data, separators=(",", ":"))) <= TARGET_DATA_BYTES:
            break
    data["text_limit"] = limit
    return data


def render_html(report: Report, runs: list[Run], findings: list[Finding]) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=False)
    meta = report.metadata
    models = meta.get("analysis_models") or {}
    data = embedded_data(report, runs, findings)
    from markupsafe import escape

    return env.get_template("report.html.j2").render(
        name=NAME,
        version=__version__,
        runs=len(runs),
        generated_at=escape(str(meta.get("generated_at", ""))[:16].replace("T", " ")),
        models=escape(" / ".join(models.values())) if models else "",
        analysis_cost=meta.get("analysis_cost") if meta.get("judged") else None,
        analysis_cost_cached=bool(meta.get("analysis_answers_reused")),
        css=_asset(TEMPLATES / "report.css"),
        app_js=_script_safe(_asset(TEMPLATES / "app.js")),
        echarts_js=_script_safe(_asset(VENDOR / "echarts.min.js")),
        echarts_version=_asset(VENDOR / "ECHARTS_VERSION").strip(),
        data_json=_json_safe(data),
    )
