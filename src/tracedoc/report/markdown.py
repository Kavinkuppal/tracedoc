"""report.md: the scorecard, the top three fixes, and a pointer to the HTML report."""

from __future__ import annotations

from ..schema import Report
from .data import check_titles


def _money(x: float | None) -> str:
    return "n/a" if x is None else f"${x:,.2f}"


def _secs(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1f}s"


def render_markdown(report: Report, html_name: str = "report.html") -> str:
    sc, meta = report.scorecard, report.metadata
    titles = check_titles()
    sev = sc["findings_by_severity"]
    lines = [
        "# Agent trace diagnosis",
        "",
        f"{sc['runs']} runs analyzed. Full interactive report: [{html_name}]({html_name})",
        "",
        "## Scorecard",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Completion rate | {sc['completion_rate']:.0%} |",
        f"| Median steps | {sc['median_steps']} |",
        f"| Total cost | {_money(sc['total_cost'])} |",
        f"| Median latency | {_secs(sc['median_latency_s'])} |",
        f"| Findings | {sev['high']} high, {sev['medium']} medium, {sev['low']} low |",
        "",
        "## Top three fixes",
        "",
    ]
    if not report.groups:
        lines.append("No problems found.")
    for i, g in enumerate(report.groups[:3], 1):
        f = g.representative
        tag = " (unconfirmed)" if f.needs_confirmation else ""
        where = f" at `{g.node}`" if g.node else ""
        lines += [
            f"### {i}. {titles.get(g.check_id, g.check_id)}{tag}",
            "",
            f"**{f.severity.capitalize()}** · `{g.check_id}`{where} · "
            f"{g.runs_affected} run(s) ({g.frequency:.0%})",
            "",
            f.explanation,
            "",
            f"**Suggested fix:** {f.suggested_fix}",
            "",
        ]
    if meta.get("judged"):
        models = meta.get("analysis_models") or {}
        lines += [
            "---",
            "",
            f"Judged with {models.get('triage')} (triage) and {models.get('confirm')} "
            f"(confirmation). Analysis cost: {_money(meta.get('analysis_cost'))}.",
        ]
    else:
        lines += ["---", "", "Rule checks only; no model was used."]
    return "\n".join(lines) + "\n"
