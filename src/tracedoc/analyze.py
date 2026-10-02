"""Orchestration: load traces -> normalize -> rules -> judges -> grouping -> ranking -> Report."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Literal

from .__about__ import __version__
from .checks.base import run_rule_checks
from .checks.judges.prompts import enabled_judges
from .checks.judges.runner import Estimate, Judge, JudgeOutcome, Plan
from .checks.signals import find_signals
from .classify.roles import role_from_name
from .classify.task_type import task_from_text
from .config import Config
from .group import group_findings
from .ingest.base import RawTrace
from .ingest.jsonl import read_captures
from .ingest.langsmith import fetch_project, read_export
from .llm.classify import role_classifier, task_classifier
from .llm.client import LLMClient
from .llm.cost import Spend, price_of
from .normalize import normalize
from .schema import Finding, Report, Run

SourceName = Literal["capture", "langsmith"]


def load_traces(
    path: Path | None, source: SourceName = "capture", project: str | None = None
) -> list[RawTrace]:
    if source == "langsmith":
        if project:
            return fetch_project(project)
        if path is None:
            raise ValueError("LangSmith import needs an export file PATH or --project")
        files = [path] if path.is_file() else sorted(path.glob("*.json*"))
        return [t for f in files for t in read_export(f)]
    if path is None:
        raise ValueError("PATH is required")
    return read_captures(path)


def normalize_all(
    traces: list[RawTrace],
    config: Config,
    *,
    client: LLMClient | None = None,
    spend: Spend | None = None,
) -> list[Run]:
    task_fn = role_fn = None
    if client is not None and spend is not None:
        task_fn = task_classifier(client, config.llm.triage_model, spend)
        role_fn = role_classifier(client, config.llm.triage_model, spend)

    def price(model: str | None, tokens_in: int, tokens_out: int) -> float | None:
        return price_of(config.prices, model, tokens_in, tokens_out)

    return [
        normalize(t, config, task_classifier=task_fn, role_classifier=role_fn, price=price)
        for t in traces
    ]


def classifier_calls(runs: list[Run]) -> int:
    """Model calls the classification fallbacks would make (for estimates)."""
    tasks = sum(1 for r in runs if task_from_text(r.input_preview) is None)
    names = {
        s.node for r in runs for s in r.steps_of("node") if s.node and s.role != "tool_executor"
    }
    return tasks + sum(1 for n in names if role_from_name(n) is None)


@dataclass
class Prepared:
    runs: list[Run]
    rule_findings: list[Finding]
    signals: list[Finding]
    judge: Judge
    plan: Plan
    disabled_judges: dict[str, str]

    def estimate(self) -> Estimate:
        return self.judge.estimate(self.plan)


def prepare(runs: list[Run], config: Config, client: LLMClient | None = None) -> Prepared:
    rule_findings = run_rule_checks(runs, config)
    signals = find_signals(runs, config)
    judges, disabled = enabled_judges(config)
    judge = Judge(config, judges, client=client)
    return Prepared(
        runs=runs,
        rule_findings=rule_findings,
        signals=signals,
        judge=judge,
        plan=judge.plan(runs, rule_findings),
        disabled_judges=disabled,
    )


def _scorecard(runs: list[Run], findings: list[Finding]) -> dict[str, Any]:
    n = len(runs)
    costs = [r.totals.cost for r in runs if r.totals.cost is not None]
    durations = [r.totals.duration_s for r in runs if r.totals.duration_s is not None]
    return {
        "runs": n,
        "completion_rate": sum(r.status == "completed" for r in runs) / n if n else 0.0,
        "median_steps": median(r.totals.step_count for r in runs) if n else 0,
        "total_cost": sum(costs) if costs else None,
        "median_cost": median(costs) if costs else None,
        "median_latency_s": median(durations) if durations else None,
        "tokens": sum(r.totals.tokens_in + r.totals.tokens_out for r in runs),
        "findings_by_severity": {
            sev: sum(1 for f in findings if f.severity == sev) for sev in ("high", "medium", "low")
        },
    }


def build_report(
    prepared: Prepared,
    config: Config,
    outcome: JudgeOutcome | None,
    spend: Spend | None,
) -> Report:
    findings = outcome.findings if outcome else prepared.rule_findings
    groups = group_findings(findings, len(prepared.runs))
    llm = config.llm
    return Report(
        scorecard=_scorecard(prepared.runs, findings),
        groups=groups,
        signals=prepared.signals,
        metadata={
            "tool_version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "runs_analyzed": len(prepared.runs),
            "judged": outcome is not None,
            "analysis_models": (
                {"triage": llm.triage_model, "confirm": llm.confirm_model} if outcome else None
            ),
            "analysis_cost": round(spend.cost, 4) if spend else 0.0,
            "analysis_calls": spend.calls if spend else 0,
            "analysis_unpriced_models": sorted(spend.unpriced_models) if spend else [],
            "judge_errors": outcome.errors if outcome else [],
            "stopped_for_budget": outcome.stopped_for_budget if outcome else None,
            "disabled_judge_checks": prepared.disabled_judges,
        },
    )


def analyze(
    runs: list[Run],
    config: Config,
    *,
    client: LLMClient | None = None,
    spend: Spend | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Report:
    """Run everything on already-normalized runs. Without a client, rules only."""
    prepared = prepare(runs, config, client)
    outcome = None
    if client is not None:
        if spend is not None:
            prepared.judge.spend = spend
        outcome = prepared.judge.run(prepared.plan, prepared.rule_findings, progress)
    return build_report(prepared, config, outcome, prepared.judge.spend if client else spend)
