"""Orchestration: load traces -> normalize -> rules -> judges -> grouping -> ranking -> Report."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
from .ingest.jsonl import is_capture_file, iter_files, read_capture_file
from .ingest.langsmith import fetch_project, read_export
from .llm.classify import role_classifier, task_classifier
from .llm.client import LLMClient
from .llm.cost import Spend, price_of
from .merge import merge_duplicates
from .normalize import normalize
from .schema import Finding, Report, Run

SourceName = Literal["capture", "langsmith"]


@dataclass
class LoadResult:
    traces: list[RawTrace]
    files_read: int = 0
    skipped_files: list[Path] = field(default_factory=list)
    """Files that matched the extension but are not tracedoc traces or LangSmith exports."""
    duplicates: int = 0
    """Runs found more than once (for example through symlinks); each is kept once."""

    def describe(self) -> str:
        parts = [f"Read {len(self.traces)} run(s) from {self.files_read} file(s)"]
        if self.skipped_files:
            parts.append(f"skipped {len(self.skipped_files)} file(s) that are not traces")
        if self.duplicates:
            parts.append(f"ignored {self.duplicates} duplicate run(s)")
        return "; ".join(parts) + "."


def collect_traces(
    paths: Path | Sequence[Path] | None,
    source: SourceName = "capture",
    project: str | None = None,
) -> LoadResult:
    """Read traces from any number of files and folders, searching sub-folders."""
    if source == "langsmith" and project:
        traces = fetch_project(project)
        return _dedupe_traces(LoadResult(traces=traces, files_read=0))
    path_list = [paths] if isinstance(paths, Path) else list(paths or [])
    if not path_list:
        raise ValueError(
            "LangSmith import needs an export file PATH or --project"
            if source == "langsmith"
            else "PATH is required"
        )
    result = LoadResult(traces=[])
    for path in path_list:
        if source == "langsmith":
            for f in iter_files(path, (".json", ".jsonl")):
                try:
                    found = read_export(f)
                except (ValueError, KeyError, TypeError):
                    result.skipped_files.append(f)
                    continue
                result.files_read += 1
                result.traces += found
        else:
            for f in iter_files(path, (".jsonl",)):
                if not is_capture_file(f):
                    result.skipped_files.append(f)
                    continue
                result.files_read += 1
                result.traces.append(read_capture_file(f))
    return _dedupe_traces(result)


def _dedupe_traces(result: LoadResult) -> LoadResult:
    seen: set[str] = set()
    unique = []
    for trace in result.traces:
        if trace.id in seen:
            result.duplicates += 1
            continue
        seen.add(trace.id)
        unique.append(trace)
    result.traces = unique
    return result


def system_name(trace: RawTrace) -> str:
    """What a system is called in its traces: the name of its top-level graph."""
    root = trace.root
    return (root.name if root is not None else "") or ""


def split_systems(traces: list[RawTrace]) -> dict[str, list[RawTrace]]:
    """Traces grouped by system (top-level graph name), in name order. Each system gets its
    own analysis and report, so every percentage is out of that system's runs."""
    systems: dict[str, list[RawTrace]] = defaultdict(list)
    for trace in traces:
        systems[system_name(trace)].append(trace)
    return dict(sorted(systems.items()))


def system_folders(names: Sequence[str]) -> dict[str, str]:
    """A safe, unique sub-folder name for each system's report."""
    used: set[str] = set()
    folders = {}
    for name in names:
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")[:80] or "unnamed_system"
        folder, n = base, 2
        while folder.lower() in used:
            folder, n = f"{base}-{n}", n + 1
        used.add(folder.lower())
        folders[name] = folder
    return folders


def normalize_all(
    traces: list[RawTrace],
    config: Config,
    *,
    client: LLMClient | None = None,
    spend: Spend | None = None,
) -> list[Run]:
    task_fn = role_fn = None
    if client is not None and spend is not None:
        task_fn = task_classifier(client, config.llm.triage_model, spend, config.llm.temperature)
        role_fn = role_classifier(client, config.llm.triage_model, spend, config.llm.temperature)

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
    # Checks that fired at the same agent and step of a run are one problem: one finding.
    findings = merge_duplicates(
        outcome.findings if outcome else prepared.rule_findings, prepared.runs
    )
    groups = group_findings(findings, prepared.runs)
    llm = config.llm
    return Report(
        scorecard=_scorecard(prepared.runs, findings),
        groups=groups,
        # A run with any finding is not an example of an efficient path.
        signals=[
            s
            for s in prepared.signals
            if s.check_id != "good.efficient_path" or s.run_id not in {f.run_id for f in findings}
        ],
        metadata={
            "tool_version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "runs_analyzed": len(prepared.runs),
            "judged": outcome is not None,
            "analysis_models": (
                {"triage": llm.triage_model, "confirm": llm.confirm_model} if outcome else None
            ),
            # What the model answers behind this report cost, including answers reused from
            # the judge cache (so a re-run reports the same figure); spent_now is this run only.
            "analysis_cost": round(spend.report_cost, 4) if spend else 0.0,
            "analysis_cost_spent_now": round(spend.cost, 4) if spend else 0.0,
            "analysis_calls": spend.calls if spend else 0,
            "analysis_answers_reused": spend.reused_calls if spend else 0,
            "analysis_unpriced_models": sorted(spend.unpriced_models) if spend else [],
            "judge_errors": outcome.errors if outcome else [],
            "stopped_for_budget": outcome.stopped_for_budget if outcome else None,
            "disabled_judge_checks": prepared.disabled_judges,
            "unreviewed_by_judge": {
                "rule_candidates": prepared.plan.unreviewed_candidates,
                "handoffs": prepared.plan.unreviewed_handoffs,
                "unsampled_runs": prepared.plan.unsampled_runs,
            }
            if outcome is not None
            else None,
        },
    )


def analyze(
    runs: list[Run],
    config: Config,
    *,
    client: LLMClient | None = None,
    spend: Spend | None = None,
    progress: Callable[[int, int], None] | None = None,
    spend_start: Spend | None = None,
) -> Report:
    """Run everything on already-normalized runs. Without a client, rules only.

    A shared `spend` may carry earlier work (other systems analyzed in the same session); the
    report counts only what was spent since `spend_start` (default: now)."""
    start = spend_start or (spend.snapshot() if spend is not None else None)
    prepared = prepare(runs, config, client)
    outcome = None
    if client is not None:
        if spend is not None:
            prepared.judge.spend = spend
        outcome = prepared.judge.run(prepared.plan, prepared.rule_findings, progress)
    used = spend.since(start) if spend is not None and start is not None else None
    report = build_report(
        prepared, config, outcome, used or (prepared.judge.spend if client else None)
    )
    fallback = getattr(client, "fallback", None)
    if fallback is not None:
        report.metadata["model_fallback"] = {
            "unavailable": fallback.unavailable,
            "used_instead": fallback.replacement,
            "reason": fallback.detail,
        }
        report.metadata["analysis_models"] = {
            "triage": fallback.replacement,
            "confirm": config.llm.confirm_model,
        }
    return report
