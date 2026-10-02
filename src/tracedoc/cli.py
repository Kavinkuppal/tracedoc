"""Command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer

from .__about__ import NAME, __version__
from .analyze import SourceName, classifier_calls, load_traces, normalize_all, prepare
from .analyze import analyze as run_analysis
from .config import Config, load_config
from .feedback import check_for, import_file, record
from .llm.client import has_credentials, make_client
from .llm.cost import Spend, price_of
from .report import write_reports
from .schema import Report

app = typer.Typer(name=NAME, help="Diagnose LangGraph agent traces.", no_args_is_help=True)

# Rough size of one classification fallback call, for estimates.
_CLASSIFY_IN, _CLASSIFY_OUT = 150, 20


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        typer.echo(f"{NAME} {__version__}")
        raise typer.Exit()


def _config(
    config_path: Path | None,
    model: str | None,
    triage_model: str | None,
    include_unvalidated: bool,
) -> Config:
    config = load_config(config_path)
    if model:
        config.llm.confirm_model = model
    if triage_model:
        config.llm.triage_model = triage_model
    if include_unvalidated:
        config.judge.include_unvalidated = True
    return config


def _money(value: float | None) -> str:
    return "unknown (model not in price table)" if value is None else f"${value:,.2f}"


def _print_estimate(config: Config, runs: list, prepared) -> float | None:  # noqa: ANN001
    est = prepared.estimate()
    calls = classifier_calls(runs)
    classify_cost = price_of(
        config.prices, config.llm.triage_model, calls * _CLASSIFY_IN, calls * _CLASSIFY_OUT
    )
    total = None if est.cost is None or classify_cost is None else est.cost + classify_cost
    total_max = (
        None if est.cost_max is None or classify_cost is None else est.cost_max + classify_cost
    )
    typer.echo(f"Runs: {len(runs)}   rule findings: {len(prepared.rule_findings)}")
    typer.echo(
        f"Judge windows: {len(prepared.plan.tasks)} "
        f"({prepared.plan.candidates} rule candidates to confirm, "
        f"{est.triage_calls} triage calls on {config.llm.triage_model})"
    )
    typer.echo(
        f"Confirmation calls on {config.llm.confirm_model}: about {est.confirm_calls_expected} "
        f"(at most {est.confirm_calls_max})"
    )
    if calls:
        typer.echo(f"Classification fallback calls: {calls}")
    typer.echo(f"Estimated cost: {_money(total)} (at most {_money(total_max)})")
    if est.unpriced_models:
        typer.echo(f"Add prices for {', '.join(est.unpriced_models)} under [prices] in config.")
    if prepared.disabled_judges:
        typer.echo(
            f"Judge checks disabled ({len(prepared.disabled_judges)}): "
            + ", ".join(sorted(prepared.disabled_judges))
            + ". Run evals/run_evals.py, or pass --include-unvalidated."
        )
    return total


@app.command()
def estimate(
    path: Path | None = typer.Argument(None, help="Capture directory/file or LangSmith export."),
    source: SourceName = typer.Option("capture", help="capture or langsmith"),
    project: str | None = typer.Option(None, help="LangSmith project to pull from."),
    model: str | None = typer.Option(None, help="Confirmation model."),
    triage_model: str | None = typer.Option(None, help="Triage model."),
    include_unvalidated: bool = typer.Option(
        False, help="Include judge checks below the precision bar."
    ),
    config_path: Path | None = typer.Option(None, "--config", help="Extra config TOML."),
) -> None:
    """Print what analysis would cost, without calling any model."""
    config = _config(config_path, model, triage_model, include_unvalidated)
    runs = normalize_all(load_traces(path, source, project), config)
    _print_estimate(config, runs, prepare(runs, config))


@app.command()
def analyze(
    path: Path | None = typer.Argument(None, help="Capture directory/file or LangSmith export."),
    source: SourceName = typer.Option("capture", help="capture or langsmith"),
    project: str | None = typer.Option(None, help="LangSmith project to pull from."),
    model: str | None = typer.Option(None, help="Confirmation model."),
    triage_model: str | None = typer.Option(None, help="Triage model."),
    max_cost: float | None = typer.Option(
        None, help="Stop judging before spend passes this (USD)."
    ),
    out: Path = typer.Option(Path("tracedoc-report"), help="Output directory."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask before spending."),
    no_judge: bool = typer.Option(False, help="Rule checks only; no model calls."),
    include_unvalidated: bool = typer.Option(
        False, help="Include judge checks below the precision bar."
    ),
    config_path: Path | None = typer.Option(None, "--config", help="Extra config TOML."),
) -> None:
    """Analyze traces and write a report."""
    config = _config(config_path, model, triage_model, include_unvalidated)
    traces = load_traces(path, source, project)
    if not traces:
        typer.echo("No traces found.", err=True)
        raise typer.Exit(1)

    use_model = not no_judge and has_credentials(config.llm)
    if not no_judge and not use_model:
        typer.echo(
            f"No key found for {config.llm.provider}; running rule checks only. "
            "Set the API key (or --no-judge to silence this)."
        )
    runs = normalize_all(traces, config)
    client = spend = None
    if use_model:
        prepared = prepare(runs, config)
        estimated = _print_estimate(config, runs, prepared)
        if max_cost is not None and estimated is not None and estimated > max_cost:
            typer.echo(f"Estimate is above --max-cost {_money(max_cost)}; judging will stop there.")
        if not yes and not typer.confirm("Proceed?", default=False):
            raise typer.Exit(1)
        client = make_client(config.llm)
        spend = Spend(prices=config.prices, max_cost=max_cost)
        if classifier_calls(runs):
            runs = normalize_all(traces, config, client=client, spend=spend)

    def progress(done: int, total: int) -> None:
        typer.echo(f"\rJudging {done}/{total}", nl=done == total)

    report = run_analysis(runs, config, client=client, spend=spend, progress=progress)
    html_path = write_reports(report, runs, out)
    _print_summary(report, out, html_path)


def _print_summary(report: Report, out: Path, html_path: Path) -> None:
    sc, meta = report.scorecard, report.metadata
    typer.echo(
        f"\n{sc['runs']} runs, {sc['completion_rate']:.0%} completed, "
        f"median {sc['median_steps']} steps. Findings: "
        + ", ".join(f"{n} {sev}" for sev, n in sc["findings_by_severity"].items())
    )
    if meta.get("stopped_for_budget"):
        typer.echo(f"Judging stopped early: {meta['stopped_for_budget']}")
    if meta.get("judge_errors"):
        typer.echo(f"{len(meta['judge_errors'])} judge calls failed; see report.json.")
    typer.echo("\nTop fixes:")
    for i, group in enumerate(report.groups[:3], 1):
        f = group.representative
        unconfirmed = " (unconfirmed)" if f.needs_confirmation else ""
        typer.echo(
            f"{i}. [{f.severity}] {group.check_id}{unconfirmed}: {group.runs_affected} run(s). "
            f"{f.explanation}\n   Fix: {f.suggested_fix}"
        )
    if meta.get("judged"):
        typer.echo(
            f"\nAnalysis cost: ${meta['analysis_cost']:.2f} over {meta['analysis_calls']} calls"
        )
    typer.echo(f"\nReport: {html_path}  (also report.md, report.json)")


@app.command()
def feedback(
    finding_id: str | None = typer.Argument(None, help="Finding ID shown in the report."),
    vote: str | None = typer.Argument(None, help="up or down"),
    import_path: Path | None = typer.Option(
        None, "--import", help="Import votes exported from the HTML report."
    ),
    report_path: Path = typer.Option(
        Path("tracedoc-report/report.json"), "--report", help="Report to look the finding up in."
    ),
) -> None:
    """Record whether a finding was useful. Stored locally."""
    if import_path is not None:
        count = import_file(import_path)
        typer.echo(f"Imported {count} vote(s).")
        return
    if finding_id is None or vote not in ("up", "down"):
        typer.echo("Usage: tracedoc feedback FINDING_ID up|down, or --import FILE", err=True)
        raise typer.Exit(2)
    record(finding_id, vote, check_for(finding_id, report_path))  # type: ignore[arg-type]
    typer.echo(f"Recorded {vote} for {finding_id}.")


if __name__ == "__main__":
    app()
