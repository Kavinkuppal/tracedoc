"""Command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer

from .__about__ import NAME, __version__
from .analyze import SourceName, classifier_calls, load_traces, normalize_all, prepare
from .analyze import analyze as run_analysis
from .config import Config, ConfigError, config_dir, load_config, user_config_path
from .feedback import check_for, feedback_path, import_file, record
from .ingest.jsonl import CaptureFormatError
from .llm.client import KeyRejectedError, MissingKeyError, make_client, require_key
from .llm.cost import Spend, price_of
from .report import write_reports
from .schema import Report

app = typer.Typer(
    name=NAME,
    help="Diagnose LangGraph agent traces.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

# Problems the user can fix: one line on stderr, exit code 2, never a stack trace.
USER_ERRORS = (
    MissingKeyError,
    KeyRejectedError,
    ConfigError,
    CaptureFormatError,
    FileNotFoundError,
    NotADirectoryError,
    ValueError,
)


def _fail(error: BaseException) -> typer.Exit:
    typer.echo(f"Error: {error}", err=True)
    return typer.Exit(2)


# Rough size of one classification fallback call, for estimates.
_CLASSIFY_IN, _CLASSIFY_OUT = 150, 20


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        typer.echo(f"{NAME} {__version__}")
        raise typer.Exit()


def _load(path: Path | None, source: SourceName, project: str | None) -> list:
    if path is not None and not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    traces = load_traces(path, source, project)
    if not traces:
        where = f"LangSmith project {project}" if project else str(path)
        raise ValueError(f"no traces found in {where}")
    return traces


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
    try:
        config = _config(config_path, model, triage_model, include_unvalidated)
        try:
            require_key(config.llm)
        except MissingKeyError as e:
            hint = "or use `tracedoc analyze --no-judge` for rule checks only (no key needed)"
            raise MissingKeyError(e.env_var, hint=hint) from None
        runs = normalize_all(_load(path, source, project), config)
    except USER_ERRORS as e:
        raise _fail(e) from None
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
    try:
        config = _config(config_path, model, triage_model, include_unvalidated)
        use_model = not no_judge
        if use_model:
            require_key(config.llm)  # fail fast, before any work
        traces = _load(path, source, project)
    except USER_ERRORS as e:
        raise _fail(e) from None
    runs = normalize_all(traces, config)
    client = spend = None
    if use_model:
        prepared = prepare(runs, config)
        estimated = _print_estimate(config, runs, prepared)
        if max_cost is not None and estimated is not None and estimated > max_cost:
            typer.echo(f"Estimate is above --max-cost {_money(max_cost)}; judging will stop there.")
        if not yes and not typer.confirm("Proceed?", default=False):
            raise typer.Exit(1)

    def progress(done: int, total: int) -> None:
        typer.echo(f"\rJudging {done}/{total}", nl=done == total)

    try:
        if use_model:
            client = make_client(config.llm)
            spend = Spend(prices=config.prices, max_cost=max_cost)
            if classifier_calls(runs):
                runs = normalize_all(traces, config, client=client, spend=spend)
        report = run_analysis(runs, config, client=client, spend=spend, progress=progress)
    except (MissingKeyError, KeyRejectedError) as e:
        raise _fail(e) from None
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
def paths() -> None:
    """Show where tracedoc keeps its config file and local data."""
    typer.echo(f"Config file:    {user_config_path()}")
    typer.echo(f"Config folder:  {config_dir()}")
    typer.echo(f"Feedback votes: {feedback_path()}")


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
