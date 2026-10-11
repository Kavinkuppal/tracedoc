"""Command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from .__about__ import NAME, __version__
from .analyze import SourceName, classifier_calls, collect_traces, normalize_all, prepare
from .analyze import analyze as run_analysis
from .config import Config, ConfigError, config_dir, load_config, user_config_path
from .feedback import check_for, feedback_path, import_file, record
from .feedback import load as load_votes
from .ingest.jsonl import CaptureFormatError
from .llm.client import (
    ConfirmModelUnavailableError,
    KeyRejectedError,
    MissingKeyError,
    ModelUnavailableError,
    make_client,
    require_key,
    with_model_fallback,
)
from .llm.cost import Spend, price_of
from .report import write_reports
from .schema import Report
from .share.consent import (
    TERMS,
    load_state,
    record_choice,
    rotate_install_id,
    save_state,
    state_path,
)
from .share.payload import build_contribution, serialize
from .share.upload import (
    ShareError,
    last_payload_path,
    request_deletion,
    save_pending,
    upload,
    upload_log_path,
)

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


def _load(paths: list[Path] | None, source: SourceName, project: str | None) -> list:
    for path in paths or []:
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
    loaded = collect_traces(paths, source, project)
    if not loaded.traces:
        where = f"LangSmith project {project}" if project else ", ".join(map(str, paths or []))
        raise ValueError(f"no traces found in {where}")
    typer.echo(loaded.describe())
    return loaded.traces


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
    path: list[Path] | None = typer.Argument(
        None, help="Folders or files with traces; sub-folders are searched."
    ),
    source: SourceName = typer.Option("capture", help="capture or langsmith"),
    project: str | None = typer.Option(None, help="LangSmith project to pull from."),
    model: str | None = typer.Option(None, help="Confirmation model."),
    triage_model: str | None = typer.Option(None, help="Triage model."),
    include_unvalidated: bool = typer.Option(
        False, help="Include judge checks below the precision bar."
    ),
    config_path: Path | None = typer.Option(None, "--config", help="Extra config TOML."),
) -> None:
    """Print what analysis would cost, without calling any model. Needs no API key."""
    try:
        config = _config(config_path, model, triage_model, include_unvalidated)
        runs = normalize_all(_load(path, source, project), config)  # no model calls, no key
    except USER_ERRORS as e:
        raise _fail(e) from None
    _print_estimate(config, runs, prepare(runs, config))


@app.command()
def analyze(
    path: list[Path] | None = typer.Argument(
        None, help="Folders or files with traces; sub-folders are searched."
    ),
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

    progress_open = [False]  # a "Judging n/m" line is on screen without its newline

    def progress(done: int, total: int) -> None:
        progress_open[0] = done != total
        typer.echo(f"\rJudging {done}/{total}", nl=done == total)

    def fail(error: BaseException) -> typer.Exit:
        if progress_open[0]:
            typer.echo("")  # end the progress line so the error starts on its own line
        return _fail(error)

    try:
        if use_model:
            client = with_model_fallback(make_client(config.llm), config.llm)
            spend = Spend(prices=config.prices, max_cost=max_cost)
            if classifier_calls(runs):
                runs = normalize_all(traces, config, client=client, spend=spend)
        report = run_analysis(runs, config, client=client, spend=spend, progress=progress)
    except (MissingKeyError, KeyRejectedError, ConfirmModelUnavailableError) as e:
        raise fail(e) from None
    except ModelUnavailableError as e:  # a model that is neither triage nor confirmation
        raise fail(ValueError(f"{e}. Check the model names under [llm] in your config.")) from None
    html_path = write_reports(report, runs, out)
    _print_summary(report, out, html_path)
    if client is not None and client.fallback is not None:
        typer.echo(client.fallback.notice())
    _share_after_report(config, report, runs)


def _interactive() -> bool:
    """Only ask for consent at a real terminal; scripts and CI never get (or answer) a prompt."""
    return sys.stdin.isatty()


def _share_after_report(config, report: Report, runs: list) -> None:  # noqa: ANN001
    """Save the would-be payload for `share --preview`; ask and upload only when enabled."""
    state = load_state()  # creates the random install ID on first run
    votes = [
        (v["check_id"], v["vote"], v["ts"])
        for v in load_votes().values()
        if v.get("check_id")
        and (state.votes_sent_until is None or v["ts"] > state.votes_sent_until)
    ]
    findings = [f for g in report.groups for f in g.findings]
    payload = serialize(
        build_contribution(
            state.install_id,
            state.deletion_key_hash,
            runs,
            findings,
            [(c, v) for c, v, _ in votes],
        )
    )
    save_pending(payload)
    if not config.share.enabled:
        return
    if state.needs_asking and _interactive():
        typer.echo("\n" + TERMS)
        agreed = typer.confirm("Share anonymized statistics?", default=False)  # Enter means no
        state = record_choice(state, "yes" if agreed else "no")
    if not state.sharing or not config.share.endpoint.strip():
        return
    try:
        upload(payload, config.share)
    except ShareError as e:
        typer.echo(f"Statistics were not shared: {e}", err=True)
        return
    if votes:
        state.votes_sent_until = max(ts for _, _, ts in votes)
        save_state(state)
    typer.echo("Shared anonymized statistics (see exactly what: tracedoc share --preview).")


@app.command()
def share(
    preview: bool = typer.Option(False, "--preview", help="Print the exact payload."),
    on: bool = typer.Option(False, "--on", help="Opt in to sharing."),
    off: bool = typer.Option(False, "--off", help="Stop sharing (revoke consent)."),
    delete: bool = typer.Option(
        False, "--delete", help="Delete everything shared from this install."
    ),
    config_path: Path | None = typer.Option(None, "--config", help="Extra config TOML."),
) -> None:
    """Preview, opt in to, revoke, or delete anonymized statistics sharing."""
    if sum((preview, on, off, delete)) != 1:
        raise _fail(ValueError("use exactly one of --preview, --on, --off, --delete"))
    try:
        config = load_config(config_path)
    except USER_ERRORS as e:
        raise _fail(e) from None
    state = load_state()
    if preview:
        path = last_payload_path()
        if not path.is_file():
            raise _fail(ValueError("nothing to preview yet; run `tracedoc analyze` first"))
        sys.stdout.buffer.write(path.read_bytes())  # the exact bytes an upload sends
        sys.stdout.buffer.flush()
        return
    if off:
        record_choice(state, "no")
        typer.echo("Sharing is off. Nothing more will be sent.")
        return
    if not config.share.enabled:
        raise _fail(ShareError("data sharing is not enabled in this version"))
    if on:
        typer.echo(TERMS)
        if typer.confirm("Share anonymized statistics?", default=False):
            record_choice(state, "yes")
            typer.echo("Sharing is on. Preview with `tracedoc share --preview`.")
        else:
            record_choice(state, "no")
            typer.echo("Sharing stays off.")
        return
    try:
        request_deletion(state.install_id, state.deletion_token, config.share)
    except ShareError as e:
        raise _fail(e) from None
    record_choice(state, "no")
    rotate_install_id(state)
    typer.echo("Deletion requested and confirmed. Sharing is off; this install has a new ID.")


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
    typer.echo(f"Share state:    {state_path()}")
    typer.echo(f"Upload log:     {upload_log_path()}")


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
