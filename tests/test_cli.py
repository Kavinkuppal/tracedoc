"""CLI: estimate, rules-only analyze, judged analyze with a fake model, and --max-cost."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.fakes import FakeLLM, always, verdict
from tests.helpers import TRACES
from tracedoc import cli

runner = CliRunner()
CAPTURES = str(TRACES / "capture")


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracedoc import config as config_module

    monkeypatch.setattr(config_module, "user_config_path", lambda: tmp_path / "none.toml")
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_version() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0 and "0.0.1" in result.output


def test_estimate_prints_cost_without_calls() -> None:
    result = runner.invoke(cli.app, ["estimate", CAPTURES, "--include-unvalidated"])
    assert result.exit_code == 0, result.output
    assert "Estimated cost: $" in result.output
    assert "Judge windows:" in result.output


def test_estimate_langsmith_export() -> None:
    result = runner.invoke(
        cli.app, ["estimate", str(TRACES / "langsmith"), "--source", "langsmith"]
    )
    assert result.exit_code == 0, result.output
    assert "Runs: 7" in result.output


def test_no_judge_runs_rules_only_without_a_key(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["analyze", CAPTURES, "--no-judge", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    assert (tmp_path / "summary.pdf").is_file() and (tmp_path / "details.pdf").is_file()
    assert not (tmp_path / "report.html").exists()  # HTML only with --html
    assert report["metadata"]["judged"] is False
    assert report["groups"][0]["representative"]["severity"] == "high"
    assert report["scorecard"]["runs"] == 7


def _with_fake(monkeypatch: pytest.MonkeyPatch, client: FakeLLM) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(cli, "make_client", lambda config: client)


def test_analyze_with_model_asks_then_judges(tmp_path: Path, monkeypatch) -> None:
    client = FakeLLM(
        triage=always(True), confirm=lambda p, c: verdict(c == "agents.ignored_output")
    )
    _with_fake(monkeypatch, client)
    declined = runner.invoke(cli.app, ["analyze", CAPTURES, "--out", str(tmp_path)], input="n\n")
    assert declined.exit_code == 1
    assert not client.calls

    args = ["analyze", CAPTURES, "--out", str(tmp_path), "--yes", "--include-unvalidated"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    meta = report["metadata"]
    assert meta["judged"] is True
    assert meta["analysis_cost"] > 0
    assert meta["analysis_models"] == {
        "triage": "claude-haiku-4-5-20251001",
        "confirm": "claude-sonnet-5-5",
    }
    checks = {g["check_id"] for g in report["groups"]}
    assert "agents.ignored_output" in checks


def test_analyze_max_cost_stops_early(tmp_path: Path, monkeypatch) -> None:
    _with_fake(monkeypatch, FakeLLM(triage=always(True), confirm=lambda p, c: verdict(True)))
    args = [
        "analyze",
        CAPTURES,
        "--out",
        str(tmp_path),
        "--yes",
        "--include-unvalidated",
        "--max-cost",
        "0.01",
    ]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["metadata"]["stopped_for_budget"]
    assert report["metadata"]["analysis_cost"] <= 0.01
    assert "Judging stopped early" in result.output


def test_model_flags_override_config(tmp_path: Path, monkeypatch) -> None:
    client = FakeLLM()
    _with_fake(monkeypatch, client)
    args = [
        "analyze",
        CAPTURES,
        "--out",
        str(tmp_path),
        "--yes",
        "--model",
        "claude-sonnet-5",
        "--triage-model",
        "claude-haiku-4-5",
    ]
    assert runner.invoke(cli.app, args).exit_code == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["metadata"]["analysis_models"]["confirm"] == "claude-sonnet-5"


# ---- missing, rejected, and misplaced keys: one line, exit 2, no traceback -----------------


def _one_line_error(result) -> str:
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output
    lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
    assert len(lines) == 1, result.stderr
    return lines[0]


def test_analyze_without_key_is_one_clear_line(tmp_path: Path) -> None:
    line = _one_line_error(runner.invoke(cli.app, ["analyze", CAPTURES, "--out", str(tmp_path)]))
    assert line.startswith("Error: ANTHROPIC_API_KEY is not set.")
    assert "README: 'Set up your API key'" in line


def test_analyze_missing_openai_key_names_openai_variable(tmp_path: Path) -> None:
    config = tmp_path / "c.toml"
    config.write_text('[llm]\nprovider = "openai"\ntriage_model = "a"\nconfirm_model = "b"\n')
    args = ["analyze", CAPTURES, "--out", str(tmp_path / "r"), "--config", str(config)]
    assert "OPENAI_API_KEY is not set" in _one_line_error(runner.invoke(cli.app, args))


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_estimate_needs_no_key(provider: str, tmp_path: Path) -> None:
    config = tmp_path / "c.toml"
    config.write_text(f'[llm]\nprovider = "{provider}"\n')
    result = runner.invoke(cli.app, ["estimate", CAPTURES, "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "Estimated cost:" in result.output


def test_key_in_config_file_is_refused(tmp_path: Path) -> None:
    config = tmp_path / "c.toml"
    config.write_text('[llm]\napi_key = "sk-should-not-be-here"\n')
    line = _one_line_error(runner.invoke(cli.app, ["estimate", CAPTURES, "--config", str(config)]))
    assert "api_key is not supported" in line  # a key in config is refused even by estimate
    assert "sk-should-not-be-here" not in line


def test_rejected_key_stops_with_one_line(tmp_path: Path, monkeypatch) -> None:
    from tracedoc.llm.client import KeyRejectedError

    class Rejecting(FakeLLM):
        def complete_json(self, *args, **kwargs):
            raise KeyRejectedError("ANTHROPIC_API_KEY", "HTTP 401")

    _with_fake(monkeypatch, Rejecting())
    args = ["analyze", CAPTURES, "--out", str(tmp_path), "--yes"]
    line = _one_line_error(runner.invoke(cli.app, args))
    assert "rejected ANTHROPIC_API_KEY (HTTP 401)" in line
    assert not (tmp_path / "report.json").exists()


def test_langsmith_project_needs_langsmith_key() -> None:
    args = ["analyze", "--source", "langsmith", "--project", "p", "--no-judge"]
    line = _one_line_error(runner.invoke(cli.app, args))
    assert "LANGSMITH_API_KEY is not set" in line


def test_missing_path_is_one_line() -> None:
    line = _one_line_error(runner.invoke(cli.app, ["analyze", "no/such/dir", "--no-judge"]))
    assert "does not exist" in line


def test_paths_command(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["paths"])
    assert result.exit_code == 0
    assert "Config file:" in result.output and "Feedback votes:" in result.output


def test_estimate_reports_judge_caps(tmp_path: Path) -> None:
    config = tmp_path / "caps.toml"
    config.write_text("[judge]\nmax_windows_per_signature = 1\nsample_runs = 2\n")
    result = runner.invoke(
        cli.app, ["estimate", CAPTURES, "--include-unvalidated", "--config", str(config)]
    )
    assert result.exit_code == 0, result.output
    assert "AI judge caps:" in result.output
    assert "run(s) were not sampled for whole-run checks (sample of 2)" in result.output
