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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


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
    assert "Runs: 6" in result.output


def test_analyze_without_key_runs_rules_only(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["analyze", CAPTURES, "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "running rule checks only" in result.output
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["metadata"]["judged"] is False
    assert report["groups"][0]["representative"]["severity"] == "high"
    assert report["scorecard"]["runs"] == 6


def _with_fake(monkeypatch: pytest.MonkeyPatch, client: FakeLLM) -> None:
    monkeypatch.setattr(cli, "has_credentials", lambda config: True)
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
    assert meta["analysis_models"] == {"triage": "claude-haiku-4-5", "confirm": "claude-opus-5"}
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
