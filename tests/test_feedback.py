from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tracedoc import cli, feedback

runner = CliRunner()


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "data" / "feedback.jsonl"
    monkeypatch.setattr(feedback, "feedback_path", lambda: path)
    return path


def test_record_and_latest_vote_wins(store: Path) -> None:
    feedback.record("abc", "up", "design.loop")
    feedback.record("abc", "down", "design.loop")
    feedback.record("def", "up")
    votes = feedback.load()
    assert votes["abc"]["vote"] == "down"
    assert votes["def"]["check_id"] is None


def test_import_export_format(store: Path, tmp_path: Path) -> None:
    export = tmp_path / "tracedoc-feedback.json"
    export.write_text(
        json.dumps(
            {
                "format": "tracedoc-feedback",
                "version": 1,
                "votes": [
                    {"finding_id": "a1", "check_id": "x.y", "vote": "up"},
                    {"finding_id": "a2", "check_id": "x.y", "vote": "sideways"},
                ],
            }
        )
    )
    assert feedback.import_file(export) == 1
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(ValueError):
        feedback.import_file(bad)


def test_cli_feedback_resolves_check_from_report(store: Path, tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"groups": [{"findings": [{"id": "f1", "check_id": "tools.unused_output"}]}]})
    )
    result = runner.invoke(cli.app, ["feedback", "f1", "up", "--report", str(report)])
    assert result.exit_code == 0, result.output
    assert feedback.load()["f1"]["check_id"] == "tools.unused_output"
    assert runner.invoke(cli.app, ["feedback", "f1", "maybe"]).exit_code == 2
