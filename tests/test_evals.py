"""The eval harness runs offline with mocked responses, and mock results never enable checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evals.run_evals import LABELS, checks_to_run, load_examples, main

from tracedoc.checks.judges.prompts import enabled_judges
from tracedoc.config import Config


def test_every_check_has_balanced_labels() -> None:
    for key in checks_to_run(None):
        examples = load_examples(key)
        positives = sum(e["label"] for e in examples)
        assert positives >= 2, key
        assert len(examples) - positives >= 2, key
        assert len({e["id"] for e in examples}) == len(examples)
        for e in examples:
            assert e["window"].startswith("USER TASK:")
            assert "[S1]" in e["window"]


def test_no_orphan_label_files() -> None:
    keys = set(checks_to_run(None))
    assert {p.stem for p in LABELS.glob("*.jsonl")} == keys


def test_mock_run_writes_scores_that_do_not_enable_checks(tmp_path: Path, capsys) -> None:
    out = tmp_path / "precision.json"
    assert main(["--mock", "--write", "--precision-file", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "agents.ignored_output" in printed
    data = json.loads(out.read_text())
    assert set(data) == set(checks_to_run(None))
    assert all(v["mock"] and v["precision"] == 1.0 for v in data.values())
    enabled, disabled = enabled_judges(Config(), precision=data)
    assert not enabled
    assert len(disabled) == 8


@pytest.mark.parametrize("check", ["agents.task_drift", "confirm.design.loop"])
def test_single_check(check: str, capsys) -> None:
    assert main(["--mock", "--check", check]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(check)]
    assert len(lines) == 1
