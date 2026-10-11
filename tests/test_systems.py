"""Traces from several systems (different top-level graph names) get one report each, and
every percentage in a report is out of that system's runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader
from typer.testing import CliRunner

from tests.helpers import TRACES
from tracedoc import cli
from tracedoc.analyze import split_systems, system_folders
from tracedoc.ingest.jsonl import read_capture_file

runner = CliRunner()


def _copy(fixture: str, dest: Path, n: int) -> None:
    """`n` copies of a fixture capture, each a distinct run (new run ID)."""
    text = (TRACES / "capture" / f"{fixture}.jsonl").read_text()
    run_id = json.loads(text.splitlines()[0])["run_id"]
    for i in range(n):
        new_id = f"{run_id[:-4]}{i:04d}"
        (dest / f"{fixture}-{i}.jsonl").write_text(text.replace(run_id, new_id))


def _page_text(pdf: Path, page: int = 0) -> str:
    return " ".join((PdfReader(pdf).pages[page].extract_text() or "").split())


def test_two_systems_get_separate_reports_out_of_their_own_runs(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    _copy("supervisor_no_end", traces, 3)  # 3 runs, every one with a serious problem
    _copy("healthy", traces, 1)
    out = tmp_path / "out"
    result = runner.invoke(cli.app, ["analyze", str(traces), "--no-judge", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert not (out / "summary.pdf").exists()
    assert sorted(p.name for p in out.iterdir()) == ["healthy", "supervisor_no_end"]

    sup, healthy = out / "supervisor_no_end", out / "healthy"
    for folder in (sup, healthy):
        assert {p.name for p in folder.iterdir()} == {"summary.pdf", "details.pdf", "report.json"}
    assert json.loads((sup / "report.json").read_text())["scorecard"]["runs"] == 3
    assert json.loads((healthy / "report.json").read_text())["scorecard"]["runs"] == 1

    page1 = _page_text(sup / "summary.pdf")
    assert "supervisor_no_end" in page1 and "3 runs" in page1
    assert "3 of 3 runs had a serious problem." in page1
    assert "3 (100%)" in page1
    assert not re.search(r"\bof 4 runs\b", page1)  # never out of all systems' runs
    for group in json.loads((sup / "report.json").read_text())["groups"]:
        assert group["frequency"] == group["runs_affected"] / 3

    page1 = _page_text(healthy / "summary.pdf")
    assert "The run had no serious problem." in page1
    assert "0 (0%)" in page1
    assert "=== healthy ===" in result.output and "=== supervisor_no_end ===" in result.output


def test_one_system_writes_straight_into_the_output_folder(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    _copy("healthy", traces, 2)
    out = tmp_path / "out"
    result = runner.invoke(cli.app, ["analyze", str(traces), "--no-judge", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert {p.name for p in out.iterdir()} == {"summary.pdf", "details.pdf", "report.json"}


def test_estimate_is_per_system_with_a_total() -> None:
    result = runner.invoke(cli.app, ["estimate", str(TRACES / "capture")])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("Read 7 run(s)")
    assert "7 systems, each analyzed and reported separately" in result.output
    assert "--- healthy ---" in result.output
    assert "All 7 systems: Runs: 7" in result.output


def test_systems_split_by_top_level_graph_name() -> None:
    traces = [read_capture_file(p) for p in sorted((TRACES / "capture").glob("*.jsonl"))]
    systems = split_systems(traces)
    assert list(systems) == sorted(systems)
    assert sum(len(t) for t in systems.values()) == len(traces)
    folders = system_folders(["my app", "my/app", "", "LangGraph"])
    assert folders == {
        "my app": "my_app",
        "my/app": "my_app-2",
        "": "unnamed_system",
        "LangGraph": "LangGraph",
    }
