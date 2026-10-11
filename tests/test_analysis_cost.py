"""The analysis cost in the report header says when it includes cached AI judge answers: they
were paid for once, when first made, and the figure stays the same on a rerun."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from pypdf import PdfReader
from typer.testing import CliRunner

from tests.fakes import FakeLLM, always, verdict
from tests.helpers import TRACES
from tracedoc import cli
from tracedoc.report.view import _analysis_cost

runner = CliRunner()
CACHED = "(includes cached results, paid once)"


def _header(pdf: Path) -> str:
    """The analysis cost on page 1: the header text up to "AI judge answers", and the label
    after it if there is one."""
    page1 = " ".join((PdfReader(pdf).pages[0].extract_text() or "").split())
    found = re.search(
        r"Analysis cost (\$[\d.,]+ .*?AI judge answers\)?(?: \(includes[^)]*\))?)", page1
    )
    assert found, page1
    return found[1]


def _analyze(traces: Path, out: Path) -> dict:
    args = ["analyze", str(traces), "--out", str(out), "--yes", "--include-unvalidated", "--html"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    return json.loads((out / "report.json").read_text())["metadata"]


def test_rerun_labels_the_cost_as_including_cached_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    fake = FakeLLM(triage=always(True), confirm=lambda p, c: verdict(True))
    monkeypatch.setattr(cli, "make_client", lambda config: fake)
    traces = tmp_path / "traces"
    traces.mkdir()
    shutil.copy(TRACES / "capture" / "tool_output_unused.jsonl", traces)

    first = _analyze(traces, tmp_path / "first")
    assert first["analysis_calls"] > 0 and first["analysis_answers_reused"] == 0
    header = _header(tmp_path / "first" / "summary.pdf")
    assert re.fullmatch(r"\$\d+\.\d+ \(\d+ AI judge answers\)", header), header
    assert CACHED not in header

    calls = len(fake.calls)
    again = _analyze(traces, tmp_path / "again")
    assert len(fake.calls) == calls  # every answer came from the cache
    assert again["analysis_calls"] == 0 and again["analysis_answers_reused"] > 0
    assert again["analysis_cost"] == first["analysis_cost"]  # the same figure, paid once
    cost = f"${first['analysis_cost']:.4f}" if first["analysis_cost"] < 0.1 else None
    for pdf in ("summary.pdf", "details.pdf"):
        header = _header(tmp_path / "again" / pdf)
        assert header.endswith(f"AI judge answers {CACHED}"), (pdf, header)
        assert cost is None or header.startswith(cost), (pdf, header)
    assert f"{CACHED}." in (tmp_path / "again" / "report.md").read_text()
    assert CACHED in (tmp_path / "again" / "report.html").read_text()


def test_label_only_when_answers_were_reused() -> None:
    meta = {"judged": True, "analysis_cost": 0.4366, "analysis_calls": 126}
    assert _analysis_cost({**meta, "analysis_answers_reused": 0}) == "$0.44 (126 AI judge answers)"
    mixed = _analysis_cost({**meta, "analysis_answers_reused": 50})
    assert " ".join(mixed.split()) == f"$0.44 for 176 AI judge answers {CACHED}"
    assert _analysis_cost({"judged": False}).startswith("$0.00 (rule checks only")
