"""details.pdf, section 2: each problem shows its explanation and fix once, then one worked
example with the full evidence, then one row per affected run; every finding ID still points to
the page it is on."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from reportlab.pdfbase.pdfmetrics import stringWidth

from tests.test_pdf import WORST_CASES, _case, _finding, _report, _runs, _text
from tracedoc.report import write_reports
from tracedoc.report.pdf.base import REGULAR, fit_end

AGENTS = ["supervisor", "research_expert"]
IGNORED = "supervisor's answer did not use what search_flights returned."


def _section2(pdf: Path) -> str:
    text = "\n".join(_text(pdf))
    return text.split("2. Every finding", 1)[1].split("3. Run index", 1)[0]


def _ignored(runs):
    return [
        _finding(r, "tools.unused_output", "medium", judge=False, agent="supervisor", text=IGNORED)
        for r in runs
    ]


def test_a_problem_in_many_runs_is_explained_once(tmp_path: Path) -> None:
    runs = _runs(24, AGENTS)
    files = write_reports(_report(runs, _ignored(runs[:15])), runs, tmp_path)
    (problem,) = files.view.main_problems
    section = _section2(files.details)
    flat = " ".join(section.split())

    assert flat.count(" ".join(problem.what_to_change.split())) == 1
    for once in ("What it looks for", "Why it matters", "What to change", "Worked example:"):
        assert section.count(once) == 1, once
    # the worked example carries the complete evidence, once
    for evidence in ("Agent and step", "Tool calls and results", "Agent's output", "Finding ID"):
        assert section.count(evidence) == 1 + (evidence == "Finding ID"), evidence  # + table head
    assert "Agent's output\nHere is what I found." in section
    assert "Every affected run (15 of 24 runs)" in flat

    # one row per affected run: run, request, what happened, finding ID
    assert len(problem.affected) == 15
    for row in problem.affected:
        (fid,) = row.finding_ids
        assert re.search(rf"^{row.run_id}\n.*\n.*\n{fid}$", section, re.MULTILINE), row
        assert row.what_happened == IGNORED
    runs_in_order = [int(r.run_id[1:]) for r in problem.affected]
    assert runs_in_order == sorted(runs_in_order)


def test_several_findings_in_one_run_share_its_row(tmp_path: Path) -> None:
    runs = _runs(6, AGENTS)
    findings = _ignored(runs[:3])
    second = _ignored(runs[:1])[0]
    tool = next(s for s in runs[0].steps if s.kind == "tool" and s.node == "supervisor")
    second.evidence_step_ids = [tool.id]  # same problem, another step of the same run
    files = write_reports(_report(runs, [*findings, second]), runs, tmp_path)
    (problem,) = files.view.main_problems
    assert len(problem.findings) == 4 and len(problem.affected) == 3
    shared = problem.affected[[r.run_id for r in problem.affected].index("R1")]
    assert len(shared.finding_ids) == 2
    section = _section2(files.details)
    assert f"\n{', '.join(shared.finding_ids)}" in section
    for fid in shared.finding_ids:
        assert files.view.finding_pages[fid] >= 1


def test_ai_judge_changes_that_differ_are_listed_once_each(tmp_path: Path) -> None:
    runs = _runs(8, AGENTS)
    findings = [
        _finding(
            r,
            "verification.premature_finish",
            "high",
            judge=True,
            agent="supervisor",
            text=f"supervisor stopped before booking ({i % 2}).",
        )
        for i, r in enumerate(runs[:6])
    ]
    files = write_reports(_report(runs, findings), runs, tmp_path)
    (problem,) = files.view.main_problems
    (other,) = problem.other_changes
    assert other.text not in problem.what_to_change
    assert len(other.finding_ids) == 3
    flat = " ".join(_section2(files.details).split())
    assert flat.count("Other suggested changes") == 1
    assert flat.count(" ".join(other.text.split())) == 1


@pytest.mark.parametrize("case", [*WORST_CASES, "many_runs_one_problem"])
def test_every_finding_id_points_to_its_page(case: str, tmp_path: Path) -> None:
    if case == "many_runs_one_problem":
        runs = _runs(120, AGENTS)
        report = _report(runs, _ignored(runs[:110]))
    else:
        report, runs = _case(case)
    files = write_reports(report, runs, tmp_path)
    pages = _text(files.details)
    assert set(files.view.finding_pages) == {f.id for f in files.view.findings}
    for fid, page in files.view.finding_pages.items():
        assert re.search(rf"\b{fid}\b", pages[page - 1]), (case, fid, page)
    for problem in [*files.view.main_problems, *files.view.low_problems]:
        example = problem.representative.id
        assert f"Worked example: {example} · " in pages[files.view.finding_pages[example] - 1]


def test_repeated_blocks_are_gone(tmp_path: Path) -> None:
    """A problem in 110 runs takes a few pages (about 45 runs a page), not one block per run."""
    runs = _runs(120, AGENTS)
    files = write_reports(_report(runs, _ignored(runs[:110])), runs, tmp_path)
    section = _section2(files.details)
    assert section.count("Worked example:") == 1
    assert section.count("Tool calls and results") == 1
    pages = _text(files.details)
    first = next(i for i, p in enumerate(pages) if "2. Every finding" in p)
    last = next(i for i, p in enumerate(pages) if "3. Run index" in p)
    assert last - first <= 4


def test_table_cells_fit_on_one_line() -> None:
    long = "Please plan my whole trip, with flights, hotels and a car. " * 20
    for width in (40.0, 150.0, 260.0):
        cut = fit_end(long, REGULAR, 9, width)
        assert cut.endswith("…") and stringWidth(cut, REGULAR, 9) <= width
        assert long.startswith(cut[:-1].rstrip(" ,;:."))
    assert fit_end("short", REGULAR, 9, 100) == "short"
    assert stringWidth(fit_end("x" * 300, REGULAR, 9, 50), REGULAR, 9) <= 50
