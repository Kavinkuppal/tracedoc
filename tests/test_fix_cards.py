"""Page 1 fix cards: only serious or moderate problems, cost problems only when material, and
never padded to three."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from tests.test_pdf import _finding, _report, _runs
from tracedoc.report import write_reports
from tracedoc.report.view import build_view
from tracedoc.schema import Impact

AGENTS = ["supervisor", "research_expert"]


def _cost_finding(run, cost: float, severity: str = "medium"):
    f = _finding(
        run,
        "cost.token_hotspot",
        severity,
        judge=False,
        agent="research_expert",
        text="research_expert used 80% of the run's tokens.",
    )
    f.impact = Impact(tokens=5000, cost=cost)
    return f


def test_minor_findings_never_become_fix_cards(tmp_path: Path) -> None:
    runs = _runs(10, AGENTS)
    minor = [
        _finding(r, "cost.latency_outlier", "low", judge=False, agent="supervisor", text="slow.")
        for r in runs[:6]
    ]
    view = build_view(_report(runs, minor), runs)
    assert view.cards == []
    assert [r.problem for r in view.other_rows] == ["Step was much slower than usual"]
    files = write_reports(_report(runs, minor), runs, tmp_path)
    page1 = PdfReader(files.summary).pages[0].extract_text()
    assert "No serious or moderate problem" in " ".join(page1.split())


def test_cost_findings_need_ten_percent_of_the_system_cost() -> None:
    runs = _runs(10, AGENTS)  # 0.0045 per run: 0.045 in total
    system_cost = sum(r.totals.cost for r in runs)  # type: ignore[misc]
    small = [_cost_finding(r, cost=0.1 * system_cost / 4 - 1e-9) for r in runs[:4]]
    view = build_view(_report(runs, small), runs)
    assert view.cards == []  # moderate, but under 10% of the system's cost
    assert view.other_rows[0].problem == "One agent used most of the tokens"

    material = [_cost_finding(r, cost=0.1 * system_cost / 4) for r in runs[:4]]
    view = build_view(_report(runs, material), runs)
    assert [c.title for c in view.cards] == ["One agent used most of the tokens"]


def test_fewer_qualifying_problems_mean_fewer_cards_never_padding(tmp_path: Path) -> None:
    runs = _runs(10, AGENTS)
    findings = [
        _finding(
            runs[0],
            "design.empty_answer",
            "high",
            judge=False,
            agent="supervisor",
            text="supervisor's final reply had no text.",
        ),
        *[
            _finding(
                r, "cost.latency_outlier", "low", judge=False, agent="supervisor", text="slow."
            )
            for r in runs[1:5]
        ],
    ]
    report = _report(runs, findings)
    view = build_view(report, runs)
    assert [c.title for c in view.cards] == ["Final answer was empty"]
    assert view.fewer_fixes
    assert [r.problem for r in view.other_rows] == ["Step was much slower than usual"]
    files = write_reports(report, runs, tmp_path)
    page1 = " ".join(PdfReader(files.summary).pages[0].extract_text().split())
    assert "1. Final answer was empty" in page1
    assert "2." not in page1.split("Top three fixes", 1)[1].split("Only serious", 1)[0]
    assert "Only serious or moderate problems are listed here" in page1


def test_nothing_to_highlight_when_nothing_qualifies(tmp_path: Path) -> None:
    """What's working: no clean handoff or efficient path qualifies in these runs (the senders
    wrote almost nothing of their own, and every run took the same number of steps)."""
    runs = _runs(10, AGENTS)
    report = _report(runs, [])
    assert build_view(report, runs).working == []
    files = write_reports(report, runs, tmp_path)
    page2 = " ".join(PdfReader(files.summary).pages[1].extract_text().split())
    assert "What's working Nothing to highlight yet." in page2
