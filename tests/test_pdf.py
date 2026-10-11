"""summary.pdf and details.pdf: two pages at most, same bytes every time, limits, references.

Synthetic worst cases are built as runs and findings and go through the whole pipeline
(view model and renderers), the way `tracedoc analyze` writes them.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pypdf import PdfReader

from tests.fixtures.agents import AGENTS
from tests.helpers import load_fixture
from tracedoc.analyze import analyze
from tracedoc.catalog import catalog, limit
from tracedoc.checks.signals import find_signals
from tracedoc.config import Config
from tracedoc.group import group_findings
from tracedoc.report import write_reports
from tracedoc.report.pdf.base import MIN_FONT_SIZE, Styles, safe
from tracedoc.report.view import build_view, confidence_level
from tracedoc.schema import Finding, Report, Run, Step, Totals

T0 = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
RULE_CHECKS = [
    ("design.empty_answer", "high"),
    ("design.loop", "medium"),
    ("tools.errors_retries", "medium"),
    ("tools.unused_output", "medium"),
    ("cost.token_hotspot", "low"),
    ("cost.latency_outlier", "low"),
]
JUDGE_CHECKS = [
    ("verification.premature_finish", "high"),
    ("design.ignored_instructions", "medium"),
    ("agents.incomplete_handoff", "medium"),
    ("agents.task_drift", "medium"),
]
CHECK_ID = re.compile(r"\b[a-z]+\.[a-z_]+\b")


# ---- synthetic data ----------------------------------------------------------------------------


def _runs(
    n: int,
    agents: list[str],
    *,
    request: str = "Find the cheapest flight from Boston to New York tomorrow.",
    models: list[str] | None = None,
    chain: bool = False,
    priced: bool = True,
) -> list[Run]:
    models = models or ["claude-haiku-4-5-20251001"]
    runs = []
    for i in range(n):
        start = T0 + timedelta(minutes=i)
        t, steps = start, []
        used = (
            agents if chain else [agents[(i + k) % len(agents)] for k in range(min(3, len(agents)))]
        )
        for k, agent in enumerate(used):
            if k:
                steps.append(
                    Step(
                        id=f"r{i}-h{k}",
                        run_id=f"run-{i}",
                        kind="handoff",
                        name="handoff",
                        handoff_from=used[k - 1],
                        handoff_to=agent,
                        node=agent,
                        started_at=t,
                        ended_at=t,
                        input="the plan so far",
                    )
                )
            steps.append(
                Step(
                    id=f"r{i}-n{k}",
                    run_id=f"run-{i}",
                    kind="node",
                    name=agent,
                    node=agent,
                    started_at=t,
                    ended_at=t + timedelta(seconds=2),
                )
            )
            steps.append(
                Step(
                    id=f"r{i}-l{k}",
                    run_id=f"run-{i}",
                    parent_id=f"r{i}-n{k}",
                    kind="llm",
                    name="model",
                    node=agent,
                    model=models[(i + k) % len(models)],
                    started_at=t,
                    ended_at=t + timedelta(seconds=1),
                    tokens_in=900,
                    tokens_out=120,
                    cost=0.0015 if priced else None,
                    output="Here is what I found.",
                )
            )
            steps.append(
                Step(
                    id=f"r{i}-t{k}",
                    run_id=f"run-{i}",
                    parent_id=f"r{i}-n{k}",
                    kind="tool",
                    name="search_flights",
                    node=agent,
                    started_at=t + timedelta(seconds=1),
                    ended_at=t + timedelta(seconds=2),
                    input={"from": "BOS", "to": "JFK"},
                    output='[{"airline": "Jet Blue", "id": "1"}]',
                )
            )
            t += timedelta(seconds=2)
        runs.append(
            Run(
                id=f"run-{i}",
                source="capture",
                name="travel_desk",
                started_at=start,
                ended_at=t,
                status="completed" if i % 4 else "no_answer",
                input_preview=request,
                final_answer="Booked." if i % 4 else "",
                steps=steps,
                totals=Totals(
                    tokens_in=900 * len(used),
                    tokens_out=120 * len(used),
                    cost=0.0015 * len(used) if priced else None,
                    duration_s=(t - start).total_seconds(),
                    step_count=2 * len(used),
                ),
            )
        )
    return runs


def _finding(
    run: Run, check: str, severity: str, *, judge: bool, agent: str, text: str, key: str = ""
) -> Finding:
    llm = (
        next(s for s in run.steps if s.kind == "llm" and s.node == agent)
        if any(s.kind == "llm" and s.node == agent for s in run.steps)
        else run.steps_of("llm")[0]
    )
    return Finding(
        check_id=check,
        group="design",
        method="judge" if judge else "rule",
        severity=severity,  # type: ignore[arg-type]
        run_id=run.id,
        key=key or agent,
        node=agent,
        evidence_step_ids=[llm.id],
        explanation=text,
        suggested_fix=f'Add to {agent}\'s prompt: "{text}"',
        confidence=0.9 if judge else 1.0,
        evidence_quote="Here is what I found." if judge else "",
        change_type="prompt instruction" if judge else None,
        facts={"agent": agent},
    )


def _report(runs: list[Run], findings: list[Finding], judged: bool = True) -> Report:
    return Report(
        groups=group_findings(findings, runs),
        signals=find_signals(runs),
        metadata={
            "judged": judged,
            "analysis_cost": 0.4366,
            "analysis_calls": 126,
            "analysis_answers_reused": 0,
            "analysis_unpriced_models": [],
            "unreviewed_by_judge": {
                "rule_candidates": {"design.loop": 7},
                "handoffs": 3,
                "unsampled_runs": 28,
            },
        },
    )


def _many_findings(runs: list[Run], agents: list[str], per_check: int, text: str) -> list[Finding]:
    out = []
    checks = [(c, s, False) for c, s in RULE_CHECKS] + [(c, s, True) for c, s in JUDGE_CHECKS]
    for c_i, (check, severity, judge) in enumerate(checks):
        for k in range(per_check):
            run = runs[(c_i * 7 + k) % len(runs)]
            agent = agents[(c_i + k) % len(agents)]
            out.append(
                _finding(run, check, severity, judge=judge, agent=agent, text=text, key=f"k{k % 3}")
            )
    return out


def _case(name: str) -> tuple[Report, list[Run]]:
    agents = ["supervisor", "research_expert", "math_expert"]
    sentence = (
        "the agent answered before the tool result came back and stated a number that no "
        "tool returned."
    )
    if name == "50_plus_findings":
        runs = _runs(40, agents)
        return _report(runs, _many_findings(runs, agents, 6, sentence)), runs
    if name == "20_agents":
        agents = [f"agent_{i:02d}" for i in range(20)]
        runs = _runs(40, agents)
        return _report(runs, _many_findings(runs, agents, 3, sentence)), runs
    if name == "long_agent_names":
        agents = [f"{'very_long_agent_name_' * 6}{i}" for i in range(6)]
        runs = _runs(30, agents)
        return _report(runs, _many_findings(runs, agents, 2, sentence)), runs
    if name == "long_requests":
        runs = _runs(30, agents, request="Please plan my whole trip. " * 400)
        return _report(runs, _many_findings(runs, agents, 2, sentence)), runs
    if name == "zero_findings":
        runs = _runs(10, agents)
        return _report(runs, [], judged=False), runs
    if name == "slots_at_their_limits":
        runs = _runs(20, agents)
        long_text = ("word " * 80)[: limit("what_happened")]
        return _report(runs, _many_findings(runs, agents, 2, long_text)), runs
    if name == "one_run":
        runs = _runs(1, agents)
        return _report(runs, _many_findings(runs, agents, 1, sentence)), runs
    if name == "many_models_unpriced":
        runs = _runs(25, agents, models=[f"model-{i}" for i in range(9)], priced=False)
        return _report(runs, _many_findings(runs, agents, 2, sentence)), runs
    if name == "chain_of_12_agents":
        agents = [f"step_{i}" for i in range(12)]
        runs = _runs(8, agents, chain=True)
        return _report(runs, _many_findings(runs, agents, 2, sentence)), runs
    if name == "non_latin_text":
        agents = ["研究员", "Ассистент", "agent 🤖"]
        runs = _runs(12, agents, request="帮我订一张从波士顿到纽约的机票 ✈️ " * 20)
        return _report(runs, _many_findings(runs, agents, 2, "答案是空的 😀 " * 10)), runs
    if name == "long_system_name":
        runs = _runs(12, agents)
        for r in runs:
            r.name = "an_extremely_long_compiled_graph_name_" * 8
        return _report(runs, _many_findings(runs, agents, 2, sentence)), runs
    raise ValueError(name)


WORST_CASES = [
    "50_plus_findings",
    "20_agents",
    "long_agent_names",
    "long_requests",
    "zero_findings",
    "slots_at_their_limits",
    "one_run",
    "many_models_unpriced",
    "chain_of_12_agents",
    "non_latin_text",
    "long_system_name",
]


def _text(pdf: Path) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(pdf).pages]


# ---- tests ------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", WORST_CASES)
def test_summary_is_two_pages_in_worst_cases(case: str, tmp_path: Path) -> None:
    report, runs = _case(case)
    files = write_reports(report, runs, tmp_path)
    summary = _text(files.summary)
    assert len(summary) == 2, case
    assert "What to fix" in summary[0] and "Where and how often" in summary[1]
    assert "How to read this report: details.pdf, section 1." in summary[1]
    assert len(PdfReader(files.details).pages) >= 1


def test_the_worst_case_set_covers_what_was_asked() -> None:
    assert len(WORST_CASES) >= 10
    report, _ = _case("50_plus_findings")
    assert sum(len(g.findings) for g in report.groups) >= 50
    _, runs = _case("20_agents")
    assert len({s.node for r in runs for s in r.steps_of("llm")}) == 20


def test_rows_are_dropped_not_shrunk_and_point_to_details(tmp_path: Path) -> None:
    report, runs = _case("20_agents")
    files = write_reports(report, runs, tmp_path)
    page2 = _text(files.summary)[1]
    assert re.search(r"\d+ more agents in details\.pdf, section 3\.", page2)
    assert re.search(r"\d+ more in details\.pdf", page2)


def test_rendering_twice_gives_identical_bytes(tmp_path: Path) -> None:
    report, runs = _case("50_plus_findings")
    a = write_reports(report.model_copy(deep=True), runs, tmp_path / "a")
    b = write_reports(report.model_copy(deep=True), runs, tmp_path / "b")
    assert a.summary.read_bytes() == b.summary.read_bytes()
    assert a.details.read_bytes() == b.details.read_bytes()


_RENDER = """
import hashlib, sys
from pathlib import Path
from tests.fixtures.agents import AGENTS
from tests.helpers import load_fixture
from tracedoc.analyze import analyze
from tracedoc.config import Config
from tracedoc.report import write_reports
runs = [load_fixture(n) for n in AGENTS]
files = write_reports(analyze(runs, Config()), runs, Path(sys.argv[1]))
print(hashlib.sha256(files.summary.read_bytes() + files.details.read_bytes()).hexdigest())
"""


def test_identical_bytes_across_processes_and_hash_seeds(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    digests = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "TZ": "Asia/Tokyo" if seed == "2" else "UTC"}
        env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src")])
        out = subprocess.run(
            [sys.executable, "-c", _RENDER, str(tmp_path / seed)],
            capture_output=True,
            text=True,
            env=env,
            cwd=root,
            check=True,
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1


def test_pdf_metadata_comes_from_the_traces(tmp_path: Path) -> None:
    report, runs = _case("one_run")
    files = write_reports(report, runs, tmp_path)
    info = PdfReader(files.summary).metadata
    end = max(r.ended_at for r in runs)  # type: ignore[type-var]
    assert info["/CreationDate"].startswith(end.strftime("D:%Y%m%d%H%M%S"))
    assert info["/Author"] == "tracedoc"


def test_only_the_bundled_fonts_are_used(tmp_path: Path) -> None:
    report, runs = _case("50_plus_findings")
    files = write_reports(report, runs, tmp_path)
    for pdf in (files.summary, files.details):
        fonts = set()
        for page in PdfReader(pdf).pages:
            for font in page["/Resources"]["/Font"].values():
                fonts.add(str(font.get_object()["/BaseFont"]))
        assert fonts and all("Inter" in f for f in fonts), fonts


def test_no_text_below_nine_points(tmp_path: Path) -> None:
    assert all(st.fontSize >= MIN_FONT_SIZE for st in Styles().all())
    report, runs = _case("50_plus_findings")
    files = write_reports(report, runs, tmp_path)
    for pdf in (files.summary, files.details):
        raw = pdf.read_bytes().decode("latin-1")  # page streams are uncompressed
        sizes = [float(m) for m in re.findall(r"/F\d+\+?\d* ([\d.]+) Tf", raw)]
        assert sizes and min(sizes) >= MIN_FONT_SIZE


def test_card_and_slot_limits_hold(tmp_path: Path) -> None:
    report, runs = _case("slots_at_their_limits")
    view = build_view(report, runs)
    assert view.cards
    for card in view.cards:
        assert len(card.what_happened) <= limit("what_happened")
        assert len(card.what_to_change) <= limit("what_to_change")
    for f in view.findings:
        assert len(f.quote) <= limit("evidence_quote")


def test_page_one_references_land_on_the_cited_details_page(tmp_path: Path) -> None:
    for case in ("50_plus_findings", "long_requests", "20_agents"):
        report, runs = _case(case)
        files = write_reports(report, runs, tmp_path / case)
        page1 = " ".join(_text(files.summary)[0].split())
        details = _text(files.details)
        refs = re.findall(r"Finding (F\d+), details\.pdf page (\d+)", page1)
        assert len(refs) == 3, case
        for finding_id, page in refs:
            assert f"{finding_id} · " in details[int(page) - 1], (case, finding_id, page)


def test_summary_speaks_plain_language(tmp_path: Path) -> None:
    for case in ("50_plus_findings", "20_agents", "zero_findings"):
        report, runs = _case(case)
        files = write_reports(report, runs, tmp_path / case)
        text = " ".join(_text(files.summary))
        assert not re.search(r"\b(node|nodes|span|spans)\b", text, re.IGNORECASE)
        ids = [m for m in CHECK_ID.findall(text) if not m.endswith("pdf")]
        assert not ids, ids


def test_low_confidence_never_on_page_one(tmp_path: Path) -> None:
    runs = _runs(10, ["supervisor"])
    low = [
        _finding(r, "design.empty_answer", "high", judge=False, agent="supervisor", text="x.")
        for r in runs
    ]
    for f in low:
        f.needs_confirmation = True
    report = _report(runs, low)
    view = build_view(report, runs)
    assert all(confidence_level(f) == "low" for f in low)
    assert view.cards == [] and view.main_problems == []
    assert len(view.low_problems) == 1
    assert view.verdict.startswith("None of the 10 runs had a serious problem.")
    files = write_reports(report, runs, tmp_path)
    assert "Final answer was empty" not in _text(files.summary)[0]


def test_verdict_and_tiles_follow_the_templates() -> None:
    report, runs = _case("50_plus_findings")
    view = build_view(report, runs)
    m = re.fullmatch(
        r"(\d+) of 40 runs had a serious problem\. The most common: (.+) \((\d+) runs?\)\.",
        view.verdict,
    )
    assert m, view.verdict
    assert [t.label for t in view.tiles] == [
        catalog()["labels"][k]
        for k in (
            "tile_serious",
            "tile_median_cost",
            "tile_median_time",
            "tile_median_steps",
            "tile_serious_cost",
        )
    ]
    assert view.tiles[0].value.startswith(f"{m.group(1)} (")
    assert all(re.search(r"\d", t.value) for t in view.tiles)


def test_findings_and_runs_are_numbered_in_order() -> None:
    report, runs = _case("50_plus_findings")
    view = build_view(report, runs)
    assert [f.id for f in view.findings] == [f"F{i + 1}" for i in range(len(view.findings))]
    assert [r.id for r in view.runs] == [f"R{i + 1}" for i in range(len(runs))]
    assert view.cards[0].finding_id == view.main_problems[0].findings[0].id


def test_unsupported_characters_are_replaced() -> None:
    assert safe("agent 🤖 研究") == "agent ? ??"
    assert safe("line\nbreak\x00") == "line break"
    assert safe("café → naïve “quotes” — ok") == "café → naïve “quotes” — ok"


def test_fixture_report_renders(tmp_path: Path) -> None:
    runs = [load_fixture(n) for n in AGENTS]
    files = write_reports(analyze(runs, Config()), runs, tmp_path, name="Fixture agents")
    page1 = _text(files.summary)[0]
    assert "Fixture agents" in page1
    assert "8 runs" in page1
