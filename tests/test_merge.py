"""Findings in the same run at the same agent and step are one problem: one finding, with the
most specific check, and the others listed as related. Counts are of distinct problems."""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from tests.fakes import FakeLLM, always, verdict
from tests.helpers import load_fixture
from tracedoc.analyze import analyze
from tracedoc.checks.rules.empty_answer import EmptyAnswer
from tracedoc.config import Config
from tracedoc.merge import merge_duplicates, specificity
from tracedoc.report import write_reports
from tracedoc.schema import Finding

_STEP = re.compile(r"^\[(S\d+)\] MODEL CALL · agent=supervisor", re.MULTILINE)
JUDGED = {"verification.premature_finish", "agents.ignored_output"}


def _supervisor_final_reply(prompt: str, check_id: str):
    """The fake judge confirms both judge checks at the supervisor's last model call."""
    if check_id not in JUDGED or "supervisor -> research_agent  <-- focus" in prompt:
        return verdict(False)
    return verdict(
        True,
        0.9,
        [_STEP.findall(prompt)[-1]],
        agent="supervisor",
        what=f"{check_id}: the supervisor dropped research_agent's answer.",
        change="Add to supervisor's prompt: \"Repeat research_agent's answer to the user.\"",
    )


def _judged_nested_agents():
    config = Config()
    config.judge.include_unvalidated = True
    run = load_fixture("nested_agents")
    client = FakeLLM(triage=always(True), confirm=_supervisor_final_reply)
    return analyze([run], config, client=client), run


def test_three_checks_on_the_supervisors_empty_reply_are_one_finding(tmp_path: Path) -> None:
    report, run = _judged_nested_agents()
    findings = [f for g in report.groups for f in g.findings]
    (merged,) = [f for f in findings if f.node == "supervisor"]
    assert merged.check_id == "agents.ignored_output"  # the most specific: says why
    assert merged.severity == "high"  # as serious as the empty answer it explains
    assert {r.check_id for r in merged.related} == {
        "design.empty_answer",
        "verification.premature_finish",
    }
    assert report.scorecard["findings_by_severity"]["high"] == len(
        [f for f in findings if f.severity == "high"]
    )

    files = write_reports(report, [run], tmp_path)
    view = files.view
    assert [p.title for p in view.main_problems] == ["Agent ignored another agent's output"]
    assert len(view.cards) == 1 and view.cards[0].title == "Agent ignored another agent's output"
    assert view.verdict.startswith("1 of 1 run had a serious problem.")

    details = " ".join(
        " ".join((p.extract_text() or "").split()) for p in PdfReader(files.details).pages
    )
    section2 = details.split("2. Every finding", 1)[1].split("3. Run index", 1)[0]
    assert section2.count("Worked example:") == 1  # one problem, not three
    assert "Also found at the same step Final answer was empty (1 run)" in section2
    assert "Also found at this step Final answer was empty (found by rule):" in section2
    assert "Run finished before the task was done (found by AI judge):" in section2


def test_most_specific_check_wins_but_never_a_less_certain_one() -> None:
    run = load_fixture("nested_agents")
    (empty,) = EmptyAnswer(Config()).run([run])
    vague = empty.model_copy(
        update={"check_id": "verification.premature_finish", "method": "judge", "confidence": 0.9}
    )
    unsure = empty.model_copy(
        update={
            "check_id": "agents.ignored_output",
            "method": "judge",
            "confidence": 0.5,  # low confidence
            "severity": "medium",
        }
    )
    (merged,) = merge_duplicates([vague, empty, unsure], [run])
    assert merged.check_id == "design.empty_answer"  # ignored_output is more specific, but unsure
    assert {r.check_id for r in merged.related} == {
        "verification.premature_finish",
        "agents.ignored_output",
    }
    assert specificity("agents.ignored_output") > specificity("design.empty_answer")
    assert specificity("design.empty_answer") > specificity("verification.premature_finish")


def test_different_steps_agents_runs_and_cost_findings_stay_apart() -> None:
    run = load_fixture("nested_agents")
    (empty,) = EmptyAnswer(Config()).run([run])
    research_reply = run.steps_of("llm")[-2]
    elsewhere = empty.model_copy(
        update={
            "check_id": "verification.premature_finish",
            "evidence_step_ids": [research_reply.id],
            "node": "research_agent",
        }
    )
    other_run = empty.model_copy(update={"run_id": "another-run"})
    slow = empty.model_copy(update={"check_id": "cost.latency_outlier", "severity": "low"})
    whole_run = Finding(
        check_id="design.no_stop",
        group="design",
        method="rule",
        severity="high",
        run_id=run.id,
        node="supervisor",
        explanation="x",
    )
    kept = merge_duplicates([empty, elsewhere, other_run, slow, whole_run], [run])
    assert kept == [empty, elsewhere, other_run, slow, whole_run]
    assert all(not f.related for f in kept)


def test_a_related_finding_keeps_its_judge_disagreed_mark() -> None:
    from tracedoc.report.view import build_view
    from tracedoc.schema import Report

    run = load_fixture("nested_agents")
    (empty,) = EmptyAnswer(Config()).run([run])
    split = empty.model_copy(
        update={
            "check_id": "verification.premature_finish",
            "method": "judge",
            "confidence": 0.5,
            "judge_disagreed": True,
        }
    )
    (merged,) = merge_duplicates([empty, split], [run])
    assert merged.check_id == "design.empty_answer" and merged.related[0].judge_disagreed
    from tracedoc.group import group_findings

    view = build_view(Report(groups=group_findings([merged], [run])), [run])
    (line,) = view.findings[0].related
    assert line.startswith("Run finished before the task was done (found by AI judge, judge ")


def test_the_same_check_twice_at_one_step_is_one_finding_not_related() -> None:
    run = load_fixture("nested_agents")
    (empty,) = EmptyAnswer(Config()).run([run])
    again = empty.model_copy(update={"key": "another window"})
    (merged,) = merge_duplicates([empty, again], [run])
    assert merged.related == []


def test_rules_only_report_is_unchanged_when_nothing_overlaps() -> None:
    run = load_fixture("nested_agents")
    report = analyze([run], Config())
    (finding,) = [f for g in report.groups for f in g.findings]
    assert finding.check_id == "design.empty_answer" and finding.related == []
