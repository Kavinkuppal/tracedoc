"""Judge stability: every confirmation is asked twice, independently. Both confirm: kept.
They disagree: kept at low confidence, marked "judge disagreed". Both reject: dropped."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfReader

from tests.fakes import FakeLLM, always, verdict
from tests.helpers import load_fixture
from tracedoc.analyze import analyze
from tracedoc.checks.base import run_rule_checks
from tracedoc.checks.judges.prompts import judge_prompts
from tracedoc.checks.judges.runner import REVIEWS, Judge
from tracedoc.config import Config
from tracedoc.llm.cache import CachingClient
from tracedoc.report import write_reports
from tracedoc.report.view import build_view, confidence_level, method_label

SECOND = "second, independent review"
CHECK = "verification.premature_finish"


def _reviews(first: tuple[bool, float], second: tuple[bool, float]):
    """A confirm function answering the first review with `first`, the second with `second`."""

    def confirm(prompt: str, check_id: str):
        ok, confidence = second if SECOND in prompt else first
        review = "second" if SECOND in prompt else "first"
        return verdict(ok, confidence, ["S7"], agent="supervisor", what=f"{review} review.")

    return confirm


def _judge_run(confirm, *, client=None):
    config = Config()
    config.judge.include_unvalidated = True
    prompt = judge_prompts()[CHECK]
    fake = client or FakeLLM(triage=always(True), confirm=confirm)
    judge = Judge(config, {CHECK: prompt}, client=fake)
    run = load_fixture("nested_agents")
    return judge.run(judge.plan([run], []), []).findings, fake


def test_both_reviews_confirm_keeps_the_finding_at_the_lower_confidence() -> None:
    (f,), fake = _judge_run(_reviews((True, 0.9), (True, 0.8)))
    assert f.check_id == CHECK and not f.judge_disagreed
    assert f.confidence == 0.8 and confidence_level(f) == "high"
    assert f.explanation == "first review."
    confirms = [p for p in fake.prompts if p.startswith("USER TASK") and "CHECK:" in p]
    assert len(confirms) == REVIEWS == 2 and sum(SECOND in p for p in confirms) == 1


@pytest.mark.parametrize(
    ("first", "second", "text"),
    [((True, 0.9), (False, 0.9), "first review."), ((False, 0.9), (True, 0.85), "second review.")],
)
def test_disagreeing_reviews_keep_it_at_low_confidence(first, second, text) -> None:
    (f,), _ = _judge_run(_reviews(first, second))
    assert f.judge_disagreed and confidence_level(f) == "low" and f.confidence <= 0.5
    assert f.explanation == text  # the review that confirmed it
    assert method_label(f) == "found by AI judge, judge disagreed"


def test_a_review_below_the_confidence_bar_counts_as_a_rejection() -> None:
    (f,), _ = _judge_run(_reviews((True, 0.9), (True, 0.3)))
    assert f.judge_disagreed


def test_both_reviews_reject_drops_it() -> None:
    findings, _ = _judge_run(_reviews((False, 0.9), (False, 0.9)))
    assert findings == []


def _candidate_run(first: bool, second: bool):
    """tool_output_unused: a rule candidate (tools.unused_output) the judge confirms."""
    config = Config()
    run = load_fixture("tool_output_unused")
    findings = [f for f in run_rule_checks([run]) if f.check_id == "tools.unused_output"]
    assert findings and all(f.needs_confirmation for f in findings)

    def confirm(prompt: str, check_id: str):
        return verdict(second if SECOND in prompt else first, 0.9)

    judge = Judge(config, {}, client=FakeLLM(triage=always(True), confirm=confirm))
    return judge.run(judge.plan([run], findings), findings).findings


def test_rule_candidates_are_reviewed_twice_too() -> None:
    (agreed,) = [f for f in _candidate_run(True, True) if f.check_id == "tools.unused_output"]
    assert agreed.confirmed_by_judge and not agreed.judge_disagreed
    assert confidence_level(agreed) == "high"

    (split,) = [f for f in _candidate_run(True, False) if f.check_id == "tools.unused_output"]
    assert split.judge_disagreed and not split.needs_confirmation
    assert confidence_level(split) == "low"
    assert method_label(split) == "found by rule, judge disagreed"

    assert [f for f in _candidate_run(False, False) if f.check_id == "tools.unused_output"] == []


def test_disagreement_is_marked_in_details_and_kept_off_page_one(tmp_path: Path) -> None:
    config = Config()
    config.judge.include_unvalidated = True
    run = load_fixture("healthy")  # nothing else is found where the judge points

    def confirm(prompt: str, check_id: str):
        return verdict(check_id == CHECK and SECOND not in prompt, 0.9)

    report = analyze([run], config, client=FakeLLM(triage=always(True), confirm=confirm))
    files = write_reports(report, [run], tmp_path)
    view = build_view(report, [run])
    disagreed = [f for f in view.findings if f.finding.judge_disagreed]
    assert disagreed and all(f.level == "low" for f in disagreed)
    assert all(c.finding_id not in {f.id for f in disagreed} for c in files.view.cards)
    details = " ".join(
        " ".join((p.extract_text() or "").split()) for p in PdfReader(files.details).pages
    )
    assert "found by AI judge, judge disagreed" in details


def test_both_reviews_are_cached_so_a_rerun_gives_the_same_findings(tmp_path: Path) -> None:
    fake = FakeLLM(triage=always(True), confirm=_reviews((True, 0.9), (False, 0.9)))
    first, _ = _judge_run(None, client=CachingClient(fake, tmp_path))
    calls = len(fake.calls)
    again, _ = _judge_run(None, client=CachingClient(fake, tmp_path))
    assert len(fake.calls) == calls  # every answer, both reviews included, from the cache
    assert [f.model_dump() for f in again] == [f.model_dump() for f in first]
    assert first[0].judge_disagreed


def test_the_estimate_counts_two_reviews_per_confirmation() -> None:
    config = Config()
    config.judge.include_unvalidated = True
    run = load_fixture("tool_output_unused")
    findings = run_rule_checks([run])
    judge = Judge(config, {}, client=None)
    estimate = judge.estimate(judge.plan([run], findings))
    candidates = sum(1 for t in judge.plan([run], findings).tasks if t.confirmation)
    assert candidates and estimate.confirm_calls_max == REVIEWS * candidates
