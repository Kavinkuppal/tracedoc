"""Phase 3: judge planning, windows, two-stage judging, gating, and cost."""

from __future__ import annotations

import copy

import pytest

from tests.fakes import FakeLLM, always, verdict
from tests.fixtures.agents import AGENTS
from tests.helpers import load_fixture
from tracedoc.analyze import prepare
from tracedoc.checks.base import run_rule_checks
from tracedoc.checks.judges.prompts import (
    confirmation_prompts,
    enabled_judges,
    judge_prompts,
)
from tracedoc.checks.judges.runner import Judge
from tracedoc.checks.judges.windows import run_end_window, window_around
from tracedoc.config import Config
from tracedoc.llm.cost import Spend

JUDGE_CHECKS = {
    "design.ignored_instructions",
    "design.lost_context",
    "agents.incomplete_handoff",
    "agents.ignored_output",
    "agents.reason_action_mismatch",
    "agents.no_clarification",
    "agents.task_drift",
    "verification.premature_finish",
}


def _config(**judge: object) -> Config:
    config = Config()
    config.judge.include_unvalidated = True
    for k, v in judge.items():
        setattr(config.judge, k, v)
    return config


def _runs() -> list:
    return [load_fixture(n) for n in AGENTS]


def _judge(config: Config, client: FakeLLM | None = None, spend: Spend | None = None) -> Judge:
    judges, _ = enabled_judges(config)
    return Judge(config, judges, client=client, spend=spend)


def _batch(n: int) -> list:
    base = _runs()
    runs = []
    for i in range(n):
        run = copy.deepcopy(base[i % len(base)])
        run.id = f"{run.id}-{i}"
        for s in run.steps:
            s.run_id = run.id
        runs.append(run)
    return runs


# ---- prompts and gating --------------------------------------------------------


def test_all_judge_checks_have_versioned_prompts() -> None:
    prompts = judge_prompts()
    assert set(prompts) == JUDGE_CHECKS
    for p in prompts.values():
        assert p.version >= 1 and p.triage and p.confirm and p.scope
    assert set(confirmation_prompts()) == {
        "design.loop",
        "tools.unused_output",
        "verification.unchecked_result",
    }


def test_unmeasured_checks_are_disabled_by_default() -> None:
    enabled, disabled = enabled_judges(Config(), precision={})
    assert not enabled
    assert set(disabled) == JUDGE_CHECKS


def test_gating_by_precision_and_version() -> None:
    v = judge_prompts()["agents.ignored_output"].version
    precision = {
        "agents.ignored_output": {"precision": 0.9, "prompt_version": v},
        "agents.task_drift": {"precision": 0.6, "prompt_version": 1},
        "design.lost_context": {"precision": 0.95, "prompt_version": 999},
    }
    enabled, disabled = enabled_judges(Config(), precision)
    assert set(enabled) == {"agents.ignored_output"}
    assert "below" in disabled["agents.task_drift"]
    assert "no precision measured" in disabled["design.lost_context"]

    config = Config()
    config.judge.enabled = ["agents.task_drift"]
    config.checks.disabled = ["agents.ignored_output"]
    enabled, disabled = enabled_judges(config, precision)
    assert set(enabled) == {"agents.task_drift"}
    assert disabled["agents.ignored_output"] == "disabled in config"


# ---- windows -----------------------------------------------------------------


def test_window_render_shows_task_instructions_labels_and_end() -> None:
    run = load_fixture("pipeline_dropped_caveat")
    (handoff,) = run.steps_of("handoff")
    window = window_around(run, [handoff.id], 3, "handoff")
    text = window.render(2000)
    assert text.startswith("USER TASK: Did the new onboarding flow")
    assert "Preserve every caveat" in text
    assert "[S1] MODEL CALL" in text
    assert "HANDOFF analyst -> writer  <-- focus" in text
    assert "RUN ENDED with status 'completed'" in text
    assert window.step_for("S2").id == handoff.id


def test_window_is_bounded_and_clips_long_fields() -> None:
    run = load_fixture("supervisor_no_end")
    window = run_end_window(run, 2)
    assert len(window.steps) == 5
    text = window.render(30)
    assert "earlier steps not shown" in text
    assert "chars omitted" in text


def test_node_evidence_maps_to_actions_inside_the_node() -> None:
    run = load_fixture("supervisor_no_end")
    node = run.steps_of("node")[5]
    window = window_around(run, [node.id], 1, "candidate")
    assert any(s.parent_id == node.id for s in window.steps)


# ---- planning ----------------------------------------------------------------


def test_plan_covers_candidates_handoffs_and_sample() -> None:
    runs = _runs()
    findings = run_rule_checks(runs)
    plan = _judge(_config()).plan(runs, findings)
    candidates = [t for t in plan.tasks if t.confirmation]
    assert len(candidates) == sum(1 for f in findings if f.needs_confirmation)
    handoff_runs = {t.window.run.id for t in plan.tasks if t.window.kind == "handoff"}
    assert handoff_runs == {r.id for r in runs if r.steps_of("handoff")}
    sampled = [t for t in plan.tasks if t.window.kind == "run"]
    assert len(sampled) == len(runs)  # sample_runs (12) > 6 runs


def test_plan_caps_handoffs_per_run_distinct_pairs_first() -> None:
    run = load_fixture("supervisor_no_end")
    plan = _judge(_config(max_handoffs_per_run=3)).plan([run], [])
    handoffs = [t for t in plan.tasks if t.window.kind == "handoff"]
    assert len(handoffs) == 3
    pairs = [next(s for s in t.window.steps if s.id == t.window.focus[0]).name for t in handoffs]
    assert len(set(pairs)) == 3


def test_plan_caps_windows_per_signature_and_is_deterministic() -> None:
    runs = _batch(30)
    findings = run_rule_checks(runs)
    judge = _judge(_config(max_windows_per_signature=2))
    plan = judge.plan(runs, findings)
    unused = [
        t for t in plan.tasks if t.confirmation and t.confirmation.id == "tools.unused_output"
    ]
    assert len(unused) == 2
    again = judge.plan(runs, findings)
    assert [t.window.key for t in plan.tasks] == [t.window.key for t in again.tasks]


def test_rules_only_plan_has_no_judge_windows_when_checks_disabled() -> None:
    runs = _runs()
    config = Config()  # nothing measured -> judge checks disabled
    plan = _judge(config).plan(runs, run_rule_checks(runs))
    assert all(t.confirmation for t in plan.tasks)


# ---- cost ----------------------------------------------------------------------


def test_50_fixture_runs_estimate_under_two_dollars() -> None:
    runs = _batch(50)
    prepared = prepare(runs, _config())
    estimate = prepared.estimate()
    assert estimate.cost is not None
    assert estimate.cost < 2.0, estimate


def test_estimate_reports_unpriced_models() -> None:
    config = _config()
    config.llm.confirm_model = "some-local-model"
    est = prepare(_runs(), config).estimate()
    assert est.cost is None
    assert est.unpriced_models == ["some-local-model"]


# ---- judging -------------------------------------------------------------------


def _caveat_oracle(prompt: str, check_id: str):
    dropped = "12 customers" in prompt and "rolling it out" in prompt
    if check_id == "agents.ignored_output" and dropped:
        return verdict(True, 0.85, evidence=["S2", "S3"])
    return verdict(False)


def test_judge_finds_the_dropped_caveat() -> None:
    runs = _runs()
    findings = run_rule_checks(runs)
    client = FakeLLM(triage=always(True), confirm=_caveat_oracle)
    judge = _judge(_config(), client)
    outcome = judge.run(judge.plan(runs, findings), findings)
    pipeline = load_fixture("pipeline_dropped_caveat")
    hits = [f for f in outcome.findings if f.check_id == "agents.ignored_output"]
    assert {f.run_id for f in hits} == {pipeline.id}
    (f,) = [f for f in hits if f.key == "analyst->writer"]
    assert f.method == "judge"
    assert f.severity == "high"
    assert f.node == "writer"
    step_ids = {s.id for s in pipeline.steps}
    assert f.evidence_step_ids and set(f.evidence_step_ids) <= step_ids
    assert not outcome.errors


def test_triage_gates_confirmation_calls() -> None:
    runs = [load_fixture("healthy")]
    client = FakeLLM(triage=always(False))
    judge = _judge(_config(), client)
    judge.run(judge.plan(runs, []), [])
    assert {schema for _, schema in client.calls} == {"TriageResult"}
    assert all(model == "claude-haiku-4-5-20251001" for model, _ in client.calls)


def test_low_confidence_triage_is_ignored() -> None:
    runs = [load_fixture("healthy")]
    client = FakeLLM(triage=always(True, 0.1))
    judge = _judge(_config(), client)
    judge.run(judge.plan(runs, []), [])
    assert "Verdict" not in {schema for _, schema in client.calls}


def test_candidates_confirmed_dropped_or_left_unconfirmed() -> None:
    # One unused-output candidate and one unchecked-result candidate.
    runs = [load_fixture("tool_output_unused"), load_fixture("researcher_repeat_search")]
    findings = run_rule_checks(runs)
    cands = {f.check_id for f in findings if f.needs_confirmation}
    assert cands == {"tools.unused_output", "verification.unchecked_result"}

    def confirm(prompt: str, check_id: str):
        return verdict(check_id == "tools.unused_output", 0.8)

    judge = _judge(Config(), FakeLLM(triage=always(True), confirm=confirm))
    out = judge.run(judge.plan(runs, findings), findings)
    by_check = {f.check_id: f for f in out.findings}
    assert not by_check["tools.unused_output"].needs_confirmation
    assert by_check["tools.unused_output"].confidence == 0.8
    assert "judge explanation" in by_check["tools.unused_output"].explanation
    assert "verification.unchecked_result" not in by_check  # rejected by the judge

    judge = _judge(Config(), FakeLLM(triage=always(False)))
    out = judge.run(judge.plan(runs, findings), findings)
    assert not {f.check_id for f in out.findings} & cands  # dismissed at triage

    judge = _judge(Config(), FakeLLM(fail_on=lambda p: True))
    out = judge.run(judge.plan(runs, findings), findings)
    assert {f.check_id for f in out.findings if f.needs_confirmation} == cands  # errors: kept
    assert len(out.errors) == 2


def test_certain_rule_findings_pass_through_untouched() -> None:
    runs = [load_fixture("tool_retry_same_args")]
    findings = run_rule_checks(runs)
    judge = _judge(Config(), FakeLLM())
    out = judge.run(judge.plan(runs, findings), findings)
    certain = [f for f in findings if not f.needs_confirmation]
    assert all(f in out.findings for f in certain)


def test_budget_stops_judging() -> None:
    runs = _batch(12)
    findings = run_rule_checks(runs)
    client = FakeLLM(triage=always(True), confirm=lambda p, c: verdict(True))
    spend = Spend(prices=Config().prices, max_cost=0.02)
    judge = _judge(_config(concurrency=1), client, spend)
    plan = judge.plan(runs, findings)
    out = judge.run(plan, findings)
    assert out.stopped_for_budget
    assert spend.cost <= 0.02
    assert len(client.calls) < len(plan.tasks) * 2


def test_spend_is_recorded() -> None:
    runs = [load_fixture("healthy")]
    spend = Spend(prices=Config().prices)
    judge = _judge(
        _config(), FakeLLM(triage=always(True), confirm=lambda p, c: verdict(False)), spend
    )
    judge.run(judge.plan(runs, []), [])
    assert spend.calls > 0 and spend.cost > 0
    assert not spend.unpriced_models


@pytest.mark.parametrize("concurrency", [1, 4])
def test_results_do_not_depend_on_concurrency(concurrency: int) -> None:
    runs = _runs()
    findings = run_rule_checks(runs)
    judge = _judge(
        _config(concurrency=concurrency), FakeLLM(triage=always(True), confirm=_caveat_oracle)
    )
    out = judge.run(judge.plan(runs, findings), findings)
    assert sorted(f.id for f in out.findings) == sorted(
        f.id
        for f in _judge(
            _config(concurrency=1), FakeLLM(triage=always(True), confirm=_caveat_oracle)
        )
        .run(judge.plan(runs, findings), findings)
        .findings
    )
