"""Phase 2: rule checks and signals."""

from __future__ import annotations

import copy
import time

import pytest

from tests.builders import ai, build_run, step
from tests.fixtures.agents import AGENTS
from tests.helpers import SOURCES, load_fixture
from tracedoc.checks.base import REGISTRY, enabled_checks, run_rule_checks
from tracedoc.checks.signals import find_signals
from tracedoc.config import Config, Thresholds

RULE_CHECKS = {
    "design.empty_answer",
    "design.loop",
    "design.no_stop",
    "tools.errors_retries",
    "tools.unused_output",
    "verification.unchecked_result",
    "cost.token_hotspot",
    "cost.latency_outlier",
}
RULE_BUGS = {name: module.BUG for name, module in AGENTS.items() if module.BUG in RULE_CHECKS}


def _findings(name: str, source: str = "capture", config: Config | None = None):
    return run_rule_checks([load_fixture(name, source)], config)


def test_registry_holds_all_rule_checks() -> None:
    assert {c.id for c in enabled_checks(Config(), "rule")} == RULE_CHECKS
    for cls in REGISTRY.values():
        assert cls.id.split(".")[0] in {"design", "agents", "verification", "tools", "cost"}


def test_four_fixture_bugs_are_rule_detectable() -> None:
    assert set(RULE_BUGS) == {
        "nested_agents",
        "researcher_repeat_search",
        "supervisor_no_end",
        "tool_retry_same_args",
        "tool_output_unused",
    }


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("name", sorted(RULE_BUGS))
def test_buggy_fixture_triggers_its_check(name: str, source: str) -> None:
    findings = _findings(name, source)
    assert RULE_BUGS[name] in {f.check_id for f in findings}


@pytest.mark.parametrize("source", SOURCES)
def test_healthy_fixture_has_no_high_severity_findings(source: str) -> None:
    findings = _findings("healthy", source)
    assert not [f for f in findings if f.severity == "high"]
    assert not [f for f in findings if f.severity == "medium"]


def test_loop_identifies_tool_and_waste() -> None:
    (f,) = [f for f in _findings("researcher_repeat_search") if f.check_id == "design.loop"]
    assert f.key == "web_search"
    assert len(f.evidence_step_ids) == 3
    assert f.impact.tokens > 0
    assert not f.needs_confirmation


def test_failed_retries_are_not_double_counted_as_loops() -> None:
    ids = {f.check_id for f in _findings("tool_retry_same_args")}
    assert "tools.errors_retries" in ids
    assert "design.loop" not in ids


def test_errors_retries_severity_and_evidence() -> None:
    (f,) = [f for f in _findings("tool_retry_same_args") if f.check_id == "tools.errors_retries"]
    assert f.severity == "high"
    assert len(f.evidence_step_ids) == 3
    assert "ConnectionError" in f.evidence_excerpts[0]


def test_unused_output_is_a_judge_candidate() -> None:
    (f,) = [f for f in _findings("tool_output_unused") if f.check_id == "tools.unused_output"]
    assert f.needs_confirmation
    assert f.key == "lookup_order"
    assert "CANCELLED" in f.evidence_excerpts[0]


def test_used_tool_output_is_not_flagged() -> None:
    for name in ("healthy", "researcher_repeat_search"):
        assert "tools.unused_output" not in {f.check_id for f in _findings(name)}


def test_no_stop_on_recursion_limit_and_step_limit() -> None:
    (f,) = [f for f in _findings("supervisor_no_end") if f.check_id == "design.no_stop"]
    assert f.severity == "high"
    assert f.node == "supervisor"
    config = Config(thresholds=Thresholds(step_limit=7))
    assert "design.no_stop" in {
        f.check_id for f in _findings("researcher_repeat_search", config=config)
    }


def test_supervisor_cycle_detected_as_loop() -> None:
    (f,) = [f for f in _findings("supervisor_no_end") if f.check_id == "design.loop"]
    assert "supervisor -> researcher" in f.key


def test_unchecked_result_skips_verified_and_unfinished_runs() -> None:
    flagged = {
        name
        for name in AGENTS
        if "verification.unchecked_result" in {f.check_id for f in _findings(name)}
    }
    assert "healthy" not in flagged
    assert "supervisor_no_end" not in flagged
    assert "tool_output_unused" in flagged


def test_near_duplicate_tool_calls_go_to_judge() -> None:
    specs = []
    for q in ("best pizza in rome", "best pizza in rome?", "best pizza in rome!"):
        specs += [step("node", "agent"), step("llm", "agent", tokens_in=10, tokens_out=5)]
        specs += [step("node", "tools"), step("tool", "tools", name="search", input={"q": q})]
    run = build_run("r", specs)
    loops = [f for f in run_rule_checks([run]) if f.check_id == "design.loop"]
    assert len(loops) == 1
    assert loops[0].needs_confirmation


def test_token_share_hotspot() -> None:
    specs = [
        step("node", "planner"),
        step("llm", "planner", tokens_in=500, tokens_out=100),
        step("node", "writer"),
        step("llm", "writer", tokens_in=9000, tokens_out=1000),
        step("node", "critic"),
        step("llm", "critic", tokens_in=500, tokens_out=100),
    ]
    findings = [f for f in run_rule_checks([build_run("r", specs)]) if f.key == "share:writer"]
    assert len(findings) == 1
    assert findings[0].impact.tokens == 10000


def test_context_growth() -> None:
    specs = [step("node", "agent")]
    for tokens in (800, 1600, 2400, 4000):
        specs.append(step("llm", "agent", tokens_in=tokens, tokens_out=50))
    findings = run_rule_checks([build_run("r", specs)])
    assert "context_growth" in {f.key for f in findings}


def test_latency_outlier_relative_and_absolute() -> None:
    runs = [
        build_run(f"r{i}", [step("node", "agent", dur=2.0), step("node", "writer", dur=1.0)])
        for i in range(8)
    ]
    runs.append(build_run("slow", [step("node", "agent", dur=12.0)]))
    runs.append(build_run("very_slow", [step("node", "writer", dur=90.0)]))
    findings = [f for f in run_rule_checks(runs) if f.check_id == "cost.latency_outlier"]
    by_run = {f.run_id: f for f in findings}
    assert set(by_run) == {"slow", "very_slow"}
    assert by_run["slow"].severity == "low"
    assert by_run["very_slow"].severity == "medium"


def test_latency_ignores_tiny_durations() -> None:
    runs = [build_run(f"r{i}", [step("node", "a", dur=0.001)]) for i in range(8)]
    runs.append(build_run("x", [step("node", "a", dur=0.05)]))
    assert not [f for f in run_rule_checks(runs) if f.check_id == "cost.latency_outlier"]


def test_disabled_checks_and_severity_overrides() -> None:
    config = Config()
    config.checks.disabled = ["design.loop"]
    config.checks.severity = {"tools.errors_retries": "low"}
    findings = run_rule_checks(
        [load_fixture("researcher_repeat_search"), load_fixture("tool_retry_same_args")], config
    )
    ids = {f.check_id for f in findings}
    assert "design.loop" not in ids
    assert {f.severity for f in findings if f.check_id == "tools.errors_retries"} == {"low"}


def test_finding_ids_are_stable() -> None:
    a = [f.id for f in _findings("researcher_repeat_search")]
    b = [f.id for f in _findings("researcher_repeat_search")]
    assert a == b
    assert len(set(a)) == len(a)


# ---- signals --------------------------------------------------------------


def test_signals_on_fixtures() -> None:
    runs = [load_fixture(n) for n in AGENTS]
    names = {r.id: n for r, n in zip(runs, AGENTS, strict=True)}
    signals = {(names[s.run_id], s.check_id) for s in find_signals(runs)}
    assert ("healthy", "good.verified") in signals
    assert ("healthy", "good.clean_handoff") in signals
    assert not {s for s in signals if s[0] == "tool_retry_same_args"}


def test_error_recovery_signal() -> None:
    specs = [
        step("node", "tools"),
        step("tool", "tools", name="fetch", input={"url": "a"}, error="Timeout"),
        step("node", "tools"),
        step("tool", "tools", name="search", input={"q": "a"}, output="ok"),
    ]
    (s,) = find_signals([build_run("r", specs)])
    assert s.check_id == "good.error_recovery"


def test_efficient_path_signal() -> None:
    def run(i: int, n: int):
        return build_run(f"r{i}", [step("node", "agent")] * n, task_type="research")

    runs = [run(i, 10) for i in range(5)] + [run(9, 3)]
    found = [s for s in find_signals(runs) if s.check_id == "good.efficient_path"]
    assert [s.run_id for s in found] == ["r9"]


def test_signals_never_carry_problem_severity() -> None:
    runs = [load_fixture(n) for n in AGENTS]
    assert {s.severity for s in find_signals(runs)} <= {"low"}
    assert all(s.check_id.startswith("good.") for s in find_signals(runs))


def test_unused_output_compares_against_outputs_not_inputs() -> None:
    specs = [
        step("node", "agent"),
        step("llm", "agent", output=ai("calling tool")),
        step("node", "tools"),
        step("tool", "tools", name="lookup", output="Flight BA117 departs 18:45 from gate B32"),
        step("node", "agent"),
        step(
            "llm",
            "agent",
            input=[{"type": "tool", "content": "Flight BA117 departs 18:45 from gate B32"}],
            output=ai("Have a nice trip!"),
        ),
    ]
    findings = run_rule_checks([build_run("r", specs)])
    assert "tools.unused_output" in {f.check_id for f in findings}


# ---- performance ----------------------------------------------------------


def test_rules_on_100_runs_under_10_seconds() -> None:
    base = [load_fixture(n) for n in AGENTS]
    runs = []
    for i in range(100):
        run = copy.deepcopy(base[i % len(base)])
        run.id = f"{run.id}-{i}"
        runs.append(run)
    start = time.perf_counter()
    run_rule_checks(runs)
    find_signals(runs)
    assert time.perf_counter() - start < 10


def test_empty_answer_is_high_severity_at_the_answering_agent() -> None:
    run = load_fixture("nested_agents")
    (f,) = [f for f in run_rule_checks([run]) if f.check_id == "design.empty_answer"]
    assert f.severity == "high"
    assert f.node == "supervisor"
    assert f.evidence_step_ids == [run.steps_of("llm")[-1].id]


def test_runs_without_an_answer_do_not_count_as_completed() -> None:
    from tracedoc.analyze import analyze

    runs = [load_fixture(n) for n in AGENTS]
    report = analyze(runs, Config())
    completed = sum(r.status == "completed" for r in runs)
    assert completed == len(runs) - 2  # recursion limit + empty answer
    assert report.scorecard["completion_rate"] == pytest.approx(completed / len(runs))
