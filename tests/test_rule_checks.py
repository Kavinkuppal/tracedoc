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
from tracedoc.schema import Step

RULE_CHECKS = {
    "design.empty_answer",
    "design.loop",
    "design.no_stop",
    "design.run_error",
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


def test_fixture_bugs_are_rule_detectable() -> None:
    assert set(RULE_BUGS) == {
        "nested_agents",
        "researcher_repeat_search",
        "supervisor_no_end",
        "swarm_parallel_handoff",
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
    assert "healthy" not in flagged  # a critic checked the answer
    assert "supervisor_no_end" not in flagged  # never finished
    assert "nested_agents" not in flagged  # no answer at all
    # Only answers that state facts taken from tool output count:
    assert "researcher_repeat_search" in flagged  # repeats the search results' terms
    assert "tool_output_unused" not in flagged  # its answer uses nothing from the tool
    assert "pipeline_dropped_caveat" not in flagged  # no tools at all


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
    def run(i: int, n: int, *, answer: str = "Meta had 67,317 employees.", tool: bool = True):
        specs = [step("node", "agent")] * (n - 1)
        specs += [step("tool", "agent", name="web_search" if tool else "transfer_to_math")]
        return build_run(f"r{i}", specs, task_type="research", final_answer=answer)

    runs = [run(i, 10) for i in range(5)] + [
        run(9, 3),  # counts: an answer, a tool, few steps
        run(8, 3, tool=False),  # only a handoff tool
        run(7, 3, answer=""),  # no answer
        run(6, 3, answer="Which year do you mean? Do you want all five companies?"),  # only asks
    ]
    found = [s for s in find_signals(runs) if s.check_id == "good.efficient_path"]
    assert [s.run_id for s in found] == ["r9"]


def test_efficient_path_needs_a_run_without_findings() -> None:
    from tracedoc.analyze import analyze

    def run(i: int, n: int):
        specs = [step("node", "agent")] * (n - 1) + [step("tool", "agent", name="web_search")]
        return build_run(f"r{i}", specs, task_type="research", final_answer="Done, here it is.")

    runs = [run(i, 10) for i in range(5)] + [run(9, 3)]
    assert any(s.check_id == "good.efficient_path" for s in analyze(runs, Config()).signals)
    runs[-1] = runs[-1].model_copy(update={"status": "error", "error": "ValueError: boom"})
    signals = analyze(runs, Config()).signals  # r9 now has a finding (it ended in an error)
    assert not any(s.check_id == "good.efficient_path" and s.run_id == "r9" for s in signals)


def test_clean_handoff_counts_only_what_the_sender_wrote() -> None:
    request = "What was the combined FAANG headcount in 2024?"
    table = "Meta 67,317 Apple 164,000 Amazon 1,551,000 Netflix 14,000 Google 181,269 total"
    history = [{"type": "human", "content": request}]
    specs = [
        step("node", "supervisor", role="supervisor"),
        step(
            "llm",
            "supervisor",
            input=history,
            output={"type": "ai", "content": "", "tool_calls": [{"name": "transfer_to_r"}]},
        ),
        step("node", "researcher", role="researcher"),
        step("llm", "researcher", input=history, output=ai(f"Here are the numbers: {table}")),
        step("node", "supervisor", role="supervisor"),
        step(
            "llm",
            "supervisor",
            input=[*history, ai(f"Here are the numbers: {table}")],
            output=ai("The combined headcount is 1,977,586."),
        ),
    ]
    run = build_run("r1", specs, input_preview=request, final_answer="1,977,586.")
    handoffs = [
        Step(
            id="h0",
            run_id="r1",
            kind="handoff",
            name="supervisor -> researcher",
            handoff_from="supervisor",
            handoff_to="researcher",
            node="researcher",
            started_at=run.steps[1].ended_at,
            ended_at=run.steps[2].started_at,
        ),
        Step(
            id="h1",
            run_id="r1",
            kind="handoff",
            name="researcher -> supervisor",
            handoff_from="researcher",
            handoff_to="supervisor",
            node="supervisor",
            started_at=run.steps[3].ended_at,
            ended_at=run.steps[4].started_at,
        ),
    ]
    run.steps = sorted([*run.steps, *handoffs], key=lambda s: s.started_at)
    found = [s for s in find_signals([run]) if s.check_id == "good.clean_handoff"]
    # supervisor -> researcher carried only the user's request (the supervisor wrote nothing)
    assert [(s.facts["sender"], s.facts["agent"]) for s in found] == [("researcher", "supervisor")]


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
    assert completed == len(runs) - 3  # recursion limit, empty answer, error
    assert report.scorecard["completion_rate"] == pytest.approx(completed / len(runs))


@pytest.mark.parametrize(
    "tool_name", ["transfer_to_research_expert", "transfer_back_to_supervisor"]
)
def test_handoff_tools_are_not_unused_output(tool_name: str) -> None:
    specs = [
        step("node", "supervisor", role="supervisor"),
        step("llm", "supervisor", output=ai("Routing you to the researcher now.")),
        step(
            "tool",
            "supervisor",
            name=tool_name,
            output="Successfully transferred to research_expert after the long handoff",
        ),
        step("node", "research_expert", role="researcher"),
        step("llm", "research_expert", output=ai("Here is what I found about rainfall.")),
    ]
    findings = run_rule_checks([build_run("r", specs)])
    assert "tools.unused_output" not in {f.check_id for f in findings}


def test_is_handoff_tool() -> None:
    from tracedoc.checks.util import is_handoff_tool

    assert is_handoff_tool("transfer_to_hotel_assistant")
    assert is_handoff_tool("transfer_back_to_supervisor")
    assert not is_handoff_tool("transfer_funds")
    assert not is_handoff_tool("web_search")
    assert not is_handoff_tool(None)


def _answer_run(answer: str, tool_output: str, request: str = "How big is the team?") -> object:
    specs = [
        step("node", "agent"),
        step("llm", "agent", output=ai("searching")),
        step("node", "tools"),
        step("tool", "tools", name="search", output=tool_output),
        step("node", "agent"),
        step("llm", "agent", output=ai(answer)),
    ]
    return build_run("r", specs, input_preview=request, final_answer=answer)


@pytest.mark.parametrize(
    ("answer", "tool_output", "user_request", "fires"),
    [
        (
            "The team has 1,551,000 people.",
            "Amazon: 1,551,000 employees",
            "How big is Amazon?",
            True,
        ),
        ("Revenue grew 58%.", "growth: 58% year over year", "Did revenue grow?", True),
        (
            "Pestana Vintage hotel in Porto, Ribeira district.",
            "Pestana Vintage, Porto, Ribeira district, from 112 EUR",
            "Find a hotel",
            True,
        ),
        ("Sure, 2024 was busy.", "headcount 2024: 164,000", "What happened in 2024?", False),
        ("I could not find that, sorry.", "Amazon: 1,551,000 employees", "Who is the CEO?", False),
        ("It has 3 offices.", "offices: 3", "How many offices?", False),
        ("Try again on the 10th.", "date: 2026-10-03", "Book it", False),  # bare 2-digit number
        (  # JSON field names are not facts; only values are
            "What date and which airline and airport do you want?",
            '[{"departure_airport": "BOS", "airline": "Jet Blue", "date": "2026-10-03"}]',
            "Book a flight",
            False,
        ),
        (
            "Jet Blue flight from BOS on 2026-10-03.",
            '[{"departure_airport": "BOS", "airline": "Jet Blue", "date": "2026-10-03"}]',
            "Book a flight",
            True,
        ),
    ],
)
def test_unchecked_result_needs_claims_from_tool_output(
    answer: str, tool_output: str, user_request: str, fires: bool
) -> None:
    run = _answer_run(answer, tool_output, user_request)
    found = [f for f in run_rule_checks([run]) if f.check_id == "verification.unchecked_result"]
    assert bool(found) is fires
    if fires:
        assert "Taken from tool results:" in found[0].evidence_excerpts[1]
