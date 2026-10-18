"""Made-up results: a final answer that states facts (flight numbers, routes, dates, IDs,
numbers) that contradict or don't appear in any tool result is a serious problem, "Answer
contradicts what the tools returned", not a low-severity "not checked" finding."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.builders import ai, build_run, step
from tests.helpers import SOURCES, load_fixture
from tracedoc.analyze import analyze
from tracedoc.checks.base import run_rule_checks
from tracedoc.checks.rules.contradicted_result import ContradictedResult, made_up_facts
from tracedoc.config import Config
from tracedoc.report import write_reports
from tracedoc.report.view import confidence_level, is_serious

CHECK = "verification.contradicted_result"
FLIGHTS = (
    '[{"flight": "UA 1532", "from": "BOS", "to": "ORD", "date": "2026-06-03", '
    '"departs": "08:05", "price": "$214", "ref": "K7Q2LM"}]'
)


@pytest.mark.parametrize("source", SOURCES)
def test_made_up_flight_is_a_serious_problem(source: str) -> None:
    run = load_fixture("made_up_flight", source)
    (f,) = ContradictedResult(Config()).run([run])
    assert f.severity == "high" and confidence_level(f) == "high" and is_serious(f)
    assert f.node == "flight_agent"
    assert f.explanation == (
        "flight_agent's final answer states UA 1523, LAX, June 4, 09:15, 199, which no tool "
        "returned; the tools returned AA 301, UA 1532, June 3, 08:05, 11:40, 214."
    )


def test_it_replaces_the_low_severity_not_checked_finding(tmp_path: Path) -> None:
    run = load_fixture("made_up_flight")
    checks = {f.check_id for f in run_rule_checks([run])}
    assert CHECK in checks and "verification.unchecked_result" not in checks
    files = write_reports(analyze([run], Config()), [run], tmp_path)
    assert files.view.cards[0].title == "Answer contradicts what the tools returned"
    assert files.view.verdict.startswith("1 of 1 run had a serious problem.")


def _run(answer: str, tool_output: str = FLIGHTS, request: str = "Find a flight to Chicago."):
    return build_run(
        "r1",
        [
            step("node", "agent"),
            step("llm", "agent", output={"type": "ai", "content": "", "tool_calls": []}),
            step("node", "tools", role="tool_executor"),
            step("tool", "agent", name="search", input={"q": "x"}, output=tool_output),
            step("node", "agent"),
            step("llm", "agent", output=ai(answer)),
        ],
        final_answer=answer,
        input_preview=request,
    )


@pytest.mark.parametrize(
    "answer",
    [
        "UA1532 leaves BOS at 8:05 am on June 3 for $214; your reference is K7Q2LM.",
        "United flight UA 1532, BOS to ORD, 3 June 2026, 08:05, $214.",
        "It costs about $210 and the flight is in 2026, from the USA to Chicago.",
        "Two tickets on UA 1532 come to $428.",  # computed: 2 x $214
    ],
)
def test_facts_the_tools_returned_are_fine(answer: str) -> None:
    assert made_up_facts(_run(answer)) == ([], [])


def test_facts_from_the_request_are_fine() -> None:
    run = _run("Booked for June 9 as you asked.", request="Book something for June 9.")
    assert made_up_facts(run) == ([], [])


def test_a_made_up_booking_reference_is_caught() -> None:
    made_up, returned = made_up_facts(_run("Done! Your booking reference is ZQ84TT."))
    assert made_up == ["ZQ84TT"]
    assert returned == ["K7Q2LM"]


def test_a_made_up_route_and_date_are_caught() -> None:
    made_up, _ = made_up_facts(_run("UA 1532 flies BOS to LAX on June 5."))
    assert made_up == ["LAX", "June 5"]


def test_numbers_no_tool_returned_are_caught() -> None:
    made_up, returned = made_up_facts(_run("UA 1532 costs $189 and has 140 seats left."))
    assert made_up == ["140", "189"]
    assert "214" in returned


def test_runs_without_tools_or_answers_are_not_checked() -> None:
    no_tool = build_run(
        "r1",
        [step("node", "agent"), step("llm", "agent", output=ai("UA 9999 to LAX."))],
        final_answer="UA 9999 to LAX.",
    )
    assert made_up_facts(no_tool) == ([], [])
    unanswered = _run("")
    assert made_up_facts(unanswered) == ([], [])


def test_other_fixtures_are_not_flagged() -> None:
    from tests.fixtures.agents import AGENTS

    for name in AGENTS:
        if name == "made_up_flight":
            continue
        for source in SOURCES:
            assert made_up_facts(load_fixture(name, source))[0] == [], (name, source)


@pytest.mark.parametrize(
    ("tool_date", "answer", "made_up", "returned"),
    [
        ("2026-10-03", "UA 1532 leaves on October 5.", "October 5", "October 3"),
        ("2026-11-20", "UA 1532 leaves on 21 November.", "November 21", "November 20"),
        ("2026-12-25", "UA 1532 leaves on Dec 24.", "December 24", "December 25"),
    ],
)
def test_dates_late_in_the_year_are_written_with_their_own_month(
    tool_date: str, answer: str, made_up: str, returned: str
) -> None:
    run = _run(answer, tool_output=FLIGHTS.replace("2026-06-03", tool_date))
    assert made_up_facts(run) == ([made_up], [returned])


def test_every_month_has_its_own_name() -> None:
    from tracedoc.checks.rules.contradicted_result import _fmt

    names = [_fmt("dates", f"{m:02d}-01").split()[0] for m in range(1, 13)]
    assert names == [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]  # fmt: skip
