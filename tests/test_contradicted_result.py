"""Made-up results: a final answer that states facts (flight numbers, routes, dates, IDs,
numbers) no tool result explains is a candidate for "Answer contradicts what the tools
returned". Quick filters clear what the given numbers and names explain; the AI judge confirms
the rest, and an unconfirmed candidate stays low confidence, off page 1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.builders import ai, build_run, step
from tests.fakes import FakeLLM, always
from tests.helpers import SOURCES, load_fixture
from tracedoc.analyze import analyze
from tracedoc.checks.base import run_rule_checks
from tracedoc.checks.judges.runner import Confirmation
from tracedoc.checks.rules.contradicted_result import ContradictedResult, made_up_facts
from tracedoc.config import Config
from tracedoc.report import write_reports
from tracedoc.report.view import confidence_level, is_serious
from tracedoc.schema import Run

CHECK = "verification.contradicted_result"
FLIGHTS = (
    '[{"flight": "UA 1532", "from": "BOS", "to": "ORD", "date": "2026-06-03", '
    '"departs": "08:05", "price": "$214", "ref": "K7Q2LM"}]'
)
LABELS = Path(__file__).parents[1] / "evals" / "labels" / f"confirm.{CHECK}.jsonl"


def _confirming(confirmed: bool):
    """A fake judge that confirms (or rejects) every rule candidate."""

    def confirm(prompt: str, check_id: str):
        return Confirmation(verdict=confirmed, confidence=0.9, evidence_step_ids=["S1"])

    return FakeLLM(triage=always(True), confirm=confirm)


@pytest.mark.parametrize("source", SOURCES)
def test_made_up_flight_is_a_candidate_until_the_judge_confirms_it(source: str) -> None:
    run = load_fixture("made_up_flight", source)
    (f,) = ContradictedResult(Config()).run([run])
    assert f.severity == "high" and f.needs_confirmation
    assert confidence_level(f) == "low" and not is_serious(f)  # never on page 1 unconfirmed
    assert f.node == "flight_agent"
    assert f.explanation == (
        "flight_agent's final answer states UA 1523, LAX, June 4, 09:15, $199, which no tool "
        "returned; the tools returned AA 301, UA 1532, June 3, 08:05, 11:40, $214."
    )


def test_confirmed_it_is_serious_and_on_page_1(tmp_path: Path) -> None:
    run = load_fixture("made_up_flight")
    config = Config()
    report = analyze([run], config, client=_confirming(True))
    (f,) = [f for g in report.groups for f in g.findings if f.check_id == CHECK]
    assert f.confirmed_by_judge and is_serious(f)
    files = write_reports(report, [run], tmp_path)
    assert files.view.cards[0].title == "Answer contradicts what the tools returned"
    assert files.view.verdict.startswith("1 of 1 run had a serious problem.")


def test_rejected_or_unreviewed_it_never_reaches_page_1(tmp_path: Path) -> None:
    run = load_fixture("made_up_flight")
    rejected = analyze([run], Config(), client=_confirming(False))
    assert CHECK not in {f.check_id for g in rejected.groups for f in g.findings}
    rules_only = analyze([run], Config())
    files = write_reports(rules_only, [run], tmp_path)
    assert not files.view.cards
    (f,) = [f for g in rules_only.groups for f in g.findings if f.check_id == CHECK]
    assert confidence_level(f) == "low"


def test_candidates_go_straight_to_confirmation_without_triage() -> None:
    """The filters already did what triage would, and the triage model doesn't redo the
    arithmetic (it dismissed a wrong average and a wrong total), so there is no triage."""
    from tracedoc.checks.judges.prompts import confirmation_prompts
    from tracedoc.checks.judges.runner import REVIEWS, Judge

    assert not confirmation_prompts()[CHECK].triage
    run = load_fixture("made_up_flight")
    fake = _confirming(True)
    analyze([run], Config(), client=fake)
    asked = [schema for (_, schema), p in zip(fake.calls, fake.prompts, strict=True) if CHECK in p]
    assert asked == ["Confirmation"] * REVIEWS  # no triage call about it
    judge = Judge(Config(), {}, client=None)
    candidate = [f for f in run_rule_checks([run]) if f.check_id == CHECK]
    estimate = judge.estimate(judge.plan([run], candidate))
    assert estimate.triage_calls == 0
    assert estimate.confirm_calls_expected == estimate.confirm_calls_max == REVIEWS


def test_with_the_not_checked_candidate_at_the_same_step_it_is_one_finding() -> None:
    run = _run("UA 1532 leaves at 08:05 for $214; seat 14C is $40 more, so $254 in all.")
    checks = {f.check_id for f in run_rule_checks([run])}
    assert {CHECK, "verification.unchecked_result"} <= checks
    report = analyze([run], Config(), client=_confirming(True))
    (f,) = [f for g in report.groups for f in g.findings]
    assert f.check_id == CHECK  # the more specific of the two
    assert [r.check_id for r in f.related] == ["verification.unchecked_result"]


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
    assert made_up == ["$189", "140"]
    assert "$214" in returned


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


# ---- the fourteen cases from the first test on real traces ----------------------------------
# Each case keeps the pattern of a run the first version of this check flagged, rewritten with
# other companies, cities and numbers: eight wrong flags and six real problems. The quick
# filters must clear seven of the wrong flags; what is left goes to the AI judge, and the
# labeled examples in evals/labels/confirm.verification.contradicted_result.jsonl are these
# same cases as the judge sees them.

HEADCOUNTS = (
    "Headcounts in 2025: 1. Northwind: 48,250 employees. 2. Contoso: 212,400 employees. "
    "3. Fabrikam: 9,800 employees. 4. Tailspin: 1,284,000 employees. 5. Litware: 131,600 "
    "employees."
)
SUPERVISOR_PROMPT = (
    "You coordinate two specialists: research_expert looks things up, math_expert calculates. "
    "Route each request to the right one, then answer the user."
)
RESEARCH_PROMPT = "You find facts with web_search. Leave calculations to math_expert."
MATH_PROMPT = "You calculate with the add and multiply tools."
FLIGHT_PROMPT = "You search and book flights for the user."


@dataclass(frozen=True)
class Case:
    id: str
    real: bool
    """A real problem (the judge should confirm it), or a wrong flag (it should not)."""
    note: str
    request: str
    answer: str
    research: str = ""
    """What research_expert told the supervisor (supervisor cases)."""
    flight: str = ""
    """What search_flights returned (flight cases)."""
    math: tuple[str, str, str] | None = None
    """(tool, arguments, result) of one math_expert call, for arithmetic requests."""
    first_flagged: str = ""
    """What the first version of the check listed, for wrong flags the filters now clear."""

    @property
    def filtered(self) -> bool:
        """The quick filters clear it: it never reaches the judge."""
        return bool(self.first_flagged)


CASES = (
    Case(
        "percent-larger",
        False,
        "wrong flag: a percentage difference computed from two tool values",
        "Is Contoso bigger than Litware by headcount, and by how much?",
        "Yes. Contoso has 212,400 employees and Litware 131,600, so Contoso is bigger by "
        "80,800 employees, about 61.4% larger.",
        research=HEADCOUNTS,
        first_flagged="61.4%",
    ),
    Case(
        "date-offered",
        False,
        "wrong flag: a date offered in a question, not stated as fact",
        "Can you find me a flight from Denver to Atlanta next Thursday?",
        "The search only returned a San Francisco to Seattle flight, which doesn't match your "
        "trip. Would you like me to search again, or try a day around May 21st instead?",
        flight='{"departure_airport": "SFO", "arrival_airport": "SEA", "airline": "Alaska", '
        '"date": "2026-05-15", "id": "7"}',
    ),
    Case(
        "airport-code",
        False,
        "wrong flag: the airport code of a city the user named",
        "Find me a flight from ORD to Phoenix on March 12th.",
        "The search returned an MSP to DTW flight on March 2nd instead of ORD to PHX on March "
        "12th. Would you like me to search again?",
        flight='{"departure_airport": "MSP", "arrival_airport": "DTW", "airline": "Sun Country", '
        '"date": "2026-03-02", "id": "1"}',
        first_flagged="PHX",
    ),
    Case(
        "sums-and-millions",
        False,
        "wrong flag: sums of tool values and a value rounded to millions",
        "What is the combined 2025 headcount of the five companies?",
        "The combined headcount is about 1,686,050 employees. Tailspin dominates with 1.28 "
        "million, while the other four companies together have 402,050.",
        research=HEADCOUNTS,
        first_flagged="1,686,050, 402,050, 1.28",
    ),
    Case(
        "shown-working",
        False,
        "wrong flag: correct working the answer shows",
        "What is 12.5 x 36?",
        "The answer is **450**. Here's how: - 12.5 x 36 = (12 + 0.5) x 36 - = (12 x 36) + "
        "(0.5 x 36) - = 432 + 18 - = **450**",
        math=("multiply", '{"a": 12.5, "b": 36}', "450.0"),
        first_flagged="0.5, 432",
    ),
    Case(
        "multiple-plus-another",
        False,
        "wrong flag: a few of one tool value plus another",
        "Take Fabrikam's headcount, multiply it by 4, then add Northwind's.",
        "Result: 87,450. Fabrikam's 9,800 x 4 is 39,200, and adding Northwind's 48,250 gives "
        "87,450.",
        research=HEADCOUNTS,
        first_flagged="87,450",
    ),
    Case(
        "percent-smaller",
        False,
        "wrong flag: a percentage difference relative to the larger value",
        "How does Litware's headcount compare with Contoso's?",
        "Litware has 131,600 employees, about 38% fewer than Contoso's 212,400.",
        research=HEADCOUNTS,
        first_flagged="38%",
    ),
    Case(
        "term-not-id",
        False,
        "wrong flag: a term that looks like an ID, and a value rounded to millions",
        "What was Tailspin's headcount in 2021, and how has it changed since?",
        "Tailspin had about 1.28 million employees in 2025. The search has no 2021 figure, "
        "so I can't say how much it changed; its hiring around COVID-19 isn't in the data.",
        research=HEADCOUNTS,
        first_flagged="COVID-19, 1.28",
    ),
    Case(
        "return-flight",
        True,
        "real: a return flight the search never returned",
        "Book a round trip from Denver to Atlanta, leaving tomorrow and returning in 4 days.",
        "Here's what's available. Outbound: flight ID 3, Delta, DEN to ATL, May 15, 2026. "
        "Return: flight ID 3, Delta, ATL to DEN, May 19, 2026. Shall I book both?",
        flight='{"departure_airport": "DEN", "arrival_airport": "ATL", "airline": "Delta", '
        '"date": "2026-05-15", "id": "3"}',
    ),
    Case(
        "headcount-from-memory",
        True,
        "real: a headcount from general knowledge presented as research",
        "What was Initech's headcount in 2025?",
        "Based on the research expert's findings, Initech's headcount in 2025 was "
        "approximately 30,000-35,000 employees, per its annual report for the fiscal year "
        "ending September 30.",
        research="The search results only cover Northwind, Contoso, Fabrikam, Tailspin and "
        "Litware. From general knowledge, Initech has roughly 30,000-35,000 employees, but I "
        "couldn't verify it.",
    ),
    Case(
        "wrong-average",
        True,
        "real: an average that doesn't follow from the tool values",
        "What's the average 2025 headcount across the five companies?",
        "The average headcount across the five companies in 2025 was approximately 537,210 "
        "employees.",
        research=HEADCOUNTS,
    ),
    Case(
        "revenue-from-memory",
        True,
        "real: a revenue figure no tool returned, presented as research",
        "What was Contoso's revenue in 2025?",
        "Based on the research expert's response, Contoso's revenue in 2025 was approximately "
        "$61.2 billion, up from the year before.",
        research=HEADCOUNTS,
    ),
    Case(
        "small-headcount-from-memory",
        True,
        "real: a headcount no tool returned, presented as research",
        "How many people work at Hooli?",
        "Based on the research, Hooli has approximately 2,000 employees as of 2025.",
        research=HEADCOUNTS,
    ),
    Case(
        "wrong-share",
        True,
        "real: a wrong total and the share computed from it",
        "What share of the five companies' total 2025 headcount does Tailspin have?",
        "Tailspin represents approximately 69.8% of the total, with 1,284,000 employees out of "
        "a combined 1,839,450.",
        research=HEADCOUNTS,
    ),
)


def _system(prompt: str, request: str, *more: dict) -> list[dict]:
    return [{"type": "system", "content": prompt}, {"type": "human", "content": request}, *more]


def case_run(case: Case) -> Run:
    """The case as a run of the system it comes from."""
    request = case.request
    if case.flight:
        specs = [
            step("node", "flight_agent"),
            step(
                "llm",
                "flight_agent",
                input=_system(FLIGHT_PROMPT, request),
                output={
                    "type": "ai",
                    "content": "",
                    "tool_calls": [{"name": "search_flights", "args": {}}],
                },
            ),
            step(
                "tool",
                "flight_agent",
                name="search_flights",
                input={"query": request},
                output=f"[{case.flight}]",
            ),
            step(
                "llm",
                "flight_agent",
                input=_system(FLIGHT_PROMPT, request, {"type": "tool", "content": case.flight}),
                output=ai(case.answer),
            ),
        ]
    else:
        expert, prompt = (
            ("math_expert", MATH_PROMPT)
            if case.math
            else (
                "research_expert",
                RESEARCH_PROMPT,
            )
        )
        tool, args, result = case.math or ("web_search", json.dumps({"query": request}), "")
        result = result or HEADCOUNTS
        reply = case.research or result
        specs = [
            step("node", "supervisor"),
            step(
                "llm",
                "supervisor",
                input=_system(SUPERVISOR_PROMPT, request),
                output={
                    "type": "ai",
                    "content": "",
                    "tool_calls": [{"name": f"to_{expert}", "args": {}}],
                },
            ),
            step("handoff", "supervisor", handoff_from="supervisor", handoff_to=expert),
            step("node", expert),
            step(
                "llm",
                expert,
                input=_system(prompt, request),
                output={"type": "ai", "content": "", "tool_calls": [{"name": tool, "args": {}}]},
            ),
            step("tool", expert, name=tool, input=json.loads(args), output=result),
            step(
                "llm",
                expert,
                input=_system(prompt, request, {"type": "tool", "content": result}),
                output=ai(reply),
            ),
            step("handoff", expert, handoff_from=expert, handoff_to="supervisor", input=reply),
            step("node", "supervisor"),
            step(
                "llm",
                "supervisor",
                input=_system(
                    SUPERVISOR_PROMPT,
                    request,
                    {"type": "ai", "content": reply, "name": expert},
                ),
                output=ai(case.answer),
            ),
        ]
    return build_run(
        f"case-{case.id}",
        specs,
        final_answer=case.answer,
        input_preview=request,
        output_preview=case.answer,
    )


def test_the_cases_are_eight_wrong_flags_and_six_real_problems() -> None:
    assert sum(not c.real for c in CASES) == 8 and sum(c.real for c in CASES) == 6
    assert sum(c.filtered for c in CASES) == 7  # all wrong flags but the offered date


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_quick_filters_clear_wrong_flags_and_keep_real_problems(case: Case) -> None:
    made_up, _ = made_up_facts(case_run(case))
    if case.filtered:
        assert made_up == [], case.note
    else:
        assert made_up, case.note  # a candidate for the AI judge


def test_the_wrong_average_and_share_are_not_explained_by_chance() -> None:
    by_id = {c.id: c for c in CASES}
    assert made_up_facts(case_run(by_id["wrong-average"]))[0] == ["537,210"]
    assert made_up_facts(case_run(by_id["wrong-share"]))[0] == ["69.8%", "1,839,450"]


def test_the_labeled_examples_are_these_cases() -> None:
    examples = [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]
    assert [(e["id"], e["label"], e["note"]) for e in examples] == [
        (f"confirm.{CHECK}-{c.id}", c.real, c.note) for c in CASES
    ]
    for example, case in zip(examples, CASES, strict=True):
        assert case.answer in example["window"] and case.request in example["window"]
        assert example["rule_explanation"]
