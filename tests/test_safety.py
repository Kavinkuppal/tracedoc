"""Suggested fixes never tell an agent to take an action with real-world effects (booking,
buying, sending, deleting) without the user's confirmation."""

from __future__ import annotations

import re

import pytest

from tests.fakes import FakeLLM, always, verdict
from tests.helpers import load_fixture
from tracedoc.catalog import catalog, fill, label, placeholders
from tracedoc.checks.base import Check
from tracedoc.checks.judges.prompts import judge_prompts
from tracedoc.checks.judges.runner import Judge
from tracedoc.checks.judges.slots import fit_fix, fitted
from tracedoc.checks.judges.windows import run_end_window
from tracedoc.checks.safety import safe_fix, unconfirmed_action
from tracedoc.config import Config

UNSAFE = [
    'Add to flight_agent\'s prompt: "Book the cheapest flight right away."',
    "Have hotel_agent book the hotel without asking the user.",
    "Make the agent send the confirmation email to the customer.",
    "Let cleanup_agent delete the user's old files automatically.",
    'Add to support\'s prompt: "Cancel the order as soon as the customer complains."',
    "Tell the agent to place the order instead of asking.",
    "Add to shopper's prompt: \"Buy it; don't ask for confirmation.\"",
    "Have payments transfer the money to the vendor.",
    'Add to flight_agent: "Book the flight; ask the user later if needed."',
    "Have the agent pay the invoice once the tool returns it.",
    # Found in review: other forms of the verbs, tool names, and nouns with a verb.
    "Add to flight_assistant's prompt: \"When the user gives a flight ID, call book_flight with "
    'it directly instead of asking again."',
    "Make flight_agent proceed with booking the flight without waiting.",
    "Have hotel_agent complete the reservation as soon as a hotel is found.",
    "Have the support agent issue refunds automatically for orders under $50.",
    # A real AI judge fix (report on LangChain's customer support example): it books, then tells
    # the user afterwards, which is not asking first.
    "Add to hotel_assistant's prompt: \"After a handoff, search_hotels and book_hotel for New "
    'York, then confirm both bookings to the user." Also make sure hotel_assistant has those '
    "tools bound.",
    "Have flight_assistant call book_flight right after search_flights returns a match.",
    "Have the agent call cancel_order with the order ID right away.",
    "Have billing call pay_invoice as soon as the invoice arrives.",
    "Have notifier call send_email with the summary.",
    "Have cleanup call delete_records for stale accounts.",
    "Have support call refund_order when the customer complains.",
    "Make sure the flight is booked before handing off to hotel_assistant.",
    "Have the agent get the flight booked in the same turn.",
    "Have the agent that books flights do it immediately.",
    "Have flight_agent finish the booking in the same turn.",
    "Have the agent make the payment once the invoice total is known.",
    "Book the hotel, then tell the user it is booked.",
    "Book the hotel and then ask the user to confirm the dates.",
    "Have the agent process the refund and notify the customer afterwards.",
    "Don't ask before booking; the user already said yes.",
    "Book the flight now; ask the user to confirm before booking the hotel.",
    "Book the flight, then ask the user to confirm before charging the card.",
]
SAFE = [
    'Add to flight_agent\'s prompt: "Before booking, show the flight and ask the user to confirm."',
    "Have hotel_agent book the room only after the user confirms the dates.",
    "Add to shopper's prompt: \"Ask for the user's approval, then place the order.\"",
    "Send supervisor's model less: trim old messages and unused state from its prompt.",
    "Add a graph edge that sends an empty reply back to supervisor.",
    "Make book_flight return an error message instead of raising.",
    "Route the answer through critic, which is in charge of checking it.",
    "Have the agent pay attention to the tool result.",
    "Add an exit condition: cancel the loop when a cycle produced nothing new.",
    "Never book a flight without asking the user first.",
    "Book only after the user confirms; never book without asking.",
    "Only call book_flight after the user confirms the flight.",
    "Show the booking details and wait for the user's confirmation before calling book_flight.",
    'Add to hotel_assistant\'s prompt: "Ask the user to confirm the hotel, then call book_hotel."',
    "Have the agent book the flight with the user's explicit approval.",
    # Asking first can be its own sentence, after the one that names the action.
    'Add to flight_agent\'s prompt: "Book the cheapest matching flight." Ask the user to confirm '
    "the flight before calling book_flight.",
    "Book the flight. Ask the user to confirm before booking.",
    # A tool being changed, limited or described, not called.
    "Add input validation for book_flight's arguments.",
    "Cap how many times flight_agent can call book_flight in one run.",
    "Never tell the user a flight is booked unless book_flight returned a confirmation number.",
    "Make sure the flight isn't booked twice: check the reservation, and ask the user to confirm.",
    "Have flight_agent check the booking status before answering.",
    "Have the agent wire the critic node into the graph after the last tool call.",
    "Read the file with the orders before answering.",
]


@pytest.mark.parametrize("fix", UNSAFE)
def test_unconfirmed_real_world_actions_are_caught(fix: str) -> None:
    assert unconfirmed_action(fix), fix


@pytest.mark.parametrize("fix", SAFE)
def test_confirmed_or_internal_actions_pass(fix: str) -> None:
    assert unconfirmed_action(fix) is None, fix


def _templates() -> list[tuple[str, str]]:
    found = []

    def walk(entry: dict, where: str) -> None:
        for key, value in entry.items():
            if isinstance(value, dict):
                walk(value, f"{where}.{key}")
            elif key.startswith("what_to_change"):
                found.append((f"{where}.{key}", value))

    for check_id, entry in catalog()["checks"].items():
        walk(entry, check_id)
    return found


@pytest.mark.parametrize(("where", "template"), _templates())
def test_every_fix_in_the_catalog_passes(where: str, template: str) -> None:
    values = {
        name: "book_flight" if "tool" in name else "agent_x" for name in placeholders(template)
    }
    assert unconfirmed_action(fill(template, **values)) is None, where


def test_the_replacement_fix_passes_its_own_rule() -> None:
    fixed = safe_fix("Book the flight now.", "flight_agent")
    assert fixed == label("fix_confirm_first", agent="flight_agent")
    assert unconfirmed_action(fixed) is None
    assert safe_fix("Cache search results by their arguments.", "x") == (
        "Cache search results by their arguments."
    )


def test_a_rule_finding_never_carries_an_unsafe_fix(monkeypatch) -> None:
    class Booker(Check):
        id = "design.empty_answer"
        group = "design"
        method = "rule"
        default_severity = "high"

        def run(self, runs):
            return []

    monkeypatch.setattr(
        "tracedoc.checks.base.check_templates",
        lambda *_: ("{agent} stopped.", "Have {agent} book the flight right away."),
    )
    run = load_fixture("nested_agents")
    f = Booker(Config()).finding(run, facts={}, node="supervisor")
    assert f.suggested_fix == label("fix_confirm_first", agent="supervisor")


UNSAFE_FIX = 'Add to supervisor\'s prompt: "Book the flight as soon as research_agent finds it."'
SAFE_FIX = 'Add to supervisor\'s prompt: "Show the flight and ask the user to confirm first."'


def _judge(confirm) -> tuple[Judge, str]:
    prompt = judge_prompts()["verification.premature_finish"]
    judge = Judge(Config(), {prompt.id: prompt}, client=FakeLLM(always(True), confirm))
    window = run_end_window(load_fixture("nested_agents"), 6).render(400)
    return judge, window


def test_judge_fix_that_acts_without_asking_is_sent_back(tmp_path) -> None:
    prompts: list[str] = []

    def confirm(prompt: str, check_id: str):
        prompts.append(prompt)
        retry = "YOUR PREVIOUS ANSWER" in prompt
        return verdict(
            True, 0.9, ["S7"], agent="supervisor", change=SAFE_FIX if retry else UNSAFE_FIX
        )

    judge, window = _judge(confirm)
    (answer,) = judge.judge_text(window, list(judge.judges.values())).values()
    assert answer.what_to_change == SAFE_FIX
    assert len(prompts) == 4  # two reviews, each sent back once
    assert re.search(r"tells an agent to 'Book' without the user's confirmation", prompts[1])


def test_judge_fix_still_unsafe_after_the_retry_is_replaced() -> None:
    def confirm(prompt: str, check_id: str):
        return verdict(True, 0.9, ["S7"], agent="supervisor", change=UNSAFE_FIX)

    judge, window = _judge(confirm)
    (answer,) = judge.judge_text(window, list(judge.judges.values())).values()
    assert answer.what_to_change == label("fix_confirm_first", agent="supervisor")


# ---- shortening a fix never drops the sentence that asks first -----------------------------

ASKS_LAST = (
    'Add to flight_agent\'s prompt: "Book the flight the user picked." Then show the details '
    "{pad}and ask the user to confirm first"
)


@pytest.mark.parametrize("pad", range(0, 120, 7))
def test_the_page_1_text_keeps_the_sentence_that_asks_first(pad: int) -> None:
    """The judge's fix is fitted to 200 characters, then page 1 fits it again (with a final
    period added): neither may drop "ask the user to confirm"."""
    from types import SimpleNamespace

    from tracedoc.report.view import _sentence

    fix = ASKS_LAST.format(pad="x" * pad + " " if pad else "")
    answer = SimpleNamespace(
        what_happened="x", what_to_change=fix, evidence_quote="", agent="flight_agent"
    )
    stored = fitted(answer, "")["what_to_change"]
    card = fit_fix(_sentence(stored), 200, "flight_agent")
    assert len(card) <= 200
    assert unconfirmed_action(card) is None
    assert "ask the user to confirm" in card or card == label(
        "fix_confirm_first", agent="flight_agent"
    )


def test_an_earlier_sentence_is_shortened_instead() -> None:
    fix = (
        "Add to flight_agent's prompt: \"Book the cheapest flight that matches the dates, the "
        "airline and the seat class the user asked for, using only what search_flights "
        'returned." Ask the user to confirm the flight before calling book_flight. Also log '
        "every search."
    )
    short = fit_fix(fix, 200, "flight_agent")
    assert len(short) <= 200 and len(fix) > 200
    assert short.endswith("Ask the user to confirm the flight before calling book_flight.")
    assert short.startswith("Add to flight_agent's prompt: \"Book the cheapest flight")
    assert '…" Ask the user' in short  # the shortened quote is closed
    assert "Also log every search." not in short  # sentences after the asking go first


def test_sentences_after_the_asking_are_dropped_first() -> None:
    fix = (
        "Have hotel_agent show the hotel and ask the user to confirm before calling book_hotel. "
        "Then log the booking reference in the trip summary so later agents can read it, and "
        "add the hotel's address and check-in time to the final answer for the user."
    )
    assert fit_fix(fix, 120, "hotel_agent") == (
        "Have hotel_agent show the hotel and ask the user to confirm before calling book_hotel."
    )


def test_a_fix_that_shortening_would_make_unsafe_is_replaced() -> None:
    # The confirmation is the end of one long sentence: no earlier sentence to shorten.
    fix = (
        "Have flight_agent book the cheapest flight that matches the user's dates, airline, "
        "seat class and departure airport from the search results, but only after the user "
        "confirms it."
    )
    assert unconfirmed_action(fix) is None
    assert fit_fix(fix, 120, "flight_agent") == label("fix_confirm_first", agent="flight_agent")


def test_the_card_on_page_1_keeps_the_confirmation(tmp_path) -> None:
    from tests.helpers import load_fixture
    from tracedoc.analyze import analyze
    from tracedoc.report import write_reports

    long_fix = (
        "Add to supervisor's prompt: \"Repeat research_agent's answer to the user in full, "
        'with its sources and numbers, before ending the run." Then ask the user to confirm '
        "before booking anything the answer suggests."
    )
    assert len(long_fix) > 200

    def confirm(prompt: str, check_id: str):
        return verdict(True, 0.9, ["S7"], agent="supervisor", change=long_fix)

    config = Config()
    config.judge.include_unvalidated = True
    run = load_fixture("nested_agents")
    report = analyze([run], config, client=FakeLLM(always(True), confirm))
    files = write_reports(report, [run], tmp_path)
    fixes = [f.suggested_fix for g in report.groups for f in g.findings if f.method == "judge"]
    assert fixes and all(unconfirmed_action(f) is None for f in fixes)
    assert all("ask the user to confirm" in f for f in fixes)
    for card in files.view.cards:
        assert unconfirmed_action(card.what_to_change) is None
