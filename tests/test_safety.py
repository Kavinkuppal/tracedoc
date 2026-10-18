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
