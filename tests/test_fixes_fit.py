"""Suggested fixes fit the trace: no cheaper model when the agent already uses the cheapest one,
and nothing the trace shows is already in place."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.fakes import verdict
from tracedoc.checks.judges.slots import slot_problems
from tracedoc.checks.rules.empty_answer import EmptyAnswer
from tracedoc.checks.rules.token_hotspot import TokenHotspot
from tracedoc.checks.rules.unchecked_result import UncheckedResult
from tracedoc.config import Config
from tracedoc.schema import Run, Step, Totals

T0 = datetime(2026, 10, 2, 12, tzinfo=timezone.utc)


def _step(i: int, kind: str, node: str, **kw: object) -> Step:
    return Step(
        id=f"s{i}",
        run_id="r1",
        kind=kind,  # type: ignore[arg-type]
        name=str(kw.pop("name", kind)),
        node=node,
        started_at=T0 + timedelta(seconds=i),
        ended_at=T0 + timedelta(seconds=i + 1),
        **kw,  # type: ignore[arg-type]
    )


def _run(steps: list[Step], status: str = "completed", answer: str = "Done.") -> Run:
    return Run(
        id="r1",
        source="capture",
        status=status,  # type: ignore[arg-type]
        input_preview="What was the combined headcount?",
        final_answer=answer,
        started_at=T0,
        ended_at=T0 + timedelta(seconds=30),
        steps=steps,
        totals=Totals(cost=0.01, duration_s=30.0, step_count=len(steps)),
    )


def _hotspot_run(model: str) -> Run:
    return _run(
        [
            _step(0, "llm", "supervisor", model=model, tokens_in=900, tokens_out=100),
            _step(1, "llm", "research_expert", model=model, tokens_in=24_000, tokens_out=900),
            _step(2, "llm", "research_expert", model=model, tokens_in=26_000, tokens_out=900),
        ]
    )


def test_no_cheaper_model_advice_for_the_cheapest_model_in_its_family() -> None:
    (cheapest,) = TokenHotspot(Config()).run([_hotspot_run("claude-haiku-4-5-20251001")])
    assert cheapest.node == "research_expert"
    assert "cheaper model" not in cheapest.suggested_fix
    assert "trim old messages" in cheapest.suggested_fix

    (pricier,) = TokenHotspot(Config()).run([_hotspot_run("claude-sonnet-5-5")])
    assert "use a cheaper model for research_expert" in pricier.suggested_fix

    (unpriced,) = TokenHotspot(Config()).run([_hotspot_run("my-local-model")])
    assert "cheaper model" not in unpriced.suggested_fix  # nothing shows a cheaper one exists


def _empty_answer_run(system_prompt: str) -> Run:
    messages = [
        {"type": "system", "content": system_prompt},
        {"type": "human", "content": "What was the combined headcount?"},
    ]
    return _run(
        [_step(0, "llm", "supervisor", input=messages, output={"type": "ai", "content": ""})],
        status="no_answer",
        answer="",
    )


def test_instruction_already_in_the_prompt_is_not_suggested_again() -> None:
    (usual,) = EmptyAnswer(Config()).run([_empty_answer_run("You are a team supervisor.")])
    assert "Add to supervisor's prompt: \"Always end with a written answer" in usual.suggested_fix

    prompt = "You are a team supervisor. Always end with a written answer for the user!"
    (in_place,) = EmptyAnswer(Config()).run([_empty_answer_run(prompt)])
    assert "Always end with a written answer" not in in_place.suggested_fix
    assert in_place.suggested_fix.startswith(
        "supervisor's prompt already asks for a written answer"
    )
    assert "graph" in in_place.suggested_fix


def test_existing_critic_is_suggested_instead_of_a_new_one() -> None:
    tool_output = '{"Meta": "67,317", "Apple": "164,000", "Amazon": "1,551,000"}'
    steps = [
        _step(0, "node", "researcher", role="researcher"),
        _step(1, "node", "critic", role="critic"),
        _step(2, "tool", "researcher", name="web_search", output=tool_output),
        _step(3, "node", "researcher", role="researcher"),
        _step(4, "llm", "researcher", output={"type": "ai", "content": "Total: 1,782,634."}),
    ]
    answer = "Meta has 67,317, Apple 164,000 and Amazon 1,551,000 employees."
    (finding,) = UncheckedResult(Config()).run([_run(steps, answer=answer)])
    assert "Route researcher's final answer through critic" in finding.suggested_fix
    assert "critic agent after" not in finding.suggested_fix


def test_judge_fix_already_in_the_prompt_is_sent_back() -> None:
    window = (
        "USER TASK: What was the combined headcount?\n\nAGENT INSTRUCTIONS (system prompts):\n"
        "- supervisor: You are a team supervisor. Always use math_agent for arithmetic.\n"
        "\nSTEPS:\n[S1] MODEL CALL · agent=supervisor\n  output: The total is 1,977,586."
    )
    answer = verdict(
        True,
        what="supervisor added the numbers itself.",
        change='Add to supervisor\'s prompt: "Always use math_agent for arithmetic."',
        quote="The total is 1,977,586.",
    )
    assert any("already has" in p for p in slot_problems(answer, window))
    fresh = verdict(
        True,
        what="supervisor added the numbers itself.",
        change="Remove the add tool from supervisor so only math_agent can do arithmetic.",
        quote="The total is 1,977,586.",
    )
    assert slot_problems(fresh, window) == []
