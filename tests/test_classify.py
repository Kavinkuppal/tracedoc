from __future__ import annotations

import pytest

from tracedoc.classify.roles import map_role, role_from_name
from tracedoc.classify.task_type import classify_task


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("supervisor", "supervisor"),
        ("TeamManager", "supervisor"),
        ("research_agent", "researcher"),
        ("web-searcher", "researcher"),
        ("draftWriter", "writer"),
        ("plan_step", "planner"),
        ("reflect", "critic"),
        ("fact_checker", "critic"),
        ("tools", "tool_executor"),
        ("agent", None),
        ("support_agent", None),
    ],
)
def test_role_from_name(name: str, role: str | None) -> None:
    assert role_from_name(name) == role


def test_structure_beats_name() -> None:
    assert map_role("writer", only_runs_tools=True) == "tool_executor"


def test_model_fallback_only_when_name_unknown() -> None:
    calls = []

    def fallback(name: str) -> str:
        calls.append(name)
        return "planner"

    assert map_role("writer", fallback=fallback) == "writer"
    assert map_role("agent", fallback=fallback) == "planner"
    assert calls == ["agent"]


def test_invalid_fallback_answer_becomes_other() -> None:
    assert map_role("agent", fallback=lambda _: "wizard") == "other"


@pytest.mark.parametrize(
    ("text", "task"),
    [
        ("Fix the bug in this Python function", "coding"),
        ("Where is my order #123? I want a refund", "customer_support"),
        ("Extract the invoice fields as JSON", "data_extraction"),
        ("Plan a 3-day itinerary for Lisbon", "planning"),
        ("Write a blog post about solar panels", "writing"),
        ("What are the best sources on battery chemistry?", "research"),
    ],
)
def test_task_heuristics(text: str, task: str) -> None:
    assert classify_task(text) == task


def test_task_fallback_gets_truncated_preview() -> None:
    seen = []

    def fallback(preview: str) -> str:
        seen.append(preview)
        return "coding"

    assert classify_task("x" * 2000, fallback=fallback, max_chars=100) == "coding"
    assert seen == ["x" * 100]
    assert classify_task("zzz") == "other"
