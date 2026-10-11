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


# ---- labeled task-type sets ----------------------------------------------------------------
# task_types.jsonl was used to write the rules (a regression guard). task_types_holdout.jsonl
# was written before the rules were scored on it and must not be used to tune them: it is the
# honest estimate of how the keyword rules do on new requests (16/21 when it was added).

import json  # noqa: E402
from pathlib import Path  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _accuracy(name: str) -> tuple[float, list[str]]:
    rows = [json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line]
    misses = [
        f"want {r['label']}, got {classify_task(r['text'])}: {r['text']}"
        for r in rows
        if classify_task(r["text"]) != r["label"]
    ]
    return 1 - len(misses) / len(rows), misses


def test_labeled_task_types_regression() -> None:
    accuracy, misses = _accuracy("task_types.jsonl")
    assert accuracy >= 0.95, misses


def test_labeled_task_types_holdout() -> None:
    accuracy, misses = _accuracy("task_types_holdout.jsonl")
    assert accuracy >= 0.75, misses


def test_every_task_type_is_labeled() -> None:
    rows = [json.loads(line) for line in (FIXTURES / "task_types.jsonl").read_text().splitlines()]
    from tracedoc.schema import TASK_TYPES

    assert {r["label"] for r in rows} == set(TASK_TYPES)


@pytest.mark.parametrize(
    ("text", "task"),
    [
        ("Book me a flight from Boston to New York tomorrow.", "customer_support"),
        ("What's 17.5 multiplied by 24?", "other"),
        ("What was the combined headcount of the big tech firms in 2024?", "research"),
        ("Write a SQL query for monthly revenue.", "coding"),
        ("Draft a product roadmap.", "planning"),
    ],
)
def test_boundary_cases(text: str, task: str) -> None:
    assert classify_task(text) == task
