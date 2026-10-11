"""Phase 0: fixture agents run offline, fast, deterministically, and exhibit their bug."""

from __future__ import annotations

import socket
import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from tests.fixtures.agents import AGENTS, run_agent

COMPLETING = [
    name for name in AGENTS if name != "supervisor_no_end" and not hasattr(AGENTS[name], "ERROR")
]


def _tool_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        call
        for message in result["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def _dump(result: dict[str, Any]) -> Any:
    return {
        key: [(type(m).__name__, m.content, getattr(m, "tool_calls", None)) for m in value]
        if key == "messages"
        else value
        for key, value in result.items()
    }


def test_network_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="Network access"):
        socket.create_connection(("example.com", 80))


def test_every_agent_declares_its_bug() -> None:
    assert {m.BUG for m in AGENTS.values()} == {
        "design.loop",
        "agents.ignored_output",
        "design.no_stop",
        "tools.errors_retries",
        "tools.unused_output",
        "design.empty_answer",
        "design.run_error",
        None,
    }


@pytest.mark.parametrize("name", list(AGENTS))
def test_agent_runs_under_one_second(name: str) -> None:
    start = time.perf_counter()
    try:
        run_agent(name)
    except GraphRecursionError:
        assert name == "supervisor_no_end"
    except Exception as error:
        assert isinstance(error, getattr(AGENTS[name], "ERROR", ()))
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize("name", COMPLETING)
def test_agent_is_deterministic(name: str) -> None:
    assert _dump(run_agent(name)) == _dump(run_agent(name))


def test_researcher_repeats_identical_search() -> None:
    calls = _tool_calls(run_agent("researcher_repeat_search"))
    assert len(calls) == 3
    assert len({(c["name"], str(c["args"])) for c in calls}) == 1


def test_writer_drops_caveat() -> None:
    from tests.fixtures.agents.pipeline_dropped_caveat import CAVEAT

    result = run_agent("pipeline_dropped_caveat")
    assert CAVEAT in result["analysis"]
    assert "12 customers" not in result["report"]


def test_supervisor_hits_recursion_limit() -> None:
    with pytest.raises(GraphRecursionError):
        run_agent("supervisor_no_end")


def test_tool_fails_and_is_retried_with_same_args() -> None:
    result = run_agent("tool_retry_same_args")
    calls = _tool_calls(result)
    errors = [m for m in result["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(calls) == 3
    assert len({str(c["args"]) for c in calls}) == 1
    assert len(errors) == 3


def test_tool_output_is_ignored() -> None:
    result = run_agent("tool_output_unused")
    tool_output = next(m for m in result["messages"] if isinstance(m, ToolMessage)).content
    final = result["messages"][-1].content
    assert "CANCELLED" in tool_output
    assert "cancel" not in final.lower()
    assert "refund" not in final.lower()


def test_healthy_agent_searches_once_and_verifies() -> None:
    result = run_agent("healthy")
    assert len(_tool_calls(result)) == 1
    assert result["messages"][-1].content.startswith("APPROVED")


def test_fake_model_reports_token_usage() -> None:
    result = run_agent("healthy")
    usages = [m.usage_metadata for m in result["messages"] if isinstance(m, AIMessage)]
    assert all(u and u["input_tokens"] > 0 and u["output_tokens"] > 0 for u in usages)


def test_swarm_handoff_drops_the_other_tool_result_and_the_run_fails() -> None:
    from tests.fixtures.agents._fake import InvalidRequestError

    with pytest.raises(InvalidRequestError, match=r"without `tool_result` blocks.*toolu_1"):
        run_agent("swarm_parallel_handoff")
