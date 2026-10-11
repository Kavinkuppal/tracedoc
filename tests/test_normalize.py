"""Every fixture trace normalizes correctly from both sources."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.helpers import SOURCES, load_fixture

# name: (status, node sequence, handoffs, tokens_in, tokens_out, task_type)
EXPECTED = {
    "healthy": (
        "completed",
        ["researcher", "tools", "researcher", "writer", "critic"],
        [("researcher", "writer"), ("writer", "critic")],
        248,
        78,
        "research",
    ),
    "pipeline_dropped_caveat": (
        "completed",
        ["analyst", "writer"],
        [("analyst", "writer")],
        105,
        66,
        "research",
    ),
    "researcher_repeat_search": (
        "completed",
        ["researcher", "tools"] * 3 + ["researcher"],
        [],
        488,
        81,
        "research",
    ),
    "supervisor_no_end": (
        "recursion_limit",
        ["supervisor", "researcher", "supervisor", "writer"] * 3,
        [
            ("supervisor", "researcher"),
            ("researcher", "supervisor"),
            ("supervisor", "writer"),
            ("writer", "supervisor"),
        ]
        * 2
        + [("supervisor", "researcher"), ("researcher", "supervisor"), ("supervisor", "writer")],
        807,
        102,
        "writing",
    ),
    "tool_retry_same_args": (
        "completed",
        ["agent", "tools"] * 3 + ["agent"],
        [],
        308,
        46,
        "writing",
    ),
    "tool_output_unused": (
        "completed",
        ["support_agent", "tools", "support_agent"],
        [],
        56,
        29,
        "customer_support",
    ),
}
CASES = [(name, source) for name in EXPECTED for source in SOURCES]


def _agents_for(graph_nodes: list[str]) -> list[str]:
    """Steps of a shared "tools" node are credited to the agent that called the tools."""
    agents: list[str] = []
    for node in graph_nodes:
        agents.append(agents[-1] if node == "tools" and agents else node)
    return agents


@pytest.mark.parametrize(("name", "source"), CASES)
def test_status_nodes_tokens_task(name: str, source: str) -> None:
    status, nodes, _, tokens_in, tokens_out, task = EXPECTED[name]
    run = load_fixture(name, source)
    assert run.source == source
    assert run.status == status
    assert [s.graph_node for s in run.steps_of("node")] == nodes
    assert [s.node for s in run.steps_of("node")] == _agents_for(nodes)
    assert run.totals.step_count == len(nodes)
    assert run.totals.tokens_in == tokens_in
    assert run.totals.tokens_out == tokens_out
    assert run.task_type == task
    assert run.framework == "langgraph"


@pytest.mark.parametrize(("name", "source"), CASES)
def test_handoffs(name: str, source: str) -> None:
    expected = EXPECTED[name][2]
    run = load_fixture(name, source)
    handoffs = run.steps_of("handoff")
    assert [(h.handoff_from, h.handoff_to) for h in handoffs] == expected
    nodes = {s.id: s for s in run.steps_of("node")}
    for h in handoffs:
        assert h.input is not None and h.output is not None
        assert h.started_at <= h.ended_at
        assert h.node == h.handoff_to
    assert len({h.id for h in handoffs}) == len(handoffs)
    assert all(h.id not in nodes for h in handoffs)


@pytest.mark.parametrize(("name", "source"), CASES)
def test_parent_links(name: str, source: str) -> None:
    run = load_fixture(name, source)
    by_id = {s.id: s for s in run.steps}
    assert len(by_id) == len(run.steps)
    for step in run.steps:
        assert step.run_id == run.id
        if step.kind in ("node", "handoff"):
            assert step.parent_id is None
        else:
            parent = by_id[step.parent_id]
            assert parent.kind == "node"
            assert parent.node == step.node


@pytest.mark.parametrize(("name", "source"), CASES)
def test_llm_steps_have_model_tokens_and_messages(name: str, source: str) -> None:
    run = load_fixture(name, source)
    llm_steps = run.steps_of("llm")
    assert llm_steps
    for step in llm_steps:
        assert step.model == "fake-scripted-1"
        assert step.tokens_in and step.tokens_out
        assert isinstance(step.input, list)
        assert step.input[0]["type"] == "system"
        assert step.output["type"] == "ai"


@pytest.mark.parametrize("source", SOURCES)
def test_pipeline_handoff_carries_the_caveat(source: str) -> None:
    run = load_fixture("pipeline_dropped_caveat", source)
    (handoff,) = run.steps_of("handoff")
    assert "12 customers" in handoff.input["analysis"]
    assert "12 customers" in handoff.output["analysis"]
    assert "12 customers" not in run.output_preview


@pytest.mark.parametrize("source", SOURCES)
def test_tool_errors_and_args(source: str) -> None:
    run = load_fixture("tool_retry_same_args", source)
    tools = run.steps_of("tool")
    assert [t.name for t in tools] == ["fetch_url"] * 3
    assert all(t.error and "ConnectionError" in t.error for t in tools)
    assert all(t.input == {"url": "https://example.com/pricing"} for t in tools)
    assert all(t.role == "tool_executor" for t in tools)


@pytest.mark.parametrize("source", SOURCES)
def test_tool_output_recorded(source: str) -> None:
    run = load_fixture("tool_output_unused", source)
    (tool,) = run.steps_of("tool")
    assert "CANCELLED" in tool.output
    assert tool.input == {"order_id": "A-1042"}


@pytest.mark.parametrize("source", SOURCES)
def test_recursion_limit_run_keeps_error_and_last_output(source: str) -> None:
    run = load_fixture("supervisor_no_end", source)
    assert "Recursion limit" in run.error
    assert run.output_preview


def test_roles() -> None:
    roles = {
        (s.graph_node, s.node, s.role)
        for name in EXPECTED
        for s in load_fixture(name).steps_of("node")
    }
    assert roles == {
        ("researcher", "researcher", "researcher"),
        ("tools", "researcher", "tool_executor"),
        ("writer", "writer", "writer"),
        ("critic", "critic", "critic"),
        ("analyst", "analyst", "researcher"),
        ("supervisor", "supervisor", "supervisor"),
        ("agent", "agent", "other"),
        ("tools", "agent", "tool_executor"),
        ("support_agent", "support_agent", "other"),
        ("tools", "support_agent", "tool_executor"),
    }


@pytest.mark.parametrize("name", list(EXPECTED))
def test_sources_agree(name: str) -> None:
    def shape(run):
        return (
            run.status,
            run.task_type,
            run.totals,
            run.input_preview,
            run.output_preview,
            [
                (s.kind, s.name, s.node, s.role, s.tokens_in, s.tokens_out, bool(s.error))
                for s in run.steps
            ],
        )

    a, b = load_fixture(name, "capture"), load_fixture(name, "langsmith")
    assert shape(a)[:2] == shape(b)[:2]
    assert a.totals.model_dump(exclude={"duration_s"}) == b.totals.model_dump(
        exclude={"duration_s"}
    )
    assert shape(a)[3:] == shape(b)[3:]


def test_step_kind_counts() -> None:
    run = load_fixture("healthy")
    assert Counter(s.kind for s in run.steps) == {"node": 5, "llm": 4, "tool": 1, "handoff": 2}


def test_committed_traces_hold_no_local_paths() -> None:
    from tests.helpers import TRACES

    for path in TRACES.rglob("*.json*"):
        text = path.read_text()
        assert "/Users/" not in text and "/home/" not in text and "C:\\\\Users" not in text, path


INNER_NAMES = {"agent", "model", "tools", "__start__", "__end__"}


@pytest.mark.parametrize("source", SOURCES)
def test_nested_agents_are_labeled_by_outermost_agent(source: str) -> None:
    run = load_fixture("nested_agents", source)
    assert {s.node for s in run.steps} == {"supervisor", "research_agent"}
    assert not {s.node for s in run.steps} & INNER_NAMES
    assert not any(s.name == "__start__" for s in run.steps)  # LangGraph internals dropped
    # The inner LangGraph nodes survive only in graph_node, for internal logic.
    assert {s.graph_node for s in run.steps_of("node")} == {
        "supervisor",
        "research_agent/agent",
        "research_agent/tools",
    }
    assert [(h.handoff_from, h.handoff_to) for h in run.steps_of("handoff")] == [
        ("supervisor", "research_agent"),
        ("research_agent", "supervisor"),
    ]
    (tool,) = run.steps_of("tool")
    assert tool.node == "research_agent" and tool.role == "tool_executor"
    assert run.totals.step_count == 5  # supervisor, agent, tools, agent, supervisor


@pytest.mark.parametrize("source", SOURCES)
def test_judge_window_shows_each_acting_agents_instructions(source: str) -> None:
    from tests.fixtures.agents.nested_agents import RESEARCH_PROMPT, SUPERVISOR_PROMPT
    from tracedoc.checks.judges.windows import window_around

    run = load_fixture("nested_agents", source)
    (tool,) = run.steps_of("tool")
    text = window_around(run, [tool.id], 3, "candidate").render(2000)
    instructions = text[text.index("AGENT INSTRUCTIONS") : text.index("STEPS:")]
    assert f"- research_agent: {RESEARCH_PROMPT}" in instructions
    assert f"- supervisor: {SUPERVISOR_PROMPT}" in instructions
    assert "agent=research_agent" in text and "agent=supervisor" in text
    assert "agent=agent" not in text and "agent=tools" not in text


# ---- final answer and the no_answer status ----------------------------------------------


def _root_with(outputs: object):
    from tracedoc.ingest.base import RawSpan

    return RawSpan(id="r", parent_id=None, kind="chain", name="g", outputs=outputs, ended=True)


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        (
            {"messages": [{"type": "human", "content": "q"}, {"type": "ai", "content": "Paris."}]},
            "Paris.",
        ),
        ({"messages": [{"type": "ai", "content": ""}]}, ""),
        ({"messages": [{"type": "ai", "content": []}]}, ""),
        ({"messages": [{"type": "ai", "content": "  \n …  "}]}, "…"),
        (
            {
                "messages": [
                    {"type": "ai", "content": "", "tool_calls": [{"name": "t", "args": {}}]}
                ]
            },
            "",
        ),
        ({"messages": [{"type": "ai", "content": "x"}, {"type": "tool", "content": "done"}]}, ""),
    ],
)
def test_final_answer_from_messages(outputs: object, expected: str) -> None:
    from tracedoc.normalize import _final_answer

    assert _final_answer(_root_with(outputs), []) == expected


def test_final_answer_from_state_without_messages() -> None:
    from tests.builders import build_run, step
    from tracedoc.normalize import _final_answer

    run = build_run("r", [step("node", "writer", output={"report": "", "draft": "  "})])
    assert _final_answer(_root_with({"report": ""}), run.steps_of("node")) == "   "
    unknown = build_run("u", [step("node", "flag", output={"done": True})])
    assert _final_answer(_root_with({"done": True}), unknown.steps_of("node")) is None


@pytest.mark.parametrize("source", SOURCES)
def test_empty_final_reply_is_no_answer_not_completed(source: str) -> None:
    run = load_fixture("nested_agents", source)
    assert run.status == "no_answer"
    assert run.final_answer == ""
    assert load_fixture("healthy", source).final_answer.startswith("APPROVED")
