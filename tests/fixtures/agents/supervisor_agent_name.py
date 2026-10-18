"""Bug: a langgraph-supervisor system on non-OpenAI models with include_agent_name left at
its default, so the supervisor's model can't see which agent wrote each reply.

Built with the real `create_supervisor` from langgraph-supervisor; the scripted models report
Anthropic model names (the library's default only works for OpenAI's API, which passes the
message's name attribute to the model). `build(include_agent_name="inline")` is the fixed
system.
"""

from __future__ import annotations

import warnings

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from ._fake import ScriptedChatModel, ai

BUG = "framework.langgraph_supervisor.include_agent_name"
INPUT = {"messages": [HumanMessage("Who won the 2024 Nobel Prize in Physics?")]}
SUPERVISOR_MODEL = "claude-sonnet-5-5"
RESEARCH_MODEL = "claude-haiku-4-5-20251001"


@tool
def web_search(query: str) -> str:
    """Search the web."""
    return "The 2024 Nobel Prize in Physics went to John Hopfield and Geoffrey Hinton."


def build(include_agent_name: str | None = None) -> CompiledStateGraph:
    supervisor_model = ScriptedChatModel(
        model_name=SUPERVISOR_MODEL,
        responses=[
            ai(tool_calls=[{"name": "transfer_to_research_expert", "args": {}, "id": "t1"}]),
            ai("John Hopfield and Geoffrey Hinton won the 2024 Nobel Prize in Physics."),
        ],
    )
    research_model = ScriptedChatModel(
        model_name=RESEARCH_MODEL,
        responses=[
            ai(
                tool_calls=[
                    {"name": "web_search", "args": {"query": "2024 Nobel physics"}, "id": "c1"}
                ]
            ),
            ai("John Hopfield and Geoffrey Hinton won the 2024 Nobel Prize in Physics."),
        ],
    )
    with warnings.catch_warnings():
        # create_react_agent is deprecated in LangGraph 1.0, but langgraph-supervisor and many
        # real systems still use it.
        warnings.simplefilter("ignore", DeprecationWarning)
        research = create_react_agent(
            research_model, tools=[web_search], name="research_expert", prompt="You research."
        )
        workflow = create_supervisor(
            [research],
            model=supervisor_model,
            prompt="You manage research_expert. Send research questions to it.",
            include_agent_name=include_agent_name,  # type: ignore[arg-type]
        )
    return workflow.compile()
