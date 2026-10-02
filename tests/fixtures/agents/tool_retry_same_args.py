"""Bug: a tool fails and the agent retries it with identical arguments."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ._fake import ScriptedChatModel, ai

BUG = "tools.errors_retries"
INPUT = {"messages": [HumanMessage("Summarize the pricing page at https://example.com/pricing")]}
ARGS = {"url": "https://example.com/pricing"}


@tool
def fetch_url(url: str) -> str:
    """Fetch a web page and return its text."""
    raise ConnectionError(f"Timed out connecting to {url}")


def build() -> CompiledStateGraph:
    model = ScriptedChatModel(
        responses=[
            ai(tool_calls=[{"name": "fetch_url", "args": ARGS, "id": "call_1"}]),
            ai(tool_calls=[{"name": "fetch_url", "args": ARGS, "id": "call_2"}]),
            ai(tool_calls=[{"name": "fetch_url", "args": ARGS, "id": "call_3"}]),
            ai("Sorry, I could not load the pricing page."),
        ]
    ).bind_tools([fetch_url])

    def agent(state: MessagesState) -> dict[str, Any]:
        reply = model.invoke([SystemMessage("You summarize web pages."), *state["messages"]])
        return {"messages": [reply]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([fetch_url], handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
