"""Bug: the researcher runs the identical web search three times before answering."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ._fake import ScriptedChatModel, ai

BUG = "design.loop"
INPUT = {
    "messages": [HumanMessage("What are the main approaches to long-term memory in LangGraph?")]
}
SYSTEM = "You are a research assistant. Search the web, then answer concisely with sources."
QUERY = {"query": "langgraph long-term memory approaches"}


@tool
def web_search(query: str) -> str:
    """Search the web and return the top results."""
    return (
        "1. LangGraph Store: namespaced key-value store for cross-thread memory. "
        "2. Checkpointers persist thread state between turns. "
        "3. Semantic search over the store with embeddings."
    )


def build() -> CompiledStateGraph:
    model = ScriptedChatModel(
        responses=[
            ai(tool_calls=[{"name": "web_search", "args": QUERY, "id": "call_1"}]),
            ai(tool_calls=[{"name": "web_search", "args": QUERY, "id": "call_2"}]),
            ai(tool_calls=[{"name": "web_search", "args": QUERY, "id": "call_3"}]),
            ai(
                "LangGraph offers three main approaches: the Store for cross-thread memory, "
                "checkpointers for per-thread state, and semantic search over the store."
            ),
        ]
    ).bind_tools([web_search])

    def researcher(state: MessagesState) -> dict[str, Any]:
        return {"messages": [model.invoke([SystemMessage(SYSTEM), *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("researcher", researcher)
    graph.add_node("tools", ToolNode([web_search]))
    graph.add_edge(START, "researcher")
    graph.add_conditional_edges("researcher", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "researcher")
    return graph.compile()
