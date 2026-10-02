"""Healthy: one search, a writer that uses it, a critic that verifies before finishing."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ._fake import ScriptedChatModel, ai

BUG = None
INPUT = {"messages": [HumanMessage("What is the boiling point of water at the top of Everest?")]}


@tool
def web_search(query: str) -> str:
    """Search the web and return the top results."""
    return "At the summit of Everest (8,849 m), water boils at about 70 degrees Celsius."


def build() -> CompiledStateGraph:
    researcher_model = ScriptedChatModel(
        responses=[
            ai(
                tool_calls=[
                    {"name": "web_search", "args": {"query": "boiling point Everest"}, "id": "c1"}
                ]
            ),
            ai("Research: water boils at about 70 degrees Celsius at the summit of Everest."),
        ]
    ).bind_tools([web_search])
    writer_model = ScriptedChatModel(
        responses=[
            ai(
                "At the top of Everest (8,849 m) water boils at about 70 degrees Celsius, "
                "because the air pressure is roughly a third of sea level."
            )
        ]
    )
    critic_model = ScriptedChatModel(
        responses=[ai("APPROVED: the answer matches the research (about 70 degrees Celsius).")]
    )

    def researcher(state: MessagesState) -> dict[str, Any]:
        reply = researcher_model.invoke(
            [SystemMessage("You research facts with web_search."), *state["messages"]]
        )
        return {"messages": [reply]}

    def writer(state: MessagesState) -> dict[str, Any]:
        reply = writer_model.invoke(
            [SystemMessage("Write a clear answer from the research."), *state["messages"]]
        )
        return {"messages": [reply]}

    def critic(state: MessagesState) -> dict[str, Any]:
        reply = critic_model.invoke(
            [SystemMessage("Check the answer against the research."), *state["messages"]]
        )
        return {"messages": [reply]}

    graph = StateGraph(MessagesState)
    graph.add_node("researcher", researcher)
    graph.add_node("tools", ToolNode([web_search]))
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)
    graph.add_edge(START, "researcher")
    graph.add_conditional_edges("researcher", tools_condition, {"tools": "tools", END: "writer"})
    graph.add_edge("tools", "researcher")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", END)
    return graph.compile()
