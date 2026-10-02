"""Bug: the agent looks up the order, then answers without using the result."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ._fake import ScriptedChatModel, ai

BUG = "tools.unused_output"
INPUT = {"messages": [HumanMessage("Where is my order #A-1042?")]}


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order's status."""
    return f"Order {order_id}: CANCELLED by warehouse on 2026-09-28. Refund of $84.20 issued."


def build() -> CompiledStateGraph:
    model = ScriptedChatModel(
        responses=[
            ai(tool_calls=[{"name": "lookup_order", "args": {"order_id": "A-1042"}, "id": "c1"}]),
            ai("Good news! Your package is on its way and should arrive in 3 to 5 business days."),
        ]
    ).bind_tools([lookup_order])

    def support_agent(state: MessagesState) -> dict[str, Any]:
        reply = model.invoke(
            [SystemMessage("You are a customer support agent."), *state["messages"]]
        )
        return {"messages": [reply]}

    graph = StateGraph(MessagesState)
    graph.add_node("support_agent", support_agent)
    graph.add_node("tools", ToolNode([lookup_order]))
    graph.add_edge(START, "support_agent")
    graph.add_conditional_edges("support_agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "support_agent")
    return graph.compile()
