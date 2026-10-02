"""Bug: the supervisor never routes to FINISH, so the run hits the recursion limit."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ._fake import ScriptedChatModel, ai

BUG = "design.no_stop"
INPUT = {"messages": [HumanMessage("Write a one-paragraph brief on solid-state batteries.")]}
RECURSION_LIMIT = 12


class State(MessagesState):
    next: str


def build() -> CompiledStateGraph:
    supervisor_model = ScriptedChatModel(
        responses=[ai("researcher"), ai("writer")],
        cycle=True,
    )
    researcher_model = ScriptedChatModel(
        responses=[ai("Solid-state batteries replace liquid electrolyte with a solid one.")],
        cycle=True,
    )
    writer_model = ScriptedChatModel(
        responses=[ai("Solid-state batteries promise higher energy density and safety.")],
        cycle=True,
    )

    def supervisor(state: State) -> dict[str, Any]:
        reply = supervisor_model.invoke(
            [
                SystemMessage(
                    "You manage a researcher and a writer. Reply with the next worker's name, "
                    "or FINISH when the brief is complete."
                ),
                *state["messages"],
            ]
        )
        return {"next": str(reply.content).strip()}

    def researcher(state: State) -> dict[str, Any]:
        reply = researcher_model.invoke([SystemMessage("You research."), *state["messages"]])
        return {"messages": [reply]}

    def writer(state: State) -> dict[str, Any]:
        reply = writer_model.invoke([SystemMessage("You write briefs."), *state["messages"]])
        return {"messages": [reply]}

    def route(state: State) -> Literal["researcher", "writer", "__end__"]:
        return END if state["next"] == "FINISH" else state["next"]  # type: ignore[return-value]

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route)
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    return graph.compile()
