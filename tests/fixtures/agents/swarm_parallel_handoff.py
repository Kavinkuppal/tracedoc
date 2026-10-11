"""Bug: the run ends in an error. flight_assistant books a flight and hands off to
hotel_assistant in the same reply; the handoff passes control before book_flight's result is
recorded, so hotel_assistant's model call is rejected (a tool call without a result).

The swarm shape of real multi-agent systems: each agent is a compiled subgraph, and a handoff
tool returns a Command that jumps to the other agent in the parent graph.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command

from ._fake import InvalidRequestError, ScriptedChatModel, ai

BUG = "design.run_error"
ERROR = InvalidRequestError
INPUT = {
    "messages": [
        HumanMessage("Book flight BA117 to London and a hotel near Heathrow for 2 nights.")
    ]
}
FLIGHT_PROMPT = "You book flights. Hand hotel requests to hotel_assistant."
HOTEL_PROMPT = "You book hotels."


@tool
def book_flight(flight_id: str) -> str:
    """Book a flight by its ID."""
    return f"Booked flight {flight_id}, confirmation K7Q2LM."


@tool
def transfer_to_hotel_assistant(
    state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Hand the conversation to the hotel assistant."""
    done = ToolMessage(
        "Transferred to hotel_assistant.",
        name="transfer_to_hotel_assistant",
        tool_call_id=tool_call_id,
    )
    return Command(
        goto="hotel_assistant",
        graph=Command.PARENT,
        update={"messages": [*state["messages"], done]},
    )


def build() -> CompiledStateGraph:
    flight_model = ScriptedChatModel(
        responses=[
            ai(
                "Booking flight BA117 and passing you to the hotel assistant.",
                tool_calls=[
                    {"name": "book_flight", "args": {"flight_id": "BA117"}, "id": "toolu_1"},
                    {"name": "transfer_to_hotel_assistant", "args": {}, "id": "toolu_2"},
                ],
            )
        ],
        require_tool_results=True,
    )
    hotel_model = ScriptedChatModel(
        responses=[ai("Booked 2 nights at the Heathrow Lodge.")], require_tool_results=True
    )

    def flight_agent(state: MessagesState) -> dict[str, Any]:
        return {
            "messages": [flight_model.invoke([SystemMessage(FLIGHT_PROMPT), *state["messages"]])]
        }

    def hotel_agent(state: MessagesState) -> dict[str, Any]:
        return {"messages": [hotel_model.invoke([SystemMessage(HOTEL_PROMPT), *state["messages"]])]}

    flight = StateGraph(MessagesState)
    flight.add_node("agent", flight_agent)
    flight.add_node("tools", ToolNode([book_flight, transfer_to_hotel_assistant]))
    flight.add_edge(START, "agent")
    flight.add_conditional_edges("agent", tools_condition)
    flight.add_edge("tools", "agent")

    hotel = StateGraph(MessagesState)
    hotel.add_node("agent", hotel_agent)
    hotel.add_edge(START, "agent")

    graph = StateGraph(MessagesState)
    graph.add_node(
        "flight_assistant",
        flight.compile(name="flight_assistant"),
        destinations=("hotel_assistant",),
    )
    graph.add_node("hotel_assistant", hotel.compile(name="hotel_assistant"))
    graph.add_edge(START, "flight_assistant")
    return graph.compile()
