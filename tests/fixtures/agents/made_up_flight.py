"""Bug: the final answer states a flight the search never returned.

search_flights returns UA 1532 from BOS to ORD on 2026-06-03 at 08:05 for $214; the agent
answers with UA 1523 to LAX on June 4 at 9:15 for $199. Every fact in the answer is made up.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ._fake import ScriptedChatModel, ai

BUG = "verification.contradicted_result"
INPUT = {"messages": [HumanMessage("Find me the cheapest flight from Boston to Chicago.")]}
PROMPT = "You find flights with search_flights and tell the user the best option."
TOOL_RESULT = (
    '[{"flight": "UA 1532", "from": "BOS", "to": "ORD", "date": "2026-06-03", '
    '"departs": "08:05", "price": "$214"}, {"flight": "AA 301", "from": "BOS", "to": "ORD", '
    '"date": "2026-06-03", "departs": "11:40", "price": "$259"}]'
)
ANSWER = "The cheapest is United UA 1523 from BOS to LAX on June 4, departing 9:15 am, for $199."


@tool
def search_flights(origin: str, destination: str) -> str:
    """Search flights between two cities."""
    return TOOL_RESULT


def build() -> CompiledStateGraph:
    model = ScriptedChatModel(
        responses=[
            ai(
                tool_calls=[
                    {
                        "name": "search_flights",
                        "args": {"origin": "Boston", "destination": "Chicago"},
                        "id": "c1",
                    }
                ]
            ),
            ai(ANSWER),
        ]
    ).bind_tools([search_flights])

    def flight_agent(state: MessagesState) -> dict[str, Any]:
        return {"messages": [model.invoke([SystemMessage(PROMPT), *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("tools", ToolNode([search_flights]))
    graph.add_edge(START, "flight_agent")
    graph.add_conditional_edges("flight_agent", tools_condition)
    graph.add_edge("tools", "flight_agent")
    return graph.compile()
