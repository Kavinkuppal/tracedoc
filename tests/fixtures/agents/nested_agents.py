"""Bug: after the research agent hands back, the supervisor's final reply is empty.

Also the nested-agent shape of real multi-agent systems: `research_agent` is a prebuilt ReAct
agent compiled as a subgraph, so its steps run inside inner LangGraph nodes named "agent" and
"tools", and the graph enters through a router on START (a "__start__" step).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from ._fake import ScriptedChatModel, ai

BUG = "design.empty_answer"
INPUT = {"messages": [HumanMessage("What was Apple's 2024 headcount?")]}
SUPERVISOR_PROMPT = "You are a supervisor. Send research questions to research_agent."
RESEARCH_PROMPT = "You are a researcher with web search. Do not do any math."


class State(MessagesState):
    next: str


@tool
def web_search(query: str) -> str:
    """Search the web."""
    return "Apple had 164,000 employees in 2024."


def build() -> CompiledStateGraph:
    supervisor_model = ScriptedChatModel(responses=[ai("research_agent"), ai("")])
    research_model = ScriptedChatModel(
        responses=[
            ai(
                tool_calls=[
                    {"name": "web_search", "args": {"query": "Apple 2024 headcount"}, "id": "c1"}
                ]
            ),
            ai("Apple had 164,000 employees in 2024."),
        ]
    )
    with warnings.catch_warnings():
        # Deprecated in LangGraph 1.0, but still what many real systems (and LangChain's own
        # supervisor example) use; the nested shape is what this fixture is for.
        warnings.simplefilter("ignore", DeprecationWarning)
        research_agent = create_react_agent(
            research_model, tools=[web_search], prompt=RESEARCH_PROMPT, name="research_agent"
        )

    def supervisor(state: State) -> dict[str, Any]:
        reply = supervisor_model.invoke([SystemMessage(SUPERVISOR_PROMPT), *state["messages"]])
        route = str(reply.content).strip()
        return {"messages": [reply], "next": route if route == "research_agent" else "FINISH"}

    def entry(state: State) -> Literal["supervisor"]:
        return "supervisor"

    def route(state: State) -> Literal["research_agent", "__end__"]:
        return "research_agent" if state["next"] == "research_agent" else END

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("research_agent", research_agent)
    graph.add_conditional_edges(START, entry)
    graph.add_conditional_edges("supervisor", route)
    graph.add_edge("research_agent", "supervisor")
    return graph.compile()
