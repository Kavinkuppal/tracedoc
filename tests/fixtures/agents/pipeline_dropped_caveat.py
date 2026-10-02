"""Bug: the writer drops the analyst's caveat about the tiny sample size."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ._fake import ScriptedChatModel, ai

BUG = "agents.ignored_output"
INPUT = {"question": "Did the new onboarding flow improve 30-day retention?"}
CAVEAT = "CAVEAT: the sample is only 12 customers, so this result is not statistically reliable."


class State(TypedDict, total=False):
    question: str
    analysis: str
    report: str


def build() -> CompiledStateGraph:
    analyst_model = ScriptedChatModel(
        responses=[
            ai(
                "Retention rose from 41% to 58% for users who saw the new onboarding flow. "
                + CAVEAT
            )
        ]
    )
    writer_model = ScriptedChatModel(
        responses=[
            ai(
                "The new onboarding flow lifted 30-day retention from 41% to 58%. "
                "We recommend rolling it out to all users."
            )
        ]
    )

    def analyst(state: State) -> dict[str, Any]:
        reply = analyst_model.invoke(
            [
                SystemMessage("You are a data analyst. Report findings and their limitations."),
                HumanMessage(state["question"]),
            ]
        )
        return {"analysis": reply.content}

    def writer(state: State) -> dict[str, Any]:
        reply = writer_model.invoke(
            [
                SystemMessage(
                    "You write short executive summaries. Preserve every caveat from the analysis."
                ),
                HumanMessage(f"Question: {state['question']}\n\nAnalysis: {state['analysis']}"),
            ]
        )
        return {"report": reply.content}

    graph = StateGraph(State)
    graph.add_node("analyst", analyst)
    graph.add_node("writer", writer)
    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)
    return graph.compile()
