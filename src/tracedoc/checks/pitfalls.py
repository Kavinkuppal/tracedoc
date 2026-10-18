"""Known framework pitfalls: settings in a framework that fail in ways the traces show.

Each pitfall is one `Pitfall` entry in PITFALLS below, and that is the only place to add one:
its check ID, severity, reviewed report text (title, what it looks for, why it matters, what
happened, what to change), the documentation it cites, and a detector that reads one run.
The catalog (catalog.toml) takes the text from here, and checks/rules/framework.py registers
one rule check per entry, so a pitfall is reported, ranked, disabled and documented like any
other check. Detectors only read the trace; they never guess when it doesn't show enough (an
unknown model is not flagged).

This module must not import the catalog (the catalog imports it).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..messages import message_text
from ..schema import Run, Severity, Step


@dataclass(frozen=True)
class Detection:
    """Where a pitfall shows in a run: the step, and the facts that fill its text."""

    step: Step
    facts: dict[str, str]


@dataclass(frozen=True)
class Pitfall:
    id: str
    """Check ID: "framework.<library>.<setting>"."""
    framework: str
    severity: Severity
    title: str
    looks_for: str
    why: str
    what_happened: str
    what_to_change: str
    docs: str
    """The documentation the fix cites: where it is, and what it says."""
    detect: Callable[[Run], Detection | None]
    evidence: str = ""
    """Template for the evidence excerpt (facts as placeholders)."""

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "how": "rule",
            "looks_for": self.looks_for,
            "why": self.why,
            "what_happened": self.what_happened,
            "what_to_change": self.what_to_change,
            "docs": self.docs,
        }


# ---- detectors ------------------------------------------------------------------------------

_OPENAI_MODEL = re.compile(
    r"^(?:openai[/:.]|azure[/:.])?(?:gpt-|o\d|chatgpt|ft:gpt|davinci|babbage)", re.IGNORECASE
)
"""OpenAI's chat models by name (also behind an "openai/" or "azure/" prefix)."""


def _messages(step: Step) -> list[Mapping[str, Any]]:
    value = step.input
    return [m for m in value if isinstance(m, Mapping)] if isinstance(value, list) else []


def _is_handoff_back(message: Mapping[str, Any]) -> bool:
    """The message pair langgraph-supervisor adds when an agent hands back to the supervisor."""
    calls = message.get("tool_calls") or []
    if any(str(c.get("name", "")).startswith("transfer_back_to_") for c in calls):
        return True
    return message.get("type") == "tool" and str(message.get("content", "")).startswith(
        "Successfully transferred back to"
    )


def supervisor_agent_name(run: Run) -> Detection | None:
    """A langgraph-supervisor supervisor (its input holds the hand-back messages the library
    adds) whose model is not OpenAI's, shown other agents' replies with only the name
    attribute, not the inline <name> tags include_agent_name="inline" adds."""
    for step in run.steps_of("llm"):
        messages = _messages(step)
        if not any(_is_handoff_back(m) for m in messages):
            continue
        others = [
            m
            for m in messages
            if m.get("type") == "ai"
            and m.get("name")
            and m.get("name") != step.node
            and not _is_handoff_back(m)
        ]
        if not others:
            continue
        if any("<name>" in message_text(m) for m in messages if m.get("type") == "ai"):
            return None  # include_agent_name="inline" is set
        if not step.model or _OPENAI_MODEL.match(step.model):
            return None  # OpenAI models get the name attribute; an unknown model can't be judged
        return Detection(
            step=step,
            facts={
                "agent": step.node or "the supervisor",
                "model": step.model,
                "sender": str(others[-1]["name"]),
            },
        )
    return None


# ---- the catalog ----------------------------------------------------------------------------

PITFALLS: tuple[Pitfall, ...] = (
    Pitfall(
        id="framework.langgraph_supervisor.include_agent_name",
        framework="langgraph-supervisor",
        severity="high",
        title="Supervisor can't see which agent replied",
        looks_for=(
            "A langgraph-supervisor system whose supervisor runs on a model that is not "
            "OpenAI's, with include_agent_name left at its default. The library then marks who "
            "wrote each message only in the message's name attribute, which only OpenAI's API "
            "passes to the model."
        ),
        why=(
            "The supervisor's model sees the other agents' replies with no names, so it can "
            "mistake whose work it is reading, redo it, or drop it from the answer."
        ),
        what_happened=(
            "{agent} runs on {model} and received {sender}'s reply with its name only in the "
            "name attribute, which this model never sees (include_agent_name is not set)."
        ),
        what_to_change=(
            'Pass include_agent_name="inline" to create_supervisor, so {agent}\'s model sees '
            "each agent's name in the message text."
        ),
        docs=(
            'langgraph-supervisor, create_supervisor(include_agent_name): "None: Relies on the '
            "LLM provider using the name attribute on the AI message. Currently, only OpenAI "
            "supports this. 'inline': Add the agent name directly into the content field of "
            'the AI message using XML-style tags." '
            "https://github.com/langchain-ai/langgraph-supervisor-py"
        ),
        detect=supervisor_agent_name,
        evidence="{agent} on {model} received a reply from {sender} with no inline name",
    ),
)


def catalog_entries() -> dict[str, dict[str, Any]]:
    return {p.id: p.catalog_entry() for p in PITFALLS}
