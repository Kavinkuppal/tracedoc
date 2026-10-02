"""Map LangGraph node names to a fixed list of roles."""

from __future__ import annotations

import re
from collections.abc import Callable

from ..schema import ROLES, Role

RoleClassifier = Callable[[str], "Role | None"]
"""Optional model fallback: given a node name (and nothing else), return a role or None."""

# Checked in order; the first role with a matching word wins.
_KEYWORDS: list[tuple[Role, tuple[str, ...]]] = [
    ("tool_executor", ("tools", "tool", "toolnode", "action", "actions", "executor", "execute")),
    ("supervisor", ("supervisor", "manager", "orchestrator", "router", "coordinator", "boss")),
    (
        "critic",
        (
            "critic",
            "critique",
            "review",
            "reviewer",
            "verify",
            "verifier",
            "checker",
            "judge",
            "evaluator",
            "evaluate",
            "grader",
            "grade",
            "reflect",
            "reflection",
            "validator",
            "validate",
        ),
    ),
    ("planner", ("planner", "plan", "planning", "decompose", "decomposer")),
    (
        "researcher",
        (
            "researcher",
            "research",
            "search",
            "searcher",
            "retriever",
            "retrieve",
            "browse",
            "browser",
            "analyst",
            "analyze",
            "analyzer",
            "investigator",
        ),
    ),
    (
        "writer",
        (
            "writer",
            "write",
            "drafter",
            "draft",
            "author",
            "composer",
            "summarizer",
            "summarize",
            "reporter",
            "responder",
            "respond",
            "answer",
            "editor",
        ),
    ),
]


def _words(name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [w for w in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if w]


def role_from_name(name: str) -> Role | None:
    words = set(_words(name))
    for role, keywords in _KEYWORDS:
        if words & set(keywords):
            return role
    return None


def map_role(
    name: str,
    *,
    only_runs_tools: bool = False,
    fallback: RoleClassifier | None = None,
) -> Role:
    """Structure first (a node that only runs tools is a tool executor), then name, then model."""
    if only_runs_tools:
        return "tool_executor"
    role = role_from_name(name)
    if role is None and fallback is not None:
        guess = fallback(name)
        role = guess if guess in ROLES else None
    return role or "other"
