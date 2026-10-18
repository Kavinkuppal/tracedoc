"""Small LangGraph agents with deliberate bugs, driven by scripted fake models.

Each module exposes `build()`, `INPUT`, and `BUG` (the check it should trigger,
or None for the healthy agent). Modules may set `RECURSION_LIMIT`, and `ERROR`, the
exception a run of an agent that ends in an error raises.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from . import (
    healthy,
    made_up_flight,
    nested_agents,
    pipeline_dropped_caveat,
    researcher_repeat_search,
    supervisor_agent_name,
    supervisor_no_end,
    swarm_parallel_handoff,
    tool_output_unused,
    tool_retry_same_args,
)

AGENTS: dict[str, ModuleType] = {
    "researcher_repeat_search": researcher_repeat_search,
    "pipeline_dropped_caveat": pipeline_dropped_caveat,
    "supervisor_no_end": supervisor_no_end,
    "tool_retry_same_args": tool_retry_same_args,
    "tool_output_unused": tool_output_unused,
    "healthy": healthy,
    "nested_agents": nested_agents,
    "swarm_parallel_handoff": swarm_parallel_handoff,
    "supervisor_agent_name": supervisor_agent_name,
    "made_up_flight": made_up_flight,
}

DEFAULT_RECURSION_LIMIT = 25


def run_agent(name: str, callbacks: list[BaseCallbackHandler] | None = None) -> dict[str, Any]:
    """Build a fresh agent and invoke it. Errors such as GraphRecursionError, or an agent's
    `ERROR`, propagate."""
    module = AGENTS[name]
    config: dict[str, Any] = {
        "recursion_limit": getattr(module, "RECURSION_LIMIT", DEFAULT_RECURSION_LIMIT),
        "callbacks": callbacks or [],
        "run_name": name,
    }
    return module.build().invoke(module.INPUT, config=config)
