"""design.run_error: the run ended in an error instead of an answer.

The finding goes to the agent whose step failed: the last model or tool call that raised an
error, else the last node that did. LangGraph's control-flow signals are not failures:
ParentCommand is how a handoff tool passes control to another agent, and the interrupt
signals pause a run on purpose.

One failure is recognized from the trace itself: a model call rejected because an earlier
reply's tool call never got a result, right after that reply both handed off and called
another tool (the handoff passes control before the other call's result is recorded).
"""

from __future__ import annotations

import re

from ...messages import payload_text
from ...schema import Finding, Run, Step
from ..base import Check, register
from ..util import excerpt, is_handoff_tool

CONTROL_FLOW = ("ParentCommand", "GraphBubbleUp", "GraphInterrupt", "NodeInterrupt", "Interrupt(")
_UNANSWERED_CALL = re.compile(
    r"tool_use.{0,40}without.{0,40}tool_result|must be followed by tool messages",
    re.IGNORECASE | re.DOTALL,
)
_API_MESSAGE = re.compile(r"""['"]message['"]:\s*(['"])(?P<message>.+?)\1""", re.DOTALL)
_ERROR_CLASS = re.compile(r"\s*(?P<kind>[A-Za-z_][\w.]*)\s*[:(]")
"""The error's class, as LangGraph's callbacks record it ("Kind: message") or as LangSmith does
("Kind('message')")."""


def is_control_flow(error: str | None) -> bool:
    return bool(error) and error.lstrip().startswith(CONTROL_FLOW)  # type: ignore[union-attr]


def without_traceback(error: str) -> str:
    return error.split("Traceback (most recent call last)", 1)[0].strip()


def error_summary(error: str) -> str:
    """The error's class and its message, without the API's surrounding JSON."""
    error = without_traceback(error)
    head = error.splitlines()[0] if error else ""
    named = _ERROR_CLASS.match(head)
    kind = named["kind"] if named else ""
    found = _API_MESSAGE.search(error)
    if found:
        return f"{kind}: {found['message']}" if kind else found["message"]
    return head


def failed_step(run: Run) -> Step | None:
    failed = [s for s in run.steps if s.error and not is_control_flow(s.error)]
    calls = [s for s in failed if s.kind in ("llm", "tool")]
    return (calls or failed or [None])[-1]


def _parallel_handoff(run: Run, before: Step) -> tuple[Step, list[str]] | None:
    """The last model reply before `before` that called a handoff tool together with another
    tool: (that model call, the names of the tools it called)."""
    position = {s.id: i for i, s in enumerate(run.steps)}
    for step in reversed(run.steps[: position[before.id]]):
        if step.kind != "llm" or not isinstance(step.output, dict):
            continue
        names = [c.get("name", "") for c in step.output.get("tool_calls") or []]
        if len(names) > 1 and any(is_handoff_tool(n) for n in names):
            return step, names
    return None


@register
class RunError(Check):
    id = "design.run_error"
    group = "design"
    method = "rule"
    default_severity = "high"

    def run(self, runs: list[Run]) -> list[Finding]:
        findings = []
        for run in runs:
            if run.status != "error":
                continue
            step = failed_step(run)
            error = (step.error if step else None) or run.error or ""
            agent = step.node if step is not None else None
            facts = {
                "agent": agent or "the system",
                "error": excerpt(error_summary(error), 160),
                "tool": step.name if step is not None else "",
            }
            variant = "run"
            evidence = [step.id] if step is not None else []
            if step is not None and step.kind == "tool":
                variant = "tool"
            elif step is not None and step.kind == "llm":
                variant = "model"
                parallel = _parallel_handoff(run, step) if _UNANSWERED_CALL.search(error) else None
                if parallel is not None:
                    reply, names = parallel
                    variant = "parallel_handoff"
                    facts["sender"] = reply.node or "an agent"
                    facts["calls"] = " and ".join(names)
                    evidence = [reply.id, step.id]
            findings.append(
                self.finding(
                    run,
                    key=agent or "",
                    node=agent,
                    role=step.role if step is not None else None,
                    evidence_step_ids=evidence,
                    evidence_excerpts=[
                        f"Request: {excerpt(run.input_preview, 120)}",
                        f"Error: {excerpt(without_traceback(payload_text(error)), 300)}",
                    ],
                    facts=facts,
                    variant=variant,
                )
            )
        return findings
