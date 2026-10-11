"""design.empty_answer: the run finished, but its final reply had no text.

The evidence includes the answer that never reached the user: the last real answer any agent
wrote in the run before the empty reply, when there is one.
"""

from __future__ import annotations

from collections.abc import Mapping

from ...messages import reply_text
from ...schema import Finding, Impact, Run, Step
from ..base import Check, register
from ..util import excerpt, is_handoff_tool, step_cost, step_latency, step_tokens

ROUTING_WORDS = {"FINISH", "END", "__END__", "DONE"}
"""Replies that only name where to go next, like a supervisor's "FINISH"."""


def lost_answer(run: Run, empty: Step | None) -> Step | None:
    """The last model reply before `empty` (the run's final, empty reply) that answered: it has
    words, calls no tools other than a handoff, and is not a routing decision (an agent's name,
    or a word such as FINISH)."""
    llms = run.steps_of("llm")
    before = llms[: llms.index(empty)] if empty in llms else llms
    names = {s.node for s in run.steps if s.node} | {s.name for s in run.steps_of("node")}
    for step in reversed(before):
        text = reply_text(step.output).strip()
        if not any(ch.isalnum() for ch in text) or text in names or text.upper() in ROUTING_WORDS:
            continue
        calls = step.output.get("tool_calls") if isinstance(step.output, Mapping) else None
        if any(not is_handoff_tool(str(c.get("name", ""))) for c in calls or []):
            continue  # still working: it asked for a tool
        return step
    return None


@register
class EmptyAnswer(Check):
    id = "design.empty_answer"
    group = "design"
    method = "rule"
    default_severity = "high"

    def run(self, runs: list[Run]) -> list[Finding]:
        findings = []
        for run in runs:
            if run.status != "no_answer":
                continue
            llm_steps = run.steps_of("llm")
            last = llm_steps[-1] if llm_steps else (run.steps_of("node") or [None])[-1]
            agent = last.node if last is not None else None
            reply = excerpt(last.output, 120) if last is not None else "(none)"
            evidence = [last.id] if last is not None else []
            excerpts = [f"Request: {excerpt(run.input_preview, 120)}", f"Final reply: {reply}"]
            lost = lost_answer(run, last)
            if lost is not None:
                evidence.append(lost.id)
                excerpts.append(
                    f"Answer that never reached the user ({lost.node or 'an agent'}): "
                    f"{excerpt(reply_text(lost.output), 200)}"
                )
            findings.append(
                self.finding(
                    run,
                    key=agent or "",
                    node=agent,
                    role=last.role if last is not None else None,
                    evidence_step_ids=evidence,
                    evidence_excerpts=excerpts,
                    facts={"agent": agent or "the agent"},
                    impact=Impact(
                        tokens=step_tokens(llm_steps),
                        cost=step_cost(llm_steps),
                        latency_s=step_latency(llm_steps),
                    ),
                )
            )
        return findings
