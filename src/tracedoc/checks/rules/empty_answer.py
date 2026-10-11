"""design.empty_answer: the run finished, but its final reply had no text."""

from __future__ import annotations

from ...schema import Finding, Impact, Run
from ..base import Check, register
from ..util import excerpt, step_cost, step_latency, step_tokens


@register
class EmptyAnswer(Check):
    id = "design.empty_answer"
    group = "design"
    method = "rule"
    default_severity = "high"
    title = "Final answer was empty"

    def run(self, runs: list[Run]) -> list[Finding]:
        findings = []
        for run in runs:
            if run.status != "no_answer":
                continue
            llm_steps = run.steps_of("llm")
            last = llm_steps[-1] if llm_steps else (run.steps_of("node") or [None])[-1]
            agent = last.node if last is not None else None
            reply = excerpt(last.output, 120) if last is not None else "(none)"
            findings.append(
                self.finding(
                    run,
                    key=agent or "",
                    node=agent,
                    role=last.role if last is not None else None,
                    evidence_step_ids=[last.id] if last is not None else [],
                    evidence_excerpts=[
                        f"Request: {excerpt(run.input_preview, 120)}",
                        f"Final reply: {reply}",
                    ],
                    explanation=(
                        f"The run finished, but the final reply from `{agent}` had no text, "
                        "so the user got no answer."
                    ),
                    suggested_fix=(
                        f"Tell `{agent}` to always end with a written answer for the user, and "
                        "check for an empty reply before the graph routes to its end."
                    ),
                    impact=Impact(
                        tokens=step_tokens(llm_steps),
                        cost=step_cost(llm_steps),
                        latency_s=step_latency(llm_steps),
                    ),
                )
            )
        return findings
