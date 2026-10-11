"""design.no_stop: the run never reaches a clear end."""

from __future__ import annotations

from ...schema import Finding, Impact, Run
from ..base import Check, register
from ..util import step_cost, step_tokens


@register
class NoStop(Check):
    id = "design.no_stop"
    group = "design"
    method = "rule"
    default_severity = "high"

    def run(self, runs: list[Run]) -> list[Finding]:
        findings = []
        for run in runs:
            hit_recursion = run.status == "recursion_limit"
            hit_steps = run.totals.step_count >= self.t.step_limit
            if not (hit_recursion or hit_steps):
                continue
            nodes = run.steps_of("node")
            counts: dict[str, int] = {}
            for s in nodes:
                counts[s.node or s.name] = counts.get(s.node or s.name, 0) + 1
            busiest = max(counts, key=counts.__getitem__) if counts else None
            llm_steps = run.steps_of("llm")
            findings.append(
                self.finding(
                    run,
                    key=busiest or "",
                    node=busiest,
                    role=next((s.role for s in nodes if s.node == busiest), None),
                    evidence_step_ids=[s.id for s in nodes[-4:]],
                    evidence_excerpts=[
                        " -> ".join(str(s.node) for s in nodes[-8:]),
                        *(run.error.splitlines()[:1] if run.error else []),
                    ],
                    variant="recursion" if hit_recursion else "step_limit",
                    facts={
                        "agent": busiest or "the agent",
                        "count": counts.get(busiest or "", 0),
                        "steps": run.totals.step_count,
                        "limit": self.t.step_limit,
                    },
                    impact=Impact(
                        tokens=step_tokens(llm_steps),
                        cost=step_cost(llm_steps),
                        latency_s=run.totals.duration_s or 0.0,
                    ),
                )
            )
        return findings
