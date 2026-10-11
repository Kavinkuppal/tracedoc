"""tools.unused_output: a tool's result never shows up in what the model says or does next.

The tool result always reaches the next model *input* in an agent loop, so we compare
against later model *outputs* (text and tool-call arguments). Low overlap makes a
candidate for the judge, not a confirmed finding.
"""

from __future__ import annotations

from ...schema import Finding, Impact, Run
from ..base import Check, register
from ..util import (
    content_tokens,
    coverage,
    excerpt,
    is_handoff_tool,
    llm_output_text,
    step_latency,
)


@register
class UnusedOutput(Check):
    id = "tools.unused_output"
    group = "tools_cost"
    method = "rule"
    default_severity = "medium"

    def run(self, runs: list[Run]) -> list[Finding]:
        findings = []
        for run in runs:
            llm_steps = run.steps_of("llm")
            for tool in run.steps_of("tool"):
                if tool.error or tool.output is None or tool.ended_at is None:
                    continue
                if is_handoff_tool(tool.name):
                    continue  # a transfer confirmation is not data to use
                tool_tokens = content_tokens(excerpt(tool.output, 4000))
                if len(tool_tokens) < self.t.unused_output_min_tokens:
                    continue
                later = [s for s in llm_steps if s.started_at and s.started_at >= tool.ended_at]
                if not later:
                    continue
                later_tokens = set().union(*(content_tokens(llm_output_text(s)) for s in later))
                overlap = coverage(tool_tokens, later_tokens)
                if overlap > self.t.unused_output_max_overlap:
                    continue
                final = later[-1]
                findings.append(
                    self.finding(
                        run,
                        key=tool.name,
                        node=tool.node,
                        role=tool.role,
                        evidence_step_ids=[tool.id, final.id],
                        evidence_excerpts=[
                            f"{tool.name} returned: {excerpt(tool.output, 140)}",
                            f"Model then said: {excerpt(final.output, 140)}",
                        ],
                        facts={
                            "agent": tool.node or "the agent",
                            "tool": tool.name,
                            "overlap": f"{overlap:.0%}",
                        },
                        confidence=round(1.0 - overlap, 2) * 0.8,
                        needs_confirmation=True,
                        impact=Impact(latency_s=step_latency([tool])),
                    )
                )
        return findings
