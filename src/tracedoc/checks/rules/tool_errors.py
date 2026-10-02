"""tools.errors_retries: tool failures, and retries with identical arguments."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

from ...schema import Finding, Impact, Run, Step
from ..base import Check, register
from ..util import canonical, excerpt, preceding_llm, step_cost, step_latency, step_tokens


@register
class ToolErrorsRetries(Check):
    id = "tools.errors_retries"
    group = "tools_cost"
    method = "rule"
    default_severity = "medium"
    title = "Tool failures and blind retries"

    def run(self, runs: list[Run]) -> list[Finding]:
        findings = []
        for run in runs:
            by_tool: dict[str, list[Step]] = defaultdict(list)
            for step in run.steps_of("tool"):
                by_tool[step.name].append(step)
            for tool, steps in by_tool.items():
                errors = [s for s in steps if s.error]
                if not errors:
                    continue
                blind_retries = [
                    cur
                    for prev, cur in pairwise(steps)
                    if prev.error and canonical(prev.input) == canonical(cur.input)
                ]
                if len(blind_retries) >= 2:
                    severity = "high"
                elif blind_retries:
                    severity = "medium"
                else:
                    severity = "low"
                issuers = [s for s in (preceding_llm(run, r) for r in blind_retries) if s]
                detail = (
                    f" and was retried {len(blind_retries)} time(s) with the same arguments"
                    if blind_retries
                    else ""
                )
                findings.append(
                    self.finding(
                        run,
                        severity=severity,
                        key=tool,
                        node=steps[0].node,
                        role=steps[0].role,
                        evidence_step_ids=[s.id for s in errors],
                        evidence_excerpts=[
                            excerpt(errors[0].error, 160),
                            f"{tool}({excerpt(errors[0].input, 100)})",
                        ],
                        explanation=f"`{tool}` failed {len(errors)} time(s){detail}.",
                        suggested_fix=(
                            f"Return the `{tool}` error to the model with guidance to change "
                            "arguments or approach, and stop after one identical retry; use "
                            "backoff for transient failures."
                            if blind_retries
                            else f"Investigate why `{tool}` fails and handle the error case."
                        ),
                        impact=Impact(
                            tokens=step_tokens(issuers),
                            cost=step_cost(issuers + blind_retries),
                            latency_s=step_latency(issuers + blind_retries),
                        ),
                    )
                )
        return findings
