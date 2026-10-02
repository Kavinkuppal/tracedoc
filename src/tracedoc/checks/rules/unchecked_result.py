"""verification.unchecked_result: the run finishes without a check after the last tool output."""

from __future__ import annotations

from ...schema import Finding, Run
from ..base import Check, register
from ..util import excerpt


def verified_after_last_tool(run: Run) -> bool | None:
    """True/False for completed runs (is there a critic after the last tool output?), else None."""
    if run.status != "completed" or not run.steps_of("node"):
        return None
    tools = [s for s in run.steps_of("tool") if s.ended_at]
    anchor = max((s.ended_at for s in tools), default=run.started_at)  # type: ignore[type-var]
    return any(
        s.role == "critic" and (anchor is None or (s.started_at and s.started_at >= anchor))
        for s in run.steps_of("node")
    )


@register
class UncheckedResult(Check):
    id = "verification.unchecked_result"
    group = "verification"
    method = "rule"
    default_severity = "low"
    title = "No checking before finishing"

    def run(self, runs: list[Run]) -> list[Finding]:
        findings = []
        for run in runs:
            if verified_after_last_tool(run) is not False:
                continue
            nodes = run.steps_of("node")
            last = nodes[-1]
            findings.append(
                self.finding(
                    run,
                    key=last.node or last.name,
                    node=last.node,
                    role=last.role,
                    evidence_step_ids=[last.id],
                    evidence_excerpts=[f"Final output: {excerpt(run.output_preview, 140)}"],
                    explanation=(
                        "The run produced its final output with no critic or verification step "
                        "after the last tool result."
                    ),
                    suggested_fix=(
                        "Add a verification step before finishing: a critic node, or a "
                        "self-check prompt that compares the answer against the tool results."
                    ),
                    confidence=0.5,
                    needs_confirmation=True,
                )
            )
        return findings
