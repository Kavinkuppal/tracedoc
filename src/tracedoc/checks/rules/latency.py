"""cost.latency_outlier: steps far slower than the batch's p95 for the same node or tool."""

from __future__ import annotations

from collections import defaultdict

from ...schema import Finding, Impact, Run, Step
from ..base import Check, register
from ..util import quantile


@register
class LatencyOutlier(Check):
    id = "cost.latency_outlier"
    group = "tools_cost"
    method = "rule"
    default_severity = "low"

    def run(self, runs: list[Run]) -> list[Finding]:
        by_key: dict[str, list[tuple[Run, Step]]] = defaultdict(list)
        for run in runs:
            for s in run.steps:
                if s.duration_s is None:
                    continue
                if s.kind == "node":
                    by_key[f"node:{s.node or s.name}"].append((run, s))
                elif s.kind == "tool":
                    by_key[f"tool:{s.name}"].append((run, s))
        findings = []
        for key, items in by_key.items():
            durations = sorted(s.duration_s or 0.0 for _, s in items)
            for run, step in items:
                d = step.duration_s or 0.0
                absolute = d >= self.t.latency_absolute_s
                relative = False
                p95 = 0.0
                if len(items) >= self.t.latency_min_samples and d >= self.t.latency_min_s:
                    others = list(durations)
                    others.remove(d)
                    p95 = quantile(others, 0.95)
                    relative = d > self.t.latency_p95_multiplier * p95
                if not (absolute or relative):
                    continue
                label = key.split(":", 1)[1]
                findings.append(
                    self.finding(
                        run,
                        severity="medium" if absolute else "low",
                        key=key,
                        node=step.node,
                        role=step.role,
                        evidence_step_ids=[step.id],
                        evidence_excerpts=[f"{label}: {d:.1f}s"],
                        variant="relative" if relative else "absolute",
                        facts={
                            "agent": step.node or "the agent",
                            "label": label,
                            "seconds": f"{d:.1f} s",
                            "ratio": f"{d / p95:.1f}" if p95 else "",
                            "p95": f"{p95:.1f} s",
                            "limit": f"{self.t.latency_absolute_s:.0f} s",
                        },
                        impact=Impact(latency_s=d - p95 if relative else d),
                    )
                )
        return findings
