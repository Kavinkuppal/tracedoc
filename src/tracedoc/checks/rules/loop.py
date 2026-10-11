"""design.loop: repeated steps."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher

from ...schema import Finding, Impact, Run, Step
from ..base import Check, register
from ..util import canonical, excerpt, preceding_llm, step_cost, step_latency, step_tokens


@register
class Loop(Check):
    id = "design.loop"
    group = "design"
    method = "rule"
    default_severity = "medium"

    def run(self, runs: list[Run]) -> list[Finding]:
        findings: list[Finding] = []
        for run in runs:
            findings.extend(self._identical_tool_calls(run))
            findings.extend(self._near_duplicate_tool_calls(run))
            findings.extend(self._node_cycles(run))
        return findings

    def _impact(self, run: Run, repeats: list[Step]) -> Impact:
        """Waste = the repeated calls after the first, plus the model calls that issued them."""
        issuers = [s for s in (preceding_llm(run, r) for r in repeats) if s is not None]
        return Impact(
            tokens=step_tokens(issuers),
            cost=step_cost(issuers + repeats),
            latency_s=step_latency(issuers + repeats),
        )

    def _identical_tool_calls(self, run: Run) -> list[Finding]:
        """Same tool, same normalized arguments, N+ times. Retries after errors are excluded:
        those belong to tools.errors_retries."""
        groups: dict[tuple[str, str], list[Step]] = defaultdict(list)
        for step in run.steps_of("tool"):
            groups[(step.name, canonical(step.input))].append(step)
        findings = []
        for (tool, _), steps in groups.items():
            fresh = [s for i, s in enumerate(steps) if i == 0 or steps[i - 1].error is None]
            if len(fresh) < self.t.loop_min_repeats:
                continue
            severity = "high" if len(fresh) >= 2 * self.t.loop_min_repeats else "medium"
            findings.append(
                self.finding(
                    run,
                    severity=severity,
                    key=tool,
                    node=steps[0].node,
                    role=steps[0].role,
                    evidence_step_ids=[s.id for s in fresh],
                    evidence_excerpts=[f"{tool}({excerpt(steps[0].input, 120)})"],
                    variant="identical",
                    facts={
                        "agent": steps[0].node or "the agent",
                        "tool": tool,
                        "count": len(fresh),
                    },
                    impact=self._impact(run, fresh[1:]),
                )
            )
        return findings

    def _near_duplicate_tool_calls(self, run: Run) -> list[Finding]:
        """Same tool with very similar (not identical) arguments: a judge decides."""
        by_tool: dict[str, list[Step]] = defaultdict(list)
        for step in run.steps_of("tool"):
            by_tool[step.name].append(step)
        findings = []
        for tool, steps in by_tool.items():
            args = [canonical(s.input) for s in steps]
            if len(set(args)) < 2:
                continue
            cluster = [steps[0]]
            for i in range(1, len(steps)):
                ratio = SequenceMatcher(None, args[0], args[i]).ratio()
                if args[i] != args[0] and ratio >= self.t.loop_near_duplicate_similarity:
                    cluster.append(steps[i])
            if len(cluster) < self.t.loop_min_repeats:
                continue
            findings.append(
                self.finding(
                    run,
                    key=tool,
                    node=steps[0].node,
                    role=steps[0].role,
                    evidence_step_ids=[s.id for s in cluster],
                    evidence_excerpts=[f"{tool}({excerpt(s.input, 80)})" for s in cluster[:3]],
                    variant="near_duplicate",
                    facts={
                        "agent": steps[0].node or "the agent",
                        "tool": tool,
                        "count": len(cluster),
                    },
                    confidence=0.5,
                    needs_confirmation=True,
                    impact=self._impact(run, cluster[1:]),
                )
            )
        return findings

    def _node_cycles(self, run: Run) -> list[Finding]:
        """A cycle of agent nodes repeating N+ times with identical outputs (no state change).
        Cycles through tool nodes are covered by the tool-call checks above."""
        nodes = [s for s in run.steps_of("node") if s.parent_id is None]
        if any(s.role == "tool_executor" for s in nodes):
            return []
        sigs = [(s.node, canonical(s.output)) for s in nodes]
        for length in range(1, self.t.loop_max_cycle_length + 1):
            for start in range(len(sigs) - length * self.t.loop_min_repeats + 1):
                pattern = sigs[start : start + length]
                reps = 1
                while sigs[start + reps * length : start + (reps + 1) * length] == pattern:
                    reps += 1
                if reps < self.t.loop_min_repeats:
                    continue
                cycle = nodes[start : start + reps * length]
                names = " -> ".join(str(n) for n, _ in pattern)
                return [
                    self.finding(
                        run,
                        key=names,
                        node=cycle[0].node,
                        role=cycle[0].role,
                        evidence_step_ids=[s.id for s in cycle],
                        evidence_excerpts=[f"{names} (x{reps})"],
                        variant="cycle",
                        facts={
                            "agent": cycle[0].node or "the agent",
                            "cycle": " → ".join(str(n) for n, _ in pattern),
                            "count": reps,
                        },
                        impact=Impact(
                            tokens=step_tokens(
                                s
                                for s in run.steps_of("llm")
                                if s.parent_id in {c.id for c in cycle[length:]}
                            ),
                            latency_s=step_latency(cycle[length:]),
                        ),
                    )
                ]
        return []
