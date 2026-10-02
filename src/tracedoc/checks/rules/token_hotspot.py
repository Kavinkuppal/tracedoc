"""cost.token_hotspot: one node takes most tokens, or context grows steeply across a run."""

from __future__ import annotations

from collections import defaultdict

from ...schema import Finding, Impact, Run, Step
from ..base import Check, register
from ..util import step_cost, step_tokens


@register
class TokenHotspot(Check):
    id = "cost.token_hotspot"
    group = "tools_cost"
    method = "rule"
    default_severity = "low"
    title = "Token concentration and context growth"

    def run(self, runs: list[Run]) -> list[Finding]:
        findings = []
        for run in runs:
            findings.extend(self._node_share(run))
            findings.extend(self._context_growth(run))
        return findings

    def _node_share(self, run: Run) -> list[Finding]:
        by_node: dict[str, list[Step]] = defaultdict(list)
        for s in run.steps_of("llm"):
            by_node[s.node or s.name].append(s)
        if len(by_node) < 2:
            return []  # a single-agent run trivially concentrates tokens in one node
        totals = {n: step_tokens(steps) for n, steps in by_node.items()}
        total = sum(totals.values())
        node = max(totals, key=totals.__getitem__)
        share = totals[node] / total if total else 0.0
        min_share = max(self.t.token_hotspot_share, 1.5 / len(totals))
        if share < min_share or totals[node] < self.t.token_hotspot_min_tokens:
            return []
        steps = by_node[node]
        return [
            self.finding(
                run,
                key=f"share:{node}",
                node=node,
                role=steps[0].role,
                evidence_step_ids=[s.id for s in steps],
                evidence_excerpts=[f"{node}: {totals[node]:,} of {total:,} tokens"],
                explanation=f"`{node}` used {share:.0%} of this run's tokens.",
                suggested_fix=(
                    f"Trim what `{node}` sends to the model (summarize history, drop unused "
                    "state), or move it to a cheaper model."
                ),
                impact=Impact(tokens=totals[node], cost=step_cost(steps)),
            )
        ]

    def _context_growth(self, run: Run) -> list[Finding]:
        steps = [s for s in run.steps_of("llm") if s.tokens_in]
        if len(steps) < self.t.context_growth_min_steps:
            return []
        first, last = steps[0].tokens_in or 1, steps[-1].tokens_in or 0
        growth = last / first
        if growth < self.t.context_growth_factor or last < self.t.context_growth_min_tokens:
            return []
        excess = sum(max(0, (s.tokens_in or 0) - first) for s in steps)
        return [
            self.finding(
                run,
                key="context_growth",
                node=steps[-1].node,
                role=steps[-1].role,
                evidence_step_ids=[steps[0].id, steps[-1].id],
                evidence_excerpts=[f"input tokens per call: {first:,} -> {last:,}"],
                explanation=(
                    f"Input tokens per model call grew {growth:.1f}x over {len(steps)} calls."
                ),
                suggested_fix=(
                    "Summarize or trim message history between steps, and pass only the state "
                    "each node needs."
                ),
                impact=Impact(tokens=excess),
            )
        ]
