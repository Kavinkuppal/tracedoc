"""Group findings across runs by signature, and rank the groups."""

from __future__ import annotations

from collections import defaultdict

from .schema import Finding, FindingGroup, Impact

SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def signature(f: Finding) -> tuple[str, str, str]:
    return (f.check_id, f.node or f.role or "", f.key)


def group_findings(findings: list[Finding], total_runs: int) -> list[FindingGroup]:
    buckets: dict[tuple[str, str, str], list[Finding]] = defaultdict(list)
    for f in findings:
        buckets[signature(f)].append(f)
    groups = []
    for (check_id, _, key), members in buckets.items():
        runs = {f.run_id for f in members}
        representative = min(
            members,
            key=lambda f: (SEVERITY_RANK[f.severity], f.needs_confirmation, -f.confidence),
        )
        groups.append(
            FindingGroup(
                check_id=check_id,
                node=representative.node,
                role=representative.role,
                key=key,
                runs_affected=len(runs),
                frequency=len(runs) / total_runs if total_runs else 0.0,
                impact=Impact(
                    tokens=sum(f.impact.tokens for f in members),
                    cost=sum(f.impact.cost for f in members),
                    latency_s=sum(f.impact.latency_s for f in members),
                ),
                representative=representative,
                findings=members,
            )
        )
    return rank(groups)


def rank(groups: list[FindingGroup]) -> list[FindingGroup]:
    """Severity, then frequency, then impact. Within a severity, judge-confirmed (or rule-certain)
    groups come before unconfirmed candidates, so the top three are findings we stand behind."""
    return sorted(
        groups,
        key=lambda g: (
            SEVERITY_RANK[g.representative.severity],
            g.representative.needs_confirmation,
            -g.frequency,
            -g.impact.tokens,
            -g.impact.cost,
            -g.impact.latency_s,
        ),
    )
