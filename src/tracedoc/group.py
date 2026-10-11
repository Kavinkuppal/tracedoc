"""Group findings across runs by signature, and rank the groups.

Ranking is fixed and documented (details.pdf, section 1): severity (high first), then runs
affected (more first), then cost impact (higher first), then check ID alphabetically; the agent
name and key break any remaining tie, so the order never depends on input order.
"""

from __future__ import annotations

from collections import defaultdict

from .schema import Finding, FindingGroup, Impact, Run

SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def signature(f: Finding) -> tuple[str, str, str]:
    return (f.check_id, f.node or f.role or "", f.key)


def finding_order(f: Finding) -> tuple:
    """Within a group: most severe, then most confident, then a stable ID."""
    return (SEVERITY_RANK[f.severity], f.needs_confirmation, -f.confidence, f.run_id, f.id)


def runs_impact(run_ids: set[str], runs_by_id: dict[str, Run]) -> Impact:
    """Tokens, cost and time of the affected runs, each run counted once."""
    affected = [runs_by_id[r] for r in sorted(run_ids) if r in runs_by_id]
    return Impact(
        tokens=sum(r.totals.tokens_in + r.totals.tokens_out for r in affected),
        cost=round(sum(r.totals.cost or 0.0 for r in affected), 6),
        latency_s=round(sum(r.totals.duration_s or 0.0 for r in affected), 3),
    )


def group_findings(findings: list[Finding], runs: list[Run]) -> list[FindingGroup]:
    runs_by_id = {r.id: r for r in runs}
    buckets: dict[tuple[str, str, str], list[Finding]] = defaultdict(list)
    for f in findings:
        buckets[signature(f)].append(f)
    groups = []
    for (check_id, _, key), members in buckets.items():
        members = sorted(members, key=finding_order)
        run_ids = {f.run_id for f in members}
        representative = members[0]
        groups.append(
            FindingGroup(
                check_id=check_id,
                node=representative.node,
                role=representative.role,
                key=key,
                runs_affected=len(run_ids),
                frequency=len(run_ids) / len(runs) if runs else 0.0,
                impact=runs_impact(run_ids, runs_by_id),
                representative=representative,
                findings=members,
            )
        )
    return rank(groups)


def rank_key(g: FindingGroup) -> tuple:
    return (
        SEVERITY_RANK[g.representative.severity],
        -g.runs_affected,
        -g.impact.cost,
        g.check_id,
        g.node or "",
        g.key,
    )


def rank(groups: list[FindingGroup]) -> list[FindingGroup]:
    return sorted(groups, key=rank_key)
