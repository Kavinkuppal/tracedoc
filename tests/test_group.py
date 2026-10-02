from __future__ import annotations

from tracedoc.group import group_findings
from tracedoc.schema import Finding, Impact


def _f(check: str, run: str, severity: str = "medium", **kw: object) -> Finding:
    return Finding(
        check_id=check,
        group="design",
        method="rule",
        severity=severity,
        run_id=run,
        explanation="x",
        **kw,  # type: ignore[arg-type]
    )


def test_groups_by_signature_and_counts_runs() -> None:
    findings = [
        _f("design.loop", "r1", key="search", node="agent", impact=Impact(tokens=100)),
        _f("design.loop", "r2", key="search", node="agent", impact=Impact(tokens=50)),
        _f("design.loop", "r2", key="fetch", node="agent"),
    ]
    groups = group_findings(findings, total_runs=4)
    search = next(g for g in groups if g.key == "search")
    assert search.runs_affected == 2
    assert search.frequency == 0.5
    assert search.impact.tokens == 150
    assert len(groups) == 2


def test_rank_severity_then_frequency_then_impact() -> None:
    findings = [
        _f("a.low", "r1", "low"),
        _f("b.med", "r1", "medium", key="x"),
        _f("c.med", "r1", "medium", key="y"),
        _f("c.med", "r2", "medium", key="y"),
        _f("d.med", "r3", "medium", key="z", impact=Impact(tokens=999)),
        _f("e.high", "r1", "high"),
    ]
    order = [g.check_id for g in group_findings(findings, 3)]
    assert order == ["e.high", "c.med", "d.med", "b.med", "a.low"]


def test_unconfirmed_candidates_rank_after_confirmed_at_same_severity() -> None:
    findings = [
        _f("cand", "r1", needs_confirmation=True),
        _f("cand", "r2", needs_confirmation=True),
        _f("sure", "r3"),
    ]
    assert [g.check_id for g in group_findings(findings, 3)] == ["sure", "cand"]
