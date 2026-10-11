from __future__ import annotations

import random

from tracedoc.group import group_findings, rank
from tracedoc.schema import Finding, Run, Totals


def _f(check: str, run: str, severity: str = "medium", **kw: object) -> Finding:
    return Finding(
        check_id=check,
        group="design",
        method="rule",
        severity=severity,  # type: ignore[arg-type]
        run_id=run,
        explanation="x",
        **kw,  # type: ignore[arg-type]
    )


def _run(run_id: str, cost: float | None = 0.01, seconds: float = 2.0, tokens: int = 100) -> Run:
    return Run(
        id=run_id,
        source="capture",
        status="completed",
        totals=Totals(tokens_in=tokens, tokens_out=0, cost=cost, duration_s=seconds),
    )


RUNS = [_run("r1", 0.10, 10.0, 1000), _run("r2", 0.02, 4.0, 300), _run("r3"), _run("r4")]


def test_impact_is_the_affected_runs_totals_each_counted_once() -> None:
    findings = [
        _f("design.loop", "r1", key="search", node="agent"),
        _f("design.loop", "r2", key="search", node="agent"),
        _f("design.loop", "r2", key="search", node="agent", evidence_step_ids=["s9"]),
        _f("design.loop", "r2", key="fetch", node="agent"),
    ]
    groups = group_findings(findings, RUNS)
    search = next(g for g in groups if g.key == "search")
    assert search.runs_affected == 2
    assert search.frequency == 0.5
    assert search.impact.tokens == 1300  # r1 + r2, r2 counted once
    assert search.impact.cost == 0.12
    assert search.impact.latency_s == 14.0
    assert len(groups) == 2


def test_unpriced_runs_contribute_no_cost() -> None:
    groups = group_findings([_f("a.b", "x")], [_run("x", cost=None)])
    assert groups[0].impact.cost == 0.0
    assert groups[0].impact.tokens == 100


def test_rank_severity_then_runs_then_cost_then_check_id() -> None:
    runs = [_run("r1", 0.01), _run("r2", 0.01), _run("r3", 0.50)]
    findings = [
        _f("z.low", "r1", "low"),
        _f("b.med", "r1", "medium"),  # 1 run, cheap
        _f("c.med", "r1", "medium"),  # 2 runs
        _f("c.med", "r2", "medium"),
        _f("d.med", "r3", "medium"),  # 1 run, expensive
        _f("a.med", "r1", "medium"),  # ties b.med on runs and cost: check ID decides
        _f("e.high", "r1", "high"),
    ]
    order = [g.check_id for g in group_findings(findings, runs)]
    assert order == ["e.high", "c.med", "d.med", "a.med", "b.med", "z.low"]


def test_ranking_ignores_input_order() -> None:
    findings = [
        _f(f"c{i % 3}.x", f"r{i % 4 + 1}", ("high", "medium", "low")[i % 3], node=f"n{i % 2}")
        for i in range(24)
    ]
    expected = [(g.check_id, g.node, g.key) for g in group_findings(findings, RUNS)]
    for seed in range(5):
        shuffled = findings[:]
        random.Random(seed).shuffle(shuffled)
        got = [(g.check_id, g.node, g.key) for g in group_findings(shuffled, RUNS)]
        assert got == expected
        assert [g.representative.id for g in rank(group_findings(shuffled, RUNS))] == [
            g.representative.id for g in group_findings(findings, RUNS)
        ]


def test_unconfirmed_candidates_rank_by_the_same_rules() -> None:
    findings = [
        _f("cand", "r1", needs_confirmation=True),
        _f("cand", "r2", needs_confirmation=True),
        _f("sure", "r3"),
    ]
    # Ranking does not demote unconfirmed findings; the report filters them by confidence.
    assert [g.check_id for g in group_findings(findings, RUNS)] == ["cand", "sure"]
