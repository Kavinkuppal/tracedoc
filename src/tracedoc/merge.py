"""One problem, one finding: findings in the same run at the same agent and step are merged.

Several checks can describe the same failure from different sides. A supervisor whose final
reply is empty can trigger "Final answer was empty" (a rule), "Run finished before the task
was done" and "Agent ignored another agent's output" (AI judge), all at that one reply. Those
are one problem, so the report keeps one finding: the one with the most specific check (the
one that says what went wrong, rather than how the run ended), and lists the others as
related. Every count in the report is then a count of distinct problems.

The step is the one the report shows a finding at ("Agent and step"): its anchor, with a
LangGraph node resolved to the first model or tool call inside it. Findings with no step (about
the whole run) are never merged, and neither are cost and time findings: a slow or expensive
step is a different problem from what the step got wrong.
"""

from __future__ import annotations

from collections import defaultdict

from .schema import Finding, Related, Run, Step

SPECIFICITY: dict[str, int] = {
    # What exactly went wrong, and why.
    "agents.ignored_output": 3,
    "agents.incomplete_handoff": 3,
    "agents.no_clarification": 3,
    "agents.reason_action_mismatch": 3,
    "design.ignored_instructions": 3,
    "design.lost_context": 3,
    "design.loop": 3,
    "tools.errors_retries": 3,
    "tools.unused_output": 3,
    # The symptom at the step.
    "design.empty_answer": 2,
    "design.run_error": 2,
    "verification.unchecked_result": 2,
    # How the run as a whole ended.
    "agents.task_drift": 1,
    "design.no_stop": 1,
    "verification.premature_finish": 1,
}
"""Higher is more specific. A check not listed counts as 2; framework pitfalls count as 4."""
NEVER_MERGED = ("cost.",)
_SEVERITY = {"high": 3, "medium": 2, "low": 1}


def specificity(check_id: str) -> int:
    if check_id.startswith("framework."):
        return 4  # a known framework pitfall names the exact setting to change
    return SPECIFICITY.get(check_id, 2)


def anchor_step(f: Finding, run: Run) -> Step | None:
    """The step a finding is shown at: its first cited step at the finding's own agent, else
    its first cited step. (A finding can cite another agent's step as the cause, for example
    the reply that handed off too early; "Where" and "Agent's output" are the agent's own.)"""
    by_id = {s.id: s for s in run.steps}
    cited = [by_id[i] for i in f.evidence_step_ids if i in by_id]
    if f.method == "judge":
        # The judge cites steps in any order: show the agent's last cited model or tool call
        # (where the problem shows), not a handoff to it or an earlier step it was given.
        position = {s.id: i for i, s in enumerate(run.steps)}
        own = sorted((s for s in cited if s.node == f.node), key=lambda s: position[s.id])
        acts = [s for s in own if s.kind != "handoff"] or own
        if acts:
            return acts[-1]
    return next((s for s in cited if s.node == f.node), cited[0] if cited else None)


def shown_step(run: Run, step: Step) -> Step:
    """A LangGraph node is shown as the first model or tool call inside it."""
    if step.kind != "node":
        return step
    inner = [s for s in run.steps if s.parent_id == step.id and s.kind in ("llm", "tool")]
    return inner[0] if inner else step


def _confident(f: Finding) -> bool:
    from .report.view import confidence_level  # the report's own definition of Low

    return confidence_level(f) != "low"


def _primary_order(f: Finding) -> tuple:
    """Most specific first; a low-confidence finding never hides a confident one."""
    return (
        not _confident(f),
        -specificity(f.check_id),
        -_SEVERITY[f.severity],
        f.needs_confirmation,
        -f.confidence,
        f.check_id,
        f.id,
    )


def _related(f: Finding) -> Related:
    return Related(
        id=f.id,
        check_id=f.check_id,
        method=f.method,
        severity=f.severity,
        confidence=f.confidence,
        explanation=f.explanation,
    )


def merge_duplicates(findings: list[Finding], runs: list[Run]) -> list[Finding]:
    """Findings in the same run at the same agent and step become one finding (see the module
    docstring). Order is kept: the merged finding takes the place of the first of them."""
    runs_by_id = {r.id: r for r in runs}
    buckets: dict[tuple[str, str, str], list[Finding]] = defaultdict(list)
    keys: list[tuple[str, str, str] | None] = []
    for f in findings:
        run = runs_by_id.get(f.run_id)
        anchor = anchor_step(f, run) if run is not None and f.node else None
        if anchor is None or f.check_id.startswith(NEVER_MERGED):
            keys.append(None)
            continue
        key = (f.run_id, f.node or "", shown_step(run, anchor).id)  # type: ignore[arg-type]
        keys.append(key)
        buckets[key].append(f)

    merged: list[Finding] = []
    placed: set[tuple[str, str, str]] = set()
    for f, key in zip(findings, keys, strict=True):
        if key is None or len(buckets[key]) == 1:
            merged.append(f)
            continue
        if key in placed:
            continue
        placed.add(key)
        members = sorted(buckets[key], key=_primary_order)
        primary, others = members[0], members[1:]
        # As serious as the most serious of the findings that are as sure as it is.
        peers = [m for m in members if _confident(m) or not _confident(primary)]
        severity = max((m.severity for m in peers), key=lambda s: _SEVERITY[s])
        merged.append(
            primary.model_copy(
                update={
                    "severity": severity,
                    # The same check twice (from overlapping judge windows) is one finding.
                    "related": [
                        *primary.related,
                        *(_related(m) for m in others if m.check_id != primary.check_id),
                    ],
                }
            )
        )
    return merged
