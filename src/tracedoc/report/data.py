"""The report's data: compact row tables, plus the per-section aggregates computed from them.

The HTML report embeds the rows and re-computes the same aggregates in JavaScript when the
reader changes filters (`templates/app.js` mirrors `aggregate` below; a test checks they agree).
Python computes the unfiltered aggregates for report.json, report.md, and snapshot tests.

Filter semantics, shared by both sides:
  - the run set is runs matching the task type, that visited the node, and that have a finding
    of the severity (each filter only when set);
  - findings are those in the run set at the node and of the severity;
  - steps are those in the run set at the node. The flow map and handoff matrix use all steps of
    the run set, since a single node has no edges.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import pairwise
from typing import Any

from ..catalog import catalog
from ..messages import payload_text
from ..schema import Finding, Report, Run, Step

SEVERITIES = ("high", "medium", "low")
STATUSES = ("completed", "no_answer", "error", "recursion_limit", "interrupted")
CONTEXT_GROWTH_MAX_CALLS = 50


def check_titles() -> dict[str, str]:
    """Every check's and signal's title, from the reviewed catalog."""
    cat = catalog()
    return {cid: entry["title"] for cid, entry in [*cat["checks"].items(), *cat["signals"].items()]}


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _step_io(step: Step, limit: int) -> tuple[str, str]:
    if step.kind == "llm" and isinstance(step.input, list):
        recent = [m for m in step.input if isinstance(m, dict) and m.get("type") != "system"]
        inp = payload_text(recent[-1]) if recent else ""
    elif step.kind == "handoff":
        return _clip(payload_text(step.input), limit), _clip(payload_text(step.output), limit)
    else:
        inp = payload_text(step.input)
    out = step.error or payload_text(step.output)
    return _clip(inp, limit), _clip(out, limit)


def _r(x: float | None, digits: int = 4) -> float | None:
    return None if x is None else round(x, digits)


def build_rows(
    runs: list[Run], findings: list[Finding], signals: list[Finding], text_limit: int
) -> dict[str, Any]:
    run_rows, step_rows = [], []
    for run in runs:
        t0 = run.started_at
        nodes = sorted({s.node for s in run.steps_of("node") if s.node})
        run_rows.append(
            {
                "id": run.id,
                "task": run.task_type,
                "status": run.status,
                "steps": run.totals.step_count,
                "tin": run.totals.tokens_in,
                "tout": run.totals.tokens_out,
                "cost": _r(run.totals.cost, 6),
                "dur": _r(run.totals.duration_s),
                "nodes": nodes,
                "input": _clip(run.input_preview, text_limit),
                "output": _clip(run.output_preview, text_limit),
                "error": _clip(run.error or "", text_limit),
            }
        )
        for s in run.steps:
            inp, out = _step_io(s, text_limit)
            step_rows.append(
                {
                    "run": run.id,
                    "id": s.id,
                    "parent": s.parent_id,
                    "kind": s.kind,
                    "name": s.name,
                    "node": s.node,
                    "role": s.role,
                    "t": _r((s.started_at - t0).total_seconds()) if s.started_at and t0 else None,
                    "dur": _r(s.duration_s),
                    "tin": s.tokens_in,
                    "tout": s.tokens_out,
                    "cost": _r(s.cost, 6),
                    "err": bool(s.error),
                    "from": s.handoff_from,
                    "to": s.handoff_to,
                    "in": inp,
                    "out": out,
                }
            )

    def finding_row(f: Finding) -> dict[str, Any]:
        return {
            "id": f.id,
            "check": f.check_id,
            "group": f.group,
            "method": f.method,
            "severity": f.severity,
            "run": f.run_id,
            "node": f.node,
            "role": f.role,
            "key": f.key,
            "explanation": f.explanation,
            "fix": f.suggested_fix,
            "confidence": f.confidence,
            "unconfirmed": f.needs_confirmation,
            "evidence": f.evidence_step_ids,
            "excerpts": [_clip(e, text_limit) for e in f.evidence_excerpts],
            "tokens": f.impact.tokens,
            "impact_cost": _r(f.impact.cost, 6),
            "latency": _r(f.impact.latency_s),
        }

    return {
        "runs": run_rows,
        "steps": step_rows,
        "findings": [finding_row(f) for f in findings],
        "signals": [finding_row(f) for f in signals],
    }


# ---- aggregation (mirrored in templates/app.js) -----------------------------------


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def select(
    rows: dict[str, Any], task: str | None = None, node: str | None = None, sev: str | None = None
) -> tuple[list[dict], list[dict], list[dict]]:
    sev_runs = {f["run"] for f in rows["findings"] if f["severity"] == sev} if sev else None
    runs = [
        r
        for r in rows["runs"]
        if (task is None or r["task"] == task)
        and (node is None or node in r["nodes"])
        and (sev_runs is None or r["id"] in sev_runs)
    ]
    ids = {r["id"] for r in runs}
    findings = [
        f
        for f in rows["findings"]
        if f["run"] in ids
        and (node is None or f["node"] == node)
        and (sev is None or f["severity"] == sev)
    ]
    steps = [s for s in rows["steps"] if s["run"] in ids]
    return runs, findings, steps


def aggregate(
    rows: dict[str, Any], task: str | None = None, node: str | None = None, sev: str | None = None
) -> dict[str, Any]:
    runs, findings, all_steps = select(rows, task, node, sev)
    steps = [s for s in all_steps if node is None or s["node"] == node]
    n = len(runs)

    # 1. scorecard
    costs = sorted(r["cost"] for r in runs if r["cost"] is not None)
    durs = sorted(r["dur"] for r in runs if r["dur"] is not None)
    scorecard = {
        "runs": n,
        "completion_rate": _r(sum(r["status"] == "completed" for r in runs) / n) if n else 0.0,
        "median_steps": _r(quantile(sorted(r["steps"] for r in runs), 0.5)),
        "total_cost": _r(sum(costs), 6) if costs else None,
        "median_cost": _r(quantile(costs, 0.5), 6) if costs else None,
        "median_latency_s": _r(quantile(durs, 0.5)) if durs else None,
        "tokens": sum(r["tin"] + r["tout"] for r in runs),
        "findings_by_severity": {s: sum(f["severity"] == s for f in findings) for s in SEVERITIES},
    }

    # 3. failure frequency: runs affected per check, with the worst severity seen
    by_check: dict[str, dict[str, Any]] = {}
    for f in findings:
        entry = by_check.setdefault(f["check"], {"runs": set(), "severity": "low"})
        entry["runs"].add(f["run"])
        if SEVERITIES.index(f["severity"]) < SEVERITIES.index(entry["severity"]):
            entry["severity"] = f["severity"]
    frequency = sorted(
        (
            {"check": c, "runs": len(e["runs"]), "severity": e["severity"]}
            for c, e in by_check.items()
        ),
        key=lambda x: (-x["runs"], SEVERITIES.index(x["severity"]), x["check"]),
    )

    # 4. failures by task type: runs affected per (task, check)
    task_of = {r["id"]: r["task"] for r in runs}
    cells: dict[tuple[str, str], set[str]] = defaultdict(set)
    for f in findings:
        cells[(task_of[f["run"]], f["check"])].add(f["run"])
    tasks = sorted({r["task"] for r in runs})
    runs_per_task = Counter(r["task"] for r in runs)
    by_task = {
        "tasks": tasks,
        "checks": [x["check"] for x in frequency],
        "cells": sorted(
            [t, c, len(v), _r(len(v) / runs_per_task[t])] for (t, c), v in cells.items()
        ),
    }

    # 5. how runs end
    status_by_task = {t: {s: 0 for s in STATUSES} for t in tasks}
    for r in runs:
        status_by_task[r["task"]][r["status"]] += 1

    # 6. agent flow map (all steps of the run set)
    node_steps = defaultdict(list)
    for s in all_steps:
        if s["kind"] == "node" and s["parent"] is None:
            node_steps[s["run"]].append(s)
    edges: Counter[tuple[str, str]] = Counter()
    visits: Counter[str] = Counter()
    node_runs: dict[str, set[str]] = defaultdict(set)
    status_of = {r["id"]: r["status"] for r in runs}
    for run_id, seq in node_steps.items():
        seq.sort(key=lambda s: (s["t"] is None, s["t"]))
        path = ["__start__"] + [s["node"] for s in seq]
        if status_of[run_id] == "completed":
            path.append("__end__")
        for a, b in pairwise(path):
            edges[(a, b)] += 1
        for s in seq:
            visits[s["node"]] += 1
            node_runs[s["node"]].add(run_id)
    failed_at: dict[str, set[str]] = defaultdict(set)
    all_findings_in_set = [f for f in rows["findings"] if f["run"] in task_of]
    for f in all_findings_in_set:
        if f["node"] and (sev is None or f["severity"] == sev):
            failed_at[f["node"]].add(f["run"])
    flow = {
        "nodes": sorted(
            [
                name,
                visits[name],
                _r(len(failed_at[name] & node_runs[name]) / len(node_runs[name])),
            ]
            for name in visits
        ),
        "edges": sorted([a, b, c] for (a, b), c in edges.items()),
    }

    # 7. handoffs: count per (from, to), and share of runs with that pair that had a finding on it
    pair_count: Counter[tuple[str, str]] = Counter()
    pair_runs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for s in all_steps:
        if s["kind"] == "handoff":
            pair = (s["from"], s["to"])
            pair_count[pair] += 1
            pair_runs[pair].add(s["run"])
    pair_failed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for f in all_findings_in_set:
        if "->" in f["key"] and (sev is None or f["severity"] == sev):
            a, b = f["key"].split("->", 1)
            pair_failed[(a, b)].add(f["run"])
    handoffs = sorted(
        [a, b, c, _r(len(pair_failed[(a, b)] & pair_runs[(a, b)]) / len(pair_runs[(a, b)]))]
        for (a, b), c in pair_count.items()
    )

    # 8. cost by node (model calls)
    tok: Counter[str] = Counter()
    cost: dict[str, float] = defaultdict(float)
    for s in steps:
        if s["kind"] == "llm" and s["node"]:
            tok[s["node"]] += (s["tin"] or 0) + (s["tout"] or 0)
            cost[s["node"]] += s["cost"] or 0.0
    tok_total, cost_total = sum(tok.values()), sum(cost.values())
    cost_by_node = sorted(
        (
            {
                "node": k,
                "tokens": tok[k],
                "token_share": _r(tok[k] / tok_total) if tok_total else 0.0,
                "cost": _r(cost[k], 6),
                "cost_share": _r(cost[k] / cost_total) if cost_total else None,
            }
            for k in tok
        ),
        key=lambda x: (-x["tokens"], x["node"]),
    )

    # 9. context growth: input tokens per model call, median across runs and the worst run
    per_run: dict[str, list[int]] = defaultdict(list)
    for s in sorted(steps, key=lambda s: (s["run"], s["t"] is None, s["t"])):
        if s["kind"] == "llm" and s["tin"] is not None:
            per_run[s["run"]].append(s["tin"])
    length = min(max((len(v) for v in per_run.values()), default=0), CONTEXT_GROWTH_MAX_CALLS)
    median_line = [
        _r(quantile(sorted(v[i] for v in per_run.values() if len(v) > i), 0.5))
        for i in range(length)
    ]
    worst_id = max(per_run, key=lambda k: (max(per_run[k]), k), default=None)
    context = {
        "median": median_line,
        "worst_run": worst_id,
        "worst": per_run[worst_id][:CONTEXT_GROWTH_MAX_CALLS] if worst_id else [],
    }

    # 10. steps per run
    steps_hist = sorted([k, v] for k, v in Counter(r["steps"] for r in runs).items())

    # 11. latency by node: box stats of node-step durations
    durations: dict[str, list[float]] = defaultdict(list)
    for s in steps:
        if s["kind"] == "node" and s["dur"] is not None and s["node"]:
            durations[s["node"]].append(s["dur"])
    latency = sorted(
        [
            k,
            [_r(quantile(sorted(v), q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)],
            len(v),
        ]
        for k, v in durations.items()
    )

    return {
        "scorecard": scorecard,
        "failure_frequency": frequency,
        "failures_by_task": by_task,
        "status_by_task": status_by_task,
        "flow": flow,
        "handoffs": handoffs,
        "cost_by_node": cost_by_node,
        "context_growth": context,
        "steps_per_run": steps_hist,
        "latency_by_node": latency,
    }


def node_stats(agg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for name, visits, failure_rate in agg["flow"]["nodes"]:
        stats[name] = {"visits": visits, "failure_rate": failure_rate}
    for row in agg["cost_by_node"]:
        stats.setdefault(row["node"], {}).update(
            {k: row[k] for k in ("tokens", "token_share", "cost", "cost_share")}
        )
    for name, (_lo, _q1, med, q3, hi), n in agg["latency_by_node"]:
        stats.setdefault(name, {}).update(
            {"latency_median_s": med, "latency_p75_s": q3, "latency_max_s": hi, "latency_n": n}
        )
    return stats


def fill_report(report: Report, rows: dict[str, Any]) -> Report:
    """Attach the unfiltered aggregates to the Report (node_stats, distributions)."""
    agg = aggregate(rows)
    report.node_stats = node_stats(agg)
    report.distributions = {k: v for k, v in agg.items() if k != "scorecard"}
    return report


def payload(report: Report, rows: dict[str, Any]) -> dict[str, Any]:
    """Everything the HTML page needs, as one JSON-able dict."""
    return {
        "report": json.loads(report.model_dump_json(exclude={"groups": {"__all__": {"findings"}}})),
        "rows": rows,
        "titles": check_titles(),
    }
