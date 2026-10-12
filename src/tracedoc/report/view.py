"""What summary.pdf and details.pdf show, computed once, as final text and numbers.

Everything here is deterministic: text comes from the catalog (or, for AI judge findings, the
three checked slots), numbers come from the traces, and every list has a total order. The PDF
renderers only lay this out; they never decide what to say.

Definitions used throughout (also explained in details.pdf, section 1):
- Confidence: rule findings are High; AI judge findings and judge-confirmed rule candidates
  are High at 0.8 or more, Medium at 0.6 or more, Low below; unconfirmed candidates are Low.
- Serious problem: a high-severity finding with High or Medium confidence.
- Problem (agent table): a high- or medium-severity finding with High or Medium confidence.
- Fix cards (page 1): up to three problems in ranking order that are serious or moderate; a
  cost or token problem also has to be material, at least 10% of the system's total cost.
  Fewer qualify, fewer cards: the rest are "Other findings" on page 2.
- Ranking: severity, then runs affected, then cost of those runs, then check ID (group.py).
- IDs: findings F1, F2, ... in the order details.pdf lists them; runs R1, R2, ... by start.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import median
from typing import Any, Literal

from ..catalog import catalog, check_entry, check_title, fill, label, limit, text, units
from ..checks.judges.slots import fit_sentences, squash, truncate_words
from ..checks.rules.empty_answer import lost_answer
from ..checks.rules.run_error import is_control_flow, without_traceback
from ..checks.util import is_handoff_tool
from ..group import SEVERITY_RANK, group_findings
from ..messages import payload_text, reply_text
from ..schema import Finding, FindingGroup, Report, Run, Step

Level = Literal["high", "medium", "low"]

COST_CHECKS = "cost."
"""Checks about cost, tokens and time: on page 1 only when material."""
MATERIAL_COST_SHARE = 0.10
HIGH_CONFIDENCE = 0.8
MEDIUM_CONFIDENCE = 0.6
TOP_FIXES = 3
OTHER_ROWS = 8
WORKING_LINES = 3
GENERIC_GRAPH_NAMES = {"", "LangGraph", "RunnableSequence", "RunnableLambda", "graph"}
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


# ---- definitions -------------------------------------------------------------------------------


def confidence_level(f: Finding) -> Level:
    if f.needs_confirmation:
        return "low"
    if f.method == "rule" and not f.confirmed_by_judge:
        return "high"
    if f.confidence >= HIGH_CONFIDENCE:
        return "high"
    if f.confidence >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


def is_serious(f: Finding) -> bool:
    return f.severity == "high" and confidence_level(f) != "low"


def is_problem(f: Finding) -> bool:
    return f.severity in ("high", "medium") and confidence_level(f) != "low"


def fix_card_worthy(problem: ProblemView, system_cost: float | None) -> bool:
    """Serious or moderate, and for cost and token checks, at least 10% of the system's cost
    (the cost the findings themselves are about, for example the hotspot agent's calls)."""
    if problem.representative.finding.severity not in ("high", "medium"):
        return False
    if not problem.group.check_id.startswith(COST_CHECKS):
        return True
    own_cost = sum(v.finding.impact.cost for v in problem.findings)
    return bool(system_cost) and own_cost >= MATERIAL_COST_SHARE * system_cost  # type: ignore[operator]


def method_label(f: Finding) -> str:
    key = "rule_confirmed" if f.confirmed_by_judge else f.method
    return text("method", key)


# ---- formatting: every number gets a unit --------------------------------------------------------


def fmt_cost(x: float | None) -> str:
    if x is None:
        return label("tile_unknown")
    if x == 0:
        return "$0.00"
    if x < 0.1:
        return f"${x:.4f}"
    return f"${x:,.2f}"


def fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return label("tile_unknown")
    if seconds < 60:
        return units_text("seconds", f"{seconds:.1f}")
    if seconds < 3600:
        return units_text("minutes", f"{seconds / 60:.1f}")
    return units_text("hours", f"{seconds / 3600:.1f}")


def units_text(key: str, n: str) -> str:
    return fill(catalog()["units"][key], n=n)


def fmt_pct(x: float | None) -> str:
    if x is None:
        return label("tile_unknown")
    if 0 < x < 0.005:
        return "<1%"
    return f"{x:.0%}"


def utc(d: datetime) -> datetime:
    """Aware UTC time; a time without a zone is taken as UTC (never the machine's zone)."""
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)


def fmt_date(d: datetime) -> str:
    return f"{_MONTHS[d.month - 1]} {d.day}, {d.year}"


def runs_of(n: int, total: int) -> str:
    return fill(catalog()["units"]["runs_of"], n=f"{n:,}", total_runs=units("runs", total))


def _median(values: Iterable[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return median(known) if known else None


def _sum(values: Iterable[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


_SENTENCE_END = re.compile(r"[.!?…][\"'\u201d\u2019)]*$")  # an end mark, maybe inside quotes


def _sentence(text_: str) -> str:
    text_ = squash(text_)
    return text_ if not text_ or _SENTENCE_END.search(text_) else text_ + "."


# ---- view objects --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Header:
    system: str
    dates: str
    runs: str
    models: str
    cost: str


@dataclass(frozen=True)
class Tile:
    value: str
    label: str
    detail: str = ""


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: str
    result: str
    failed: bool


@dataclass
class FindingView:
    id: str
    finding: Finding
    title: str
    severity: str
    confidence: str
    method: str
    level: Level
    agent: str
    step: str
    run_id: str
    """R-number of the run."""
    request: str
    tools: list[ToolCall]
    output: str
    error: str
    """The error the cited step (or, failing that, the run) ended with; "" when none."""
    lost_answer: str | None
    """For an empty final answer: the last real answer an agent wrote before it, labeled with
    the agent and step ("" when no agent wrote one). None for other findings."""
    what_happened: str
    quote: str
    what_to_change: str
    change_kind: str
    reference: str

    @property
    def where(self) -> str:
        return f"{self.agent}, {self.step}"


@dataclass(frozen=True)
class AffectedRun:
    """One run a problem appeared in: a row of its table in details.pdf."""

    run_id: str
    """R-number of the run."""
    request: str
    what_happened: str
    """The run's first finding in the problem, as one sentence (no "In R3, ")."""
    finding_ids: tuple[str, ...]


@dataclass(frozen=True)
class OtherChange:
    """A change some findings of a problem suggest that differs from the problem's own."""

    text: str
    finding_ids: tuple[str, ...]


@dataclass
class ProblemView:
    """A group of findings: one problem at one agent, across runs."""

    group: FindingGroup
    title: str
    agent: str
    runs: int
    findings: list[FindingView]
    serious: bool
    looks_for: str = ""
    why: str = ""
    other_changes: list[OtherChange] = field(default_factory=list)
    affected: list[AffectedRun] = field(default_factory=list)
    """One row per affected run, in run order."""

    @property
    def representative(self) -> FindingView:
        """The worked example: the problem's most severe, most confident finding."""
        return self.findings[0]

    @property
    def what_to_change(self) -> str:
        rep = self.representative
        return rep.what_to_change + (f"  ({rep.change_kind})" if rep.change_kind else "")


@dataclass(frozen=True)
class Card:
    rank: int
    title: str
    where: str
    what_happened: str
    why: str
    what_to_change: str
    finding_id: str
    confidence: str
    severity: str


@dataclass(frozen=True)
class AgentRow:
    name: str
    runs: int
    problem_runs: int
    serious_runs: int
    median_cost: float | None
    median_time: float | None
    common_problem: str

    @property
    def problem_share(self) -> float:
        return self.problem_runs / self.runs if self.runs else 0.0

    @property
    def serious_share(self) -> float:
        return self.serious_runs / self.runs if self.runs else 0.0


@dataclass(frozen=True)
class OtherRow:
    problem: str
    agent: str
    runs: str


@dataclass(frozen=True)
class RunRow:
    id: str
    request: str
    outcome: str
    cost: str
    time: str
    steps: str
    findings: str


@dataclass(frozen=True)
class CatalogEntry:
    title: str
    how: str
    looks_for: str
    why: str


@dataclass
class ReportView:
    header: Header
    created: datetime
    """Latest run end (UTC): the PDFs' creation date, so it depends only on the traces."""
    verdict: str
    tiles: list[Tile]
    cards: list[Card]
    other_rows: list[OtherRow]
    other_hidden: int
    """Problems not on summary.pdf (more than fit, plus the low-confidence ones)."""
    fewer_fixes: bool
    """Fewer than three fix cards, while other problems exist (they didn't qualify)."""
    working: list[str]
    agents: list[AgentRow]
    entry_agent: str | None
    edges: list[tuple[str, str, int]]
    main_problems: list[ProblemView]
    low_problems: list[ProblemView]
    runs: list[RunRow]
    caps: list[str]
    checks: list[CatalogEntry]
    signals: list[CatalogEntry]
    run_count: int
    finding_pages: dict[str, int] = field(default_factory=dict)
    """Finding ID -> page in details.pdf, filled in when details.pdf is rendered."""

    @property
    def findings(self) -> list[FindingView]:
        return [f for p in [*self.main_problems, *self.low_problems] for f in p.findings]


# ---- building ----------------------------------------------------------------------------------


def _system_name(runs: list[Run], override: str | None) -> str:
    if override:
        return override
    counts = Counter(r.name for r in runs if r.name not in GENERIC_GRAPH_NAMES)
    names = [n for n, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    if not names:
        generic = Counter(r.name for r in runs if r.name)
        return min(generic, key=lambda n: (-generic[n], n)) if generic else label("system_unnamed")
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return label("systems_two", first=names[0], second=names[1])
    return label("systems_many", first=names[0], second=names[1], n=len(names) - 2)


def _dates(runs: list[Run]) -> tuple[str, datetime]:
    starts = sorted(utc(r.started_at) for r in runs if r.started_at)
    ends = sorted(
        utc(r.ended_at or r.started_at)  # type: ignore[arg-type]
        for r in runs
        if r.ended_at or r.started_at
    )
    if not starts:
        return label("tile_unknown"), EPOCH
    first, last = starts[0], ends[-1]
    if first.date() == last.date():
        return label("dates_one", date=fmt_date(first)), last
    return label("dates_range", start=fmt_date(first), end=fmt_date(last)), last


def _models(runs: list[Run]) -> str:
    counts = Counter(s.model for r in runs for s in r.steps_of("llm") if s.model)
    if not counts:
        return label("tile_unknown")
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = [label("model_calls", model=m, calls=units("calls", n)) for m, n in ordered[:3]]
    if len(ordered) > 3:
        return label("models_more", models=", ".join(shown), n=len(ordered) - 3)
    return ", ".join(shown)


def _analysis_cost(meta: dict[str, Any]) -> str:
    if not meta.get("judged"):
        return label("cost_rules_only")
    if meta.get("analysis_unpriced_models"):
        return label("cost_unpriced")
    reused = int(meta.get("analysis_answers_reused", 0))
    answers = int(meta.get("analysis_calls", 0)) + reused
    key = "cost_judged_cached" if reused else "cost_judged"
    return label(key, cost=fmt_cost(meta.get("analysis_cost", 0.0)), answers=answers)


def _step_label(run: Run | None, step_id: str | None) -> str:
    steps_text = catalog()["steps"]
    if run is None or step_id is None:
        return steps_text["whole_run"]
    by_id = {s.id: s for s in run.steps}
    step = by_id.get(step_id)
    if step is not None and step.kind == "node":  # name the first model or tool call inside it
        inner = [s for s in run.steps if s.parent_id == step.id and s.kind in ("llm", "tool")]
        if inner:
            step = inner[0]
        else:
            k = [s.id for s in run.steps_of("node")].index(step.id) + 1
            return fill(steps_text["step"], k=k)
    if step is None:
        return steps_text["whole_run"]
    if step.kind == "llm":
        llms = run.steps_of("llm")
        mine = [s.id for s in llms if s.node == step.node]
        if step.error:
            return fill(steps_text["failed_call"], k=mine.index(step.id) + 1)
        if llms and llms[-1].id == step.id:
            return steps_text["final_reply"]
        return fill(steps_text["reply"], k=mine.index(step.id) + 1)
    if step.kind == "tool":
        same = [s.id for s in run.steps_of("tool") if s.name == step.name]
        return fill(steps_text["tool"], tool=step.name, k=same.index(step.id) + 1)
    return fill(steps_text["handoff"], to=step.handoff_to or label("unknown_agent"))


def _step_text(step: Step) -> str:
    return squash(payload_text(step.input if step.kind == "handoff" else step.output))


def _anchor(f: Finding, run: Run) -> Step | None:
    """The step a finding is shown at: its first cited step at the finding's own agent, else
    its first cited step. (A finding can cite another agent's step as the cause, for example
    the reply that handed off too early; "Where" and "Agent's output" are the agent's own.)"""
    by_id = {s.id: s for s in run.steps}
    cited = [by_id[i] for i in f.evidence_step_ids if i in by_id]
    return next((s for s in cited if s.node == f.node), cited[0] if cited else None)


def _lost_answer(f: Finding, run: Run | None, anchor: Step | None) -> str | None:
    if f.check_id != "design.empty_answer" or run is None:
        return None
    step = lost_answer(run, anchor)
    if step is None:
        return label("lost_answer_none")
    return label(
        "lost_answer_value",
        text=truncate_words(squash(reply_text(step.output)), 500),
        agent=step.node or label("unknown_agent"),
        step=_step_label(run, step.id),
    )


def _error(f: Finding, run: Run | None, anchor: Step | None) -> str:
    error = anchor.error if anchor is not None else None
    if not error and run is not None and f.check_id == "design.run_error":
        error = run.error
    if not error or is_control_flow(error):
        return ""
    return truncate_words(squash(without_traceback(payload_text(error))), 600)


def _tool_calls(f: Finding, run: Run, anchor: Step | None) -> list[ToolCall]:
    """The tool calls a finding cites; else the agent's last few before the cited step; else
    the run's last few before it (what the agent had to work with). Handoff tools
    (transfer_to_...) carry no data, so they are left out."""
    data_tools = [s for s in run.steps if s.kind == "tool" and not is_handoff_tool(s.name)]
    cited = [s for s in data_tools if s.id in set(f.evidence_step_ids)]
    if not cited:
        position = {s.id: i for i, s in enumerate(run.steps)}
        cutoff = position.get(anchor.id, len(run.steps)) if anchor else len(run.steps)
        before = [s for s in data_tools if position[s.id] <= cutoff]
        cited = [s for s in before if s.node == f.node][-3:] or before[-3:]
    return [
        ToolCall(
            name=s.name,
            args=truncate_words(payload_text(s.input), 160),
            result=truncate_words(s.error or payload_text(s.output), 260),
            failed=bool(s.error),
        )
        for s in cited[:4]
    ]


def _finding_view(
    fid: str, f: Finding, run: Run | None, run_ids: dict[str, str], total: int
) -> FindingView:
    entry = check_entry(f.check_id)
    level = confidence_level(f)
    anchor = _anchor(f, run) if run is not None else None
    rid = run_ids.get(f.run_id, f.run_id)
    template = "finding_what_happened_judge" if f.method == "judge" else "finding_what_happened"
    output = ""
    if anchor is not None and anchor.kind in ("llm", "handoff"):
        output = _step_text(anchor)
    elif run is not None:
        output = squash(run.final_answer)
    return FindingView(
        id=fid,
        finding=f,
        title=entry["title"],
        severity=text("severity", f.severity),
        confidence=text("confidence", level),
        method=method_label(f),
        level=level,
        agent=f.node or label("unknown_agent"),
        step=_step_label(run, anchor.id if anchor else None),
        run_id=rid,
        request=truncate_words(run.input_preview, 600) if run else "",
        tools=_tool_calls(f, run, anchor) if run else [],
        output=truncate_words(output, 500) or label("no_output"),
        error=_error(f, run, anchor),
        lost_answer=_lost_answer(f, run, anchor),
        what_happened=text("card", template, run=rid, text=_sentence(f.explanation)),
        quote=f.evidence_quote,
        what_to_change=_sentence(f.suggested_fix),
        change_kind=label("change_kind", kind=f.change_type) if f.change_type else "",
        reference=f.id,
    )


def _ordered_findings(group: FindingGroup, run_index: dict[str, int]) -> list[Finding]:
    return sorted(
        group.findings,
        key=lambda f: (
            SEVERITY_RANK[f.severity],
            f.needs_confirmation,
            -f.confidence,
            run_index.get(f.run_id, len(run_index)),
            f.id,
        ),
    )


def _affected(views: list[FindingView], run_index: dict[str, int]) -> list[AffectedRun]:
    by_run: dict[str, list[FindingView]] = defaultdict(list)
    for v in views:
        by_run[v.finding.run_id].append(v)
    ordered = sorted(by_run.items(), key=lambda kv: (run_index.get(kv[0], len(run_index)), kv[0]))
    rows = []
    for _, in_run in ordered:
        in_run = sorted(in_run, key=lambda v: int(v.id[1:]))
        first = in_run[0]
        rows.append(
            AffectedRun(
                run_id=first.run_id,
                request=first.request or label("no_output"),
                what_happened=_sentence(first.finding.explanation),
                finding_ids=tuple(v.id for v in in_run),
            )
        )
    return rows


def _other_changes(views: list[FindingView]) -> list[OtherChange]:
    """Changes suggested by findings of the problem that differ from the worked example's, each
    listed once (AI judge findings each write their own)."""
    own = views[0].what_to_change
    ids: dict[str, list[str]] = defaultdict(list)
    for v in views[1:]:
        if v.what_to_change and v.what_to_change != own:
            ids[v.what_to_change].append(v.id)
    return [OtherChange(text, tuple(found)) for text, found in ids.items()]


def _problems(
    findings: list[Finding],
    runs: list[Run],
    run_ids: dict[str, str],
    run_index: dict[str, int],
    first_id: int,
) -> list[ProblemView]:
    by_id = {r.id: r for r in runs}
    problems = []
    n = first_id
    for group in group_findings(findings, runs):
        views = []
        for f in _ordered_findings(group, run_index):
            n += 1
            views.append(_finding_view(f"F{n}", f, by_id.get(f.run_id), run_ids, len(runs)))
        entry = check_entry(group.check_id)
        problems.append(
            ProblemView(
                group=group,
                title=check_title(group.check_id),
                agent=group.node or label("unknown_agent"),
                runs=group.runs_affected,
                findings=views,
                serious=any(is_serious(v.finding) for v in views),
                looks_for=entry["looks_for"],
                why=entry["why"],
                other_changes=_other_changes(views),
                affected=_affected(views, run_index),
            )
        )
    return problems


def _card(rank: int, problem: ProblemView, runs_by_id: dict[str, Run]) -> Card:
    total = len(runs_by_id)
    rep = problem.representative
    texts = {squash(v.finding.explanation) for v in problem.findings}
    clause = _sentence(rep.finding.explanation)
    if problem.runs == 1:
        key = "what_happened_one"
    elif len(texts) == 1 and rep.finding.method == "rule":
        key = "what_happened"
    else:
        key = "what_happened_example"

    def frame(t: str) -> str:
        return text(
            "card",
            key,
            runs=problem.runs,
            total_runs=units("runs", total),
            run=rep.run_id,
            text=t,
        )

    affected = sorted({v.finding.run_id for v in problem.findings})
    cost = _sum(runs_by_id[r].totals.cost for r in affected if r in runs_by_id)
    why_values = {
        "why": check_entry(problem.group.check_id)["why"],
        "runs": problem.runs,
        "total_runs": units("runs", total),
        "pct": fmt_pct(problem.runs / total if total else None),
        "time": fmt_time(problem.group.impact.latency_s),
    }
    why = text(
        "card", "why" if cost is not None else "why_unpriced", cost=fmt_cost(cost), **why_values
    )
    return Card(
        rank=rank,
        title=problem.title,
        where=rep.where,
        # Too long for the card: end at a sentence boundary (details.pdf has the full text).
        what_happened=fit_sentences(clause, limit("what_happened"), frame),
        why=why,
        what_to_change=fit_sentences(rep.what_to_change, limit("what_to_change")),
        finding_id=rep.id,
        confidence=label("confidence_value", level=rep.confidence, method=rep.method),
        severity=rep.finding.severity,
    )


def _agent_rows(runs: list[Run], findings: list[Finding]) -> list[AgentRow]:
    runs_with: dict[str, set[str]] = defaultdict(set)
    first_seen: dict[str, int] = {}
    cost: dict[str, dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))
    time: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for run in runs:
        for s in run.steps:
            if not s.node:
                continue
            if s.kind == "llm":
                runs_with[s.node].add(run.id)
                first_seen.setdefault(s.node, len(first_seen))
                cost[s.node][run.id].append(s.cost)
            if s.kind in ("llm", "tool") and s.duration_s is not None:
                time[s.node][run.id] += s.duration_s
    problem_runs: dict[str, set[str]] = defaultdict(set)
    serious_runs: dict[str, set[str]] = defaultdict(set)
    common: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[tuple[str, str, str]] = set()
    for f in findings:
        if not f.node or not is_problem(f):
            continue
        problem_runs[f.node].add(f.run_id)
        if is_serious(f):
            serious_runs[f.node].add(f.run_id)
        if (f.node, f.run_id, f.check_id) not in seen:
            seen.add((f.node, f.run_id, f.check_id))
            common[f.node][f.check_id] += 1
    rows = []
    for agent, agent_runs in runs_with.items():
        counts = common.get(agent)
        top = min(counts, key=lambda c: (-counts[c], check_title(c))) if counts else None
        rows.append(
            AgentRow(
                name=agent,
                runs=len(agent_runs),
                problem_runs=len(problem_runs[agent] & agent_runs),
                serious_runs=len(serious_runs[agent] & agent_runs),
                median_cost=_median(_sum(cost[agent][r]) for r in sorted(agent_runs)),
                median_time=_median(time[agent].get(r) for r in sorted(agent_runs)),
                common_problem=check_title(top) if top else label("no_problem"),
            )
        )
    return sorted(rows, key=lambda a: (-a.runs, first_seen[a.name], a.name))


def _edges(runs: list[Run], agents: set[str]) -> list[tuple[str, str, int]]:
    counts = Counter(
        (h.handoff_from, h.handoff_to)
        for r in runs
        for h in r.steps_of("handoff")
        if h.handoff_from in agents and h.handoff_to in agents and h.handoff_from != h.handoff_to
    )
    return sorted(((a, b, n) for (a, b), n in counts.items()), key=lambda e: (e[0], e[1]))


def _entry_agent(runs: list[Run]) -> str | None:
    firsts = Counter(next((s.node for s in r.steps_of("llm") if s.node), None) for r in runs)
    firsts.pop(None, None)
    return min(firsts, key=lambda a: (-firsts[a], a)) if firsts else None  # type: ignore[arg-type]


def _working(signals: list[Finding], total: int) -> list[str]:
    runs: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for s in signals:
        runs[(s.check_id, s.node or "", s.facts.get("sender", ""))].add(s.run_id)
    ordered = sorted(runs.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    lines = []
    for (check_id, agent, sender), affected in ordered[:WORKING_LINES]:
        lines.append(
            fill(
                check_entry(check_id)["line"],
                agent=agent or label("unknown_agent"),
                sender=sender or label("unknown_agent"),
                runs=len(affected),
                total_runs=units("runs", total),
            )
        )
    return lines


def _caps(meta: dict[str, Any], config: Any) -> list[str]:
    how = catalog()["how_to_read"]
    if not meta.get("judged"):
        return [how["caps_off"]]
    out = []
    unreviewed = meta.get("unreviewed_by_judge") or {}
    jc = getattr(config, "judge", None)
    candidates = sum((unreviewed.get("rule_candidates") or {}).values())
    if candidates:
        out.append(
            fill(
                how["caps_candidates"],
                n=candidates,
                cap=getattr(jc, "max_windows_per_signature", "?"),
            )
        )
    if unreviewed.get("handoffs"):
        out.append(
            fill(
                how["caps_handoffs"],
                n=unreviewed["handoffs"],
                per_run=getattr(jc, "max_handoffs_per_run", "?"),
                cap=getattr(jc, "max_windows_per_signature", "?"),
            )
        )
    if unreviewed.get("unsampled_runs"):
        out.append(
            fill(
                how["caps_sample"],
                n=unreviewed["unsampled_runs"],
                sample=getattr(jc, "sample_runs", "?"),
            )
        )
    if meta.get("stopped_for_budget"):
        out.append(fill(how["caps_budget"], reason=meta["stopped_for_budget"]))
    return out or [how["caps_none"]]


def _catalog_entries() -> tuple[list[CatalogEntry], list[CatalogEntry]]:
    cat = catalog()
    how = {"rule": text("method", "rule"), "judge": text("method", "judge")}
    checks = sorted(
        (
            CatalogEntry(e["title"], how[e["how"]], e["looks_for"], e["why"])
            for e in cat["checks"].values()
        ),
        key=lambda c: (c.how != how["rule"], c.title),
    )
    signals = sorted(
        (
            CatalogEntry(e["title"], how["rule"], e["looks_for"], "")
            for e in cat["signals"].values()
        ),
        key=lambda c: c.title,
    )
    return checks, signals


def _median_steps(runs: list[Run]) -> str:
    if not runs:
        return label("tile_unknown")
    m = median(r.totals.step_count for r in runs)
    return units("steps", int(m) if m == int(m) else m)  # type: ignore[arg-type]


def build_view(
    report: Report,
    runs: list[Run],
    *,
    name: str | None = None,
    config: Any = None,
) -> ReportView:
    total = len(runs)
    ordered_runs = sorted(runs, key=lambda r: (utc(r.started_at or EPOCH), r.id))
    run_ids = {r.id: f"R{i + 1}" for i, r in enumerate(ordered_runs)}
    run_index = {r.id: i for i, r in enumerate(ordered_runs)}
    findings = [f for g in report.groups for f in g.findings]
    eligible = [f for f in findings if confidence_level(f) != "low"]
    low = [f for f in findings if confidence_level(f) == "low"]
    main = _problems(eligible, runs, run_ids, run_index, 0)
    low_problems = _problems(low, runs, run_ids, run_index, len(eligible))

    serious_runs = {f.run_id for f in eligible if is_serious(f)}
    serious_groups = [p for p in main if p.serious]
    if serious_runs:
        top = min(serious_groups, key=lambda p: -p.runs)  # min keeps ranking order on ties
        verdict = text(
            "verdict",
            "serious",
            n=len(serious_runs),
            total_runs=units("runs", total),
            title=top.title,
            k=units("runs", top.runs),
        )
    elif main:
        top = min(main, key=lambda p: -p.runs)
        verdict = text(
            "verdict",
            "none_other_one" if total == 1 else "none_other",
            total_runs=units("runs", total),
            title=top.title,
            k=units("runs", top.runs),
        )
    else:
        verdict = text(
            "verdict", "none_one" if total == 1 else "none", total_runs=units("runs", total)
        )

    costs = [r.totals.cost for r in runs]
    total_cost = _sum(costs)
    serious_cost = _sum(r.totals.cost for r in runs if r.id in serious_runs)
    tiles = [
        Tile(
            label(
                "tile_serious_value",
                n=len(serious_runs),
                pct=fmt_pct(len(serious_runs) / total if total else None),
            ),
            label("tile_serious"),
            label("tile_serious_detail", total_runs=units("runs", total)),
        ),
        Tile(fmt_cost(_median(costs)), label("tile_median_cost")),
        Tile(fmt_time(_median(r.totals.duration_s for r in runs)), label("tile_median_time")),
        Tile(_median_steps(runs), label("tile_median_steps")),
        Tile(
            fmt_pct((serious_cost or 0.0) / total_cost if total_cost else None),
            label("tile_serious_cost"),
        ),
    ]

    runs_by_id = {r.id: r for r in runs}
    fixes = [p for p in main if fix_card_worthy(p, total_cost)][:TOP_FIXES]
    cards = [_card(i + 1, p, runs_by_id) for i, p in enumerate(fixes)]
    rest = [p for p in main if p not in fixes]  # ranking order; minor findings land here
    other_rows = [OtherRow(p.title, p.agent, runs_of(p.runs, total)) for p in rest[:OTHER_ROWS]]
    agents = _agent_rows(ordered_runs, findings)
    dates, created = _dates(runs)
    checks, signals = _catalog_entries()
    meta = report.metadata
    by_run: dict[str, list[str]] = defaultdict(list)
    for p in [*main, *low_problems]:
        for v in p.findings:
            by_run[v.finding.run_id].append(v.id)
    return ReportView(
        header=Header(
            system=_system_name(runs, name),
            dates=dates,
            runs=units("runs", total),
            models=_models(runs),
            cost=_analysis_cost(meta),
        ),
        created=created,
        verdict=verdict,
        tiles=tiles,
        cards=cards,
        other_rows=other_rows,
        other_hidden=len(rest) - len(other_rows) + len(low_problems),
        fewer_fixes=len(cards) < TOP_FIXES and len(rest) > 0,
        working=_working(report.signals, total),
        agents=agents,
        entry_agent=_entry_agent(runs),
        edges=_edges(runs, {a.name for a in agents}),
        main_problems=main,
        low_problems=low_problems,
        runs=[
            RunRow(
                id=run_ids[r.id],
                request=truncate_words(r.input_preview, 160) or label("no_output"),
                outcome=text("outcome", r.status),
                cost=fmt_cost(r.totals.cost),
                time=fmt_time(r.totals.duration_s),
                steps=f"{r.totals.step_count:,}",
                findings=", ".join(sorted(by_run.get(r.id, []), key=lambda i: int(i[1:])))
                or label("no_problem"),
            )
            for r in ordered_runs
        ],
        caps=_caps(meta, config),
        checks=checks,
        signals=signals,
        run_count=total,
    )
