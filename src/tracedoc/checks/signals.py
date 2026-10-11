"""Good-behavior signals: what went right, with evidence. Reported, never ranked as problems."""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import median
from typing import Any

from ..catalog import check_entry, fill
from ..config import Config
from ..messages import payload_text, reply_text
from ..schema import Finding, Group, Run, Step
from .rules.unchecked_result import verified_after_last_tool
from .util import canonical, content_tokens, coverage, excerpt, is_handoff_tool


def _signal(
    check_id: str, group: Group, run: Run, facts: dict[str, Any], **fields: object
) -> Finding:
    """A good-behavior finding; its sentence comes from the catalog, filled with `facts`."""
    template = check_entry(check_id)["what_happened"]
    return Finding(
        check_id=check_id,
        group=group,
        method="rule",
        severity="low",
        run_id=run.id,
        explanation=fill(template, **facts),
        facts={k: str(v) for k, v in facts.items()},
        **fields,  # type: ignore[arg-type]
    )


def error_recovery(runs: list[Run], config: Config) -> list[Finding]:
    """A tool error followed by a different, successful tool call."""
    out = []
    for run in runs:
        tools = run.steps_of("tool")
        for i, failed in enumerate(tools):
            if not failed.error:
                continue
            recovery = next(
                (
                    s
                    for s in tools[i + 1 :]
                    if not s.error
                    and (s.name != failed.name or canonical(s.input) != canonical(failed.input))
                ),
                None,
            )
            if recovery is None:
                continue
            out.append(
                _signal(
                    "good.error_recovery",
                    "tools_cost",
                    run,
                    key=failed.name,
                    node=failed.node,
                    role=failed.role,
                    evidence_step_ids=[failed.id, recovery.id],
                    evidence_excerpts=[
                        f"failed: {failed.name}({excerpt(failed.input, 80)})",
                        f"then succeeded: {recovery.name}({excerpt(recovery.input, 80)})",
                    ],
                    facts={
                        "agent": failed.node or "the agent",
                        "failed": failed.name,
                        "recovered": recovery.name,
                    },
                )
            )
            break  # one per run is enough
    return out


def verified(runs: list[Run], config: Config) -> list[Finding]:
    """A verification step after the last tool output, before the end."""
    out = []
    for run in runs:
        if not verified_after_last_tool(run):
            continue
        critic = [s for s in run.steps_of("node") if s.role == "critic"][-1]
        out.append(
            _signal(
                "good.verified",
                "verification",
                run,
                key=critic.node or critic.name,
                node=critic.node,
                role="critic",
                evidence_step_ids=[critic.id],
                evidence_excerpts=[excerpt(critic.output, 140)],
                facts={"agent": critic.node or "the agent"},
            )
        )
    return out


def only_asks(answer: str) -> bool:
    """The reply only asks the user something: every sentence in it is a question."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    return bool(sentences) and all(s.endswith("?") for s in sentences)


def delivered(run: Run) -> bool:
    """An answer reached the user, it did more than ask a question, and a tool was used.
    (Findings are checked later: a run with any finding is dropped when the report is built.)"""
    used_tool = any(not is_handoff_tool(s.name) for s in run.steps_of("tool"))
    answer = run.final_answer.strip()
    return run.status == "completed" and bool(answer) and not only_asks(answer) and used_tool


def efficient_path(runs: list[Run], config: Config) -> list[Finding]:
    """Delivered an answer with at least one tool, in far fewer steps than the median for the
    same task type."""
    t = config.thresholds
    by_task: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        by_task[run.task_type].append(run)
    out = []
    for task, group in by_task.items():
        if len(group) < t.efficient_path_min_runs:
            continue
        med = median(r.totals.step_count for r in group)
        for run in group:
            steps = run.totals.step_count
            if delivered(run) and steps and steps <= t.efficient_path_ratio * med:
                out.append(
                    _signal(
                        "good.efficient_path",
                        "design",
                        run,
                        key=task,
                        evidence_excerpts=[f"{steps} steps vs median {med:g} for {task} runs"],
                        facts={"steps": steps, "median": f"{med:g}", "task": task},
                    )
                )
    return out


def _sender_wrote(run: Run, handoff: Step) -> str:
    """What the sending agent itself wrote since it last got control: the words of its model
    replies, not tool calls, and not the user's request or other agents' work it was handed."""
    sender = handoff.handoff_from
    since = max(
        (
            h.ended_at
            for h in run.steps_of("handoff")
            if h.handoff_to == sender
            and h.ended_at
            and handoff.started_at
            and h.ended_at <= handoff.started_at
        ),
        default=None,
    )
    replies = [
        reply_text(s.output)
        for s in run.steps_of("llm")
        if s.node == sender
        and s.ended_at
        and handoff.started_at
        and s.ended_at <= handoff.started_at
        and (since is None or (s.started_at is not None and s.started_at >= since))
    ]
    return " ".join(r for r in replies if r)


def clean_handoff(runs: list[Run], config: Config) -> list[Finding]:
    """A handoff from one agent to another where the receiver's first model call carries what
    the sender wrote. The user's request, which every agent sees, doesn't count."""
    out = []
    for run in runs:
        llm_by_parent: dict[str, list] = defaultdict(list)
        for s in run.steps_of("llm"):
            if s.parent_id:
                llm_by_parent[s.parent_id].append(s)
        nodes = run.steps_of("node")
        request = content_tokens(run.input_preview)
        for handoff in run.steps_of("handoff"):
            if not handoff.handoff_from or handoff.handoff_from == handoff.handoff_to:
                continue
            receiver = next(
                (
                    n
                    for n in nodes
                    if n.node == handoff.handoff_to
                    and n.started_at
                    and handoff.ended_at
                    and n.started_at >= handoff.ended_at
                ),
                None,
            )
            calls = llm_by_parent.get(receiver.id, []) if receiver else []
            sent = content_tokens(_sender_wrote(run, handoff)) - request
            if not calls or len(sent) < config.thresholds.unused_output_min_tokens:
                continue
            received = content_tokens(payload_text(calls[0].input))
            share = coverage(sent, received)
            if share < config.thresholds.clean_handoff_min_overlap:
                continue
            out.append(
                _signal(
                    "good.clean_handoff",
                    "agents",
                    run,
                    key=f"{handoff.handoff_from}->{handoff.handoff_to}",
                    node=handoff.handoff_to,
                    role=handoff.role,
                    evidence_step_ids=[handoff.id, calls[0].id],
                    evidence_excerpts=[excerpt(handoff.input, 140)],
                    facts={
                        "agent": handoff.handoff_to or "the agent",
                        "sender": handoff.handoff_from or "the previous agent",
                        "share": f"{share:.0%}",
                    },
                )
            )
    return out


SIGNALS = (error_recovery, verified, efficient_path, clean_handoff)


def find_signals(runs: list[Run], config: Config | None = None) -> list[Finding]:
    config = config or Config()
    return [f for signal in SIGNALS for f in signal(runs, config)]
