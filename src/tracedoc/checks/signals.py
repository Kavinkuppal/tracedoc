"""Good-behavior signals: what went right, with evidence. Reported, never ranked as problems."""

from __future__ import annotations

from collections import defaultdict
from statistics import median

from ..config import Config
from ..messages import payload_text
from ..schema import Finding, Group, Run
from .rules.unchecked_result import verified_after_last_tool
from .util import canonical, content_tokens, coverage, excerpt


def _signal(check_id: str, group: Group, run: Run, **fields: object) -> Finding:
    return Finding(
        check_id=check_id,
        group=group,
        method="rule",
        severity="low",
        run_id=run.id,
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
                    explanation=(
                        f"After `{failed.name}` failed, the agent changed approach and "
                        f"`{recovery.name}` succeeded."
                    ),
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
                explanation=f"`{critic.node}` checked the result before the run finished.",
            )
        )
    return out


def efficient_path(runs: list[Run], config: Config) -> list[Finding]:
    """Completed in far fewer steps than the batch median for the same task type."""
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
            if run.status == "completed" and steps and steps <= t.efficient_path_ratio * med:
                out.append(
                    _signal(
                        "good.efficient_path",
                        "design",
                        run,
                        key=task,
                        evidence_excerpts=[f"{steps} steps vs median {med:g} for {task} runs"],
                        explanation=(
                            f"Finished in {steps} steps, well under the median of {med:g} for "
                            f"{task} runs."
                        ),
                    )
                )
    return out


def clean_handoff(runs: list[Run], config: Config) -> list[Finding]:
    """The receiver's first model call carries the sender's key content."""
    out = []
    for run in runs:
        llm_by_parent: dict[str, list] = defaultdict(list)
        for s in run.steps_of("llm"):
            if s.parent_id:
                llm_by_parent[s.parent_id].append(s)
        nodes = run.steps_of("node")
        for handoff in run.steps_of("handoff"):
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
            sent = content_tokens(payload_text(handoff.input))
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
                    explanation=(
                        f"`{handoff.handoff_to}` received {share:.0%} of what "
                        f"`{handoff.handoff_from}` produced."
                    ),
                )
            )
    return out


SIGNALS = (error_recovery, verified, efficient_path, clean_handoff)


def find_signals(runs: list[Run], config: Config | None = None) -> list[Finding]:
    config = config or Config()
    return [f for signal in SIGNALS for f in signal(runs, config)]
