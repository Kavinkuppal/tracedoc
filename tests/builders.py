"""Build synthetic runs for check tests that the fixture agents can't produce."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from tracedoc.schema import Run, Step, Totals

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def step(kind: str, node: str, *, dur: float = 0.1, **fields: Any) -> dict[str, Any]:
    return {"kind": kind, "node": node, "dur": dur, **fields}


def build_run(run_id: str, specs: list[dict[str, Any]], **run_fields: Any) -> Run:
    """Lay steps out back to back. llm/tool steps are parented to the latest node step."""
    clock = T0
    steps: list[Step] = []
    current_node: Step | None = None
    for i, spec in enumerate(specs):
        spec = dict(spec)
        kind, node, dur = spec.pop("kind"), spec.pop("node"), spec.pop("dur")
        role = spec.pop("role", "tool_executor" if node == "tools" else "other")
        s = Step(
            id=f"{run_id}-s{i}",
            parent_id=current_node.id if kind in ("llm", "tool") and current_node else None,
            run_id=run_id,
            kind=kind,
            name=spec.pop("name", node),
            node=node,
            role=role,
            started_at=clock,
            ended_at=clock + timedelta(seconds=dur),
            **spec,
        )
        if kind == "node":
            current_node = s
        else:
            clock += timedelta(seconds=dur)
        steps.append(s)
    llm = [s for s in steps if s.kind == "llm"]
    run_fields.setdefault("status", "completed")
    return Run(
        id=run_id,
        source="capture",
        started_at=T0,
        ended_at=clock,
        steps=steps,
        totals=Totals(
            tokens_in=sum(s.tokens_in or 0 for s in llm),
            tokens_out=sum(s.tokens_out or 0 for s in llm),
            duration_s=(clock - T0).total_seconds(),
            step_count=sum(1 for s in steps if s.kind == "node"),
        ),
        **run_fields,
    )


def ai(text: str) -> dict[str, Any]:
    return {"type": "ai", "content": text}
