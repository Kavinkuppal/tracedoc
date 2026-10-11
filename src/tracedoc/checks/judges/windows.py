"""Windows: the slice of a run a judge sees, rendered as text.

A window is a few steps either side of a focus (a rule candidate's evidence, a handoff,
or the end of a run), plus the user's task and the system prompts of the agents involved.
Node steps are not rendered (their inputs are the whole graph state); model calls, tool
calls, and handoffs are. Each rendered step gets a short label (S1, S2, ...) that the
judge cites and we map back to step IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ...messages import payload_text
from ...schema import Finding, Run, Step

WindowKind = Literal["handoff", "run", "candidate"]


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]} […{len(text) - limit} chars omitted…] {text[-half:]}"


def _messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [m for m in value if isinstance(m, dict) and "type" in m]
    return []


@dataclass
class Window:
    kind: WindowKind
    run: Run
    steps: list[Step]
    focus: list[str]
    candidate: Finding | None = None
    labels: dict[str, str] = field(default_factory=dict)
    """label -> step id"""

    def __post_init__(self) -> None:
        self.labels = {f"S{i + 1}": s.id for i, s in enumerate(self.steps)}

    @property
    def key(self) -> str:
        return f"{self.run.id}:{self.kind}:{','.join(self.focus)}"

    def step_for(self, label: str) -> Step | None:
        step_id = self.labels.get(label.strip())
        return next((s for s in self.steps if s.id == step_id), None)

    def render(self, max_chars: int) -> str:
        out = [f"USER TASK: {_clip(self.run.input_preview, max_chars)}"]
        prompts = self._system_prompts()
        if prompts:
            out.append("\nAGENT INSTRUCTIONS (system prompts):")
            out += [f"- {node}: {_clip(text, max_chars)}" for node, text in prompts.items()]
        first = self.run.steps.index(self.steps[0]) if self.steps else 0
        if first > 0:
            out.append(f"\n(... {first} earlier steps not shown ...)")
        out.append("\nSTEPS:")
        focus = set(self.focus)
        for label, step in zip(self.labels, self.steps, strict=True):
            out.append(self._render_step(label, step, step.id in focus, max_chars))
        if self.steps and self.steps[-1] is self._last_action():
            out.append(
                f"\nRUN ENDED with status '{self.run.status}'. "
                f"Final output: {_clip(self.run.output_preview, max_chars)}"
            )
        return "\n".join(out)

    def _last_action(self) -> Step | None:
        actions = [s for s in self.run.steps if s.kind != "node"]
        return actions[-1] if actions else None

    def _system_prompts(self) -> dict[str, str]:
        """Each acting agent's own instructions: the latest system prompt it ran with in the
        window, keyed by the agent's name (never an inner LangGraph node name)."""
        prompts: dict[str, str] = {}
        for step in self.steps:
            if step.kind != "llm" or not step.node:
                continue
            system = [m for m in _messages(step.input) if m["type"] == "system"]
            if system:
                prompts[step.node] = payload_text(system[-1])
        return dict(sorted(prompts.items()))

    def _render_step(self, label: str, step: Step, focused: bool, limit: int) -> str:
        mark = "  <-- focus" if focused else ""
        where = f"agent={step.node}"
        if step.kind == "llm":
            recent = [m for m in _messages(step.input) if m["type"] != "system"][-2:]
            lines = [f"[{label}] MODEL CALL · {where}{mark}", "  latest input messages:"]
            lines += [f"    {m['type']}: {_clip(payload_text(m), limit)}" for m in recent]
            lines.append(f"  output: {_clip(payload_text(step.output), limit)}")
        elif step.kind == "tool":
            lines = [f"[{label}] TOOL CALL {step.name} · {where}{mark}"]
            lines.append(f"  arguments: {_clip(payload_text(step.input), limit)}")
            if step.error:
                lines.append(f"  ERROR: {_clip(step.error, limit)}")
            else:
                lines.append(f"  result: {_clip(payload_text(step.output), limit)}")
        else:  # handoff
            lines = [f"[{label}] HANDOFF {step.handoff_from} -> {step.handoff_to}{mark}"]
            lines.append(f"  sender produced: {_clip(payload_text(step.input), limit)}")
        return "\n".join(lines)


def _actions(run: Run) -> list[Step]:
    return [s for s in run.steps if s.kind != "node"]


def window_around(run: Run, step_ids: list[str], k: int, kind: WindowKind) -> Window:
    actions = _actions(run)
    index = {s.id: i for i, s in enumerate(actions)}
    # Node-step evidence maps to the actions inside that node.
    positions = [index[i] for i in step_ids if i in index] or [
        index[s.id] for s in actions if s.parent_id in set(step_ids)
    ]
    if not positions:
        positions = [len(actions) - 1]
    lo, hi = max(0, min(positions) - k), min(len(actions), max(positions) + k + 1)
    return Window(kind=kind, run=run, steps=actions[lo:hi], focus=list(step_ids))


def run_end_window(run: Run, k: int) -> Window:
    actions = _actions(run)
    steps = actions[-(2 * k + 1) :]
    return Window(kind="run", run=run, steps=steps, focus=[steps[-1].id] if steps else [])
