"""Turn raw traces from any source into the schema.

Order matters: redaction runs first, so nothing after it (role mapping, task
typing, checks, judges, reports) can see redacted content.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from itertools import pairwise
from typing import Any

from .classify.roles import RoleClassifier, map_role
from .classify.task_type import TaskClassifier, classify_task
from .config import Config
from .ingest.base import RawSpan, RawTrace
from .messages import message_text, payload_text
from .redact import Redactor
from .schema import Role, Run, RunStatus, Step, Totals

PriceFn = Callable[[str | None, int, int], "float | None"]
"""Given (model, tokens_in, tokens_out), return a cost in USD or None if unknown."""

_RECURSION_ERRORS = {"GraphRecursionError"}
_INTERRUPT_ERRORS = {"GraphInterrupt", "NodeInterrupt"}
_INTERRUPT_KEY = "__interrupt__"


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _messages(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping) and isinstance(payload.get("messages"), list):
        return [m for m in payload["messages"] if isinstance(m, Mapping) and "type" in m]
    return []


def _main_text(payload: Any) -> str:
    """A state dict with exactly one string field reads better as that string."""
    if isinstance(payload, Mapping):
        strings = [v for v in payload.values() if isinstance(v, str)]
        if len(payload) == 1 and len(strings) == 1:
            return strings[0]
    return payload_text(payload)


def _input_preview(root: RawSpan, limit: int) -> str:
    messages = _messages(root.inputs)
    humans = [m for m in messages if m.get("type") == "human"]
    if humans:
        return _truncate(payload_text(humans[-1]), limit)
    return _truncate(_main_text(root.inputs), limit)


def _output_preview(root: RawSpan, node_steps: list[Step], limit: int) -> str:
    messages = _messages(root.outputs)
    if messages:
        return _truncate(payload_text(messages[-1]), limit)
    for step in reversed(node_steps):
        if step.error is None and step.output:
            messages = _messages(step.output)
            if messages:
                return _truncate(payload_text(messages[-1]), limit)
            return _truncate(_main_text(step.output), limit)
    return _truncate(_main_text(root.outputs), limit)


def _has_text(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _final_answer(root: RawSpan, node_steps: list[Step]) -> str | None:
    """The text of the final reply, "" if it had none, or None if unknowable.

    With a message list, the last message is the reply: an agent message counts only for its
    text (not its tool calls); a final tool or user message means the agent never answered.
    Without messages, the last step's state update is the reply, if it has any text fields.
    """
    messages = _messages(root.outputs)
    if messages:
        last = messages[-1]
        if last.get("type") != "ai":
            return ""
        return message_text({"content": last.get("content", "")})
    for step in reversed(node_steps):
        if step.error is None and isinstance(step.output, Mapping):
            messages = _messages(step.output)
            if messages:
                last = messages[-1]
                return (
                    message_text({"content": last.get("content", "")})
                    if last.get("type") == "ai"
                    else ""
                )
            strings = [v for v in step.output.values() if isinstance(v, str)]
            return " ".join(strings) if strings else None
    return None


def _status(root: RawSpan) -> RunStatus:
    if root.error_type in _RECURSION_ERRORS:
        return "recursion_limit"
    if root.error_type in _INTERRUPT_ERRORS:
        return "interrupted"
    if isinstance(root.outputs, Mapping) and root.outputs.get(_INTERRUPT_KEY):
        return "interrupted"
    if root.error is not None or not root.ended:
        return "error"
    return "completed"


_INTERNAL_NODES = {"__start__", "__end__"}


def _ns_path(span: RawSpan) -> list[str]:
    """LangGraph node names from the outermost graph inward, e.g. ["research_agent", "agent"].

    Read from the checkpoint namespace ("research_agent:<task>|agent:<task>"), which LangGraph
    attaches to every event inside a node, including nested subgraphs.
    """
    ns = span.metadata.get("langgraph_checkpoint_ns") or span.metadata.get("checkpoint_ns")
    if isinstance(ns, str) and ns:
        return [seg.split(":", 1)[0] for seg in ns.split("|") if seg]
    node = span.metadata.get("langgraph_node")
    return [str(node)] if node else []


def _is_node(span: RawSpan, kept_parent: RawSpan | None) -> bool:
    if span.kind != "chain":
        return False
    node = span.metadata.get("langgraph_node")
    if node is None or node != span.name or node in _INTERNAL_NODES:
        return False
    # A runnable inside a node inherits the node's metadata; only the outermost counts.
    return not (
        kept_parent is not None
        and kept_parent.kind == "chain"
        and kept_parent.metadata.get("langgraph_node") == node
        and kept_parent.metadata.get("langgraph_checkpoint_ns")
        == span.metadata.get("langgraph_checkpoint_ns")
    )


def _step_payloads(span: RawSpan) -> tuple[Any, Any]:
    if span.kind == "llm" and isinstance(span.inputs, Mapping):
        if "messages" in span.inputs:
            return span.inputs["messages"], span.outputs
        if "prompts" in span.inputs:
            return span.inputs["prompts"], span.outputs
    return span.inputs, span.outputs


def _redact_span(span: RawSpan, redact: Redactor) -> RawSpan:
    return replace(
        span,
        inputs=redact(span.inputs),
        outputs=redact(span.outputs),
        error=redact.text(span.error) if span.error else None,
    )


def _infer_handoffs(run_id: str, node_steps: list[Step], agents: set[str]) -> list[Step]:
    """A handoff is control passing from one agent to a different agent.

    Steps that only run tools, and steps of plain function nodes that never call a model,
    are transparent: agent -> tools -> agent is not a handoff.
    """
    turns = [s for s in node_steps if s.role != "tool_executor" and s.node in agents]
    handoffs: list[Step] = []
    for prev, cur in pairwise(turns):
        if prev.node == cur.node:
            continue
        handoffs.append(
            Step(
                id=f"{run_id}:handoff:{len(handoffs)}",
                parent_id=None,
                run_id=run_id,
                kind="handoff",
                name=f"{prev.node} -> {cur.node}",
                role=cur.role,
                node=cur.node,
                graph_step=cur.graph_step,
                started_at=prev.ended_at,
                ended_at=cur.started_at,
                input=prev.output,
                output=cur.input,
                handoff_from=prev.node,
                handoff_to=cur.node,
            )
        )
    return handoffs


def normalize(
    trace: RawTrace,
    config: Config | None = None,
    *,
    task_classifier: TaskClassifier | None = None,
    role_classifier: RoleClassifier | None = None,
    price: PriceFn | None = None,
) -> Run:
    config = config or Config()
    redact = Redactor(config.redact)
    for span in trace.spans:
        redact.learn(span.inputs)
        redact.learn(span.outputs)
    spans = [_redact_span(s, redact) for s in trace.spans]
    by_id = {s.id: s for s in spans}
    root = trace.root
    if root is None:
        raise ValueError(f"trace {trace.id} has no spans")
    root = by_id[root.id]

    # Walk spans in start order, keeping nodes, model calls, and tool calls.
    order = sorted(
        (s for s in spans if s.id != root.id),
        key=lambda s: (s.started_at is None, s.started_at),
    )
    # `nearest[x]` is x itself if kept, else x's nearest kept ancestor (root at the top).
    # Parents start before their children, so one pass in start order is enough.
    nearest: dict[str, RawSpan] = {root.id: root}
    parent_of: dict[str, RawSpan] = {}
    kept: list[RawSpan] = []
    is_node: set[str] = set()
    has_direct_work: set[str] = set()  # node spans with model or tool calls directly inside
    wrappers: set[str] = set()  # node spans that contain other node spans (nested subgraphs)
    for span in order:
        parent = nearest.get(span.parent_id or "", root)
        parent_of[span.id] = parent
        if span.kind in ("llm", "tool") or _is_node(span, parent):
            kept.append(span)
            if span.kind == "chain":
                is_node.add(span.id)
                if parent.id in is_node:
                    wrappers.add(parent.id)
            elif parent.id in is_node:
                has_direct_work.add(parent.id)
            nearest[span.id] = span
        else:
            nearest[span.id] = parent

    # A wrapper around a nested subgraph is not a step of its own; its inner steps are.
    dropped = wrappers - has_direct_work
    kept = [s for s in kept if s.id not in dropped]

    def parent_step_id(span: RawSpan) -> str | None:
        parent = parent_of[span.id]
        while parent.id in dropped:
            parent = parent_of[parent.id]
        return None if parent.id == root.id else parent.id

    # Agents are the outermost graph nodes that made at least one model call. Outermost nodes
    # that only run tools (a shared ToolNode) are credited to the agent that asked for the
    # tool: the agent that acted most recently before them.
    top_of = {s.id: (_ns_path(s) or [s.name])[0] for s in kept}
    agents = {top_of[s.id] for s in kept if s.kind == "llm"}
    tool_tops = {top_of[s.id] for s in kept if s.kind == "tool"} - agents
    agent_of: dict[str, str] = {}
    last_agent: str | None = None
    for span in kept:
        top = top_of[span.id]
        if top in agents:
            last_agent = top
            agent_of[span.id] = top
        elif top in tool_tops and last_agent is not None:
            agent_of[span.id] = last_agent
        else:
            agent_of[span.id] = top

    run_id = trace.id
    steps: list[Step] = []
    for span in kept:
        step_input, step_output = _step_payloads(span)
        cost = span.cost
        if cost is None and price is not None and span.kind == "llm":
            cost = price(span.model, span.tokens_in or 0, span.tokens_out or 0)
        graph_step = span.metadata.get("langgraph_step")
        path = _ns_path(span)
        steps.append(
            Step(
                id=span.id,
                parent_id=parent_step_id(span),
                run_id=run_id,
                kind="node" if span.id in is_node else span.kind,  # type: ignore[arg-type]
                name=span.name,
                node=agent_of[span.id],
                graph_node="/".join(path) if path else None,
                graph_step=int(graph_step) if graph_step is not None else None,
                started_at=span.started_at,
                ended_at=span.ended_at,
                input=step_input,
                output=step_output,
                model=span.model,
                tokens_in=span.tokens_in,
                tokens_out=span.tokens_out,
                cost=cost,
                error=span.error,
            )
        )

    # Roles: steps that only run tools are tool executors; everything else takes the role of
    # its agent, mapped from the agent's name (then the optional model fallback).
    node_steps = [s for s in steps if s.kind == "node"]
    children: dict[str, list[Step]] = {}
    for step in steps:
        if step.parent_id is not None:
            children.setdefault(step.parent_id, []).append(step)
    role_of_agent: dict[str, Role] = {}
    for name in sorted({s.node for s in steps if s.node}):
        role_of_agent[name] = map_role(name, fallback=role_classifier)
    for step in steps:
        kids = children.get(step.id, [])
        tool_only_node = step.kind == "node" and bool(kids) and all(k.kind == "tool" for k in kids)
        if step.kind == "tool" or tool_only_node:
            step.role = "tool_executor"
        elif step.node is not None:
            step.role = role_of_agent[step.node]

    steps.extend(_infer_handoffs(run_id, node_steps, agents))
    steps.sort(key=lambda s: (s.started_at is None, s.started_at, s.kind != "handoff"))

    llm_steps = [s for s in steps if s.kind == "llm"]
    costs = [s.cost for s in llm_steps if s.cost is not None]
    duration = (
        (root.ended_at - root.started_at).total_seconds()
        if root.started_at and root.ended_at
        else None
    )
    totals = Totals(
        tokens_in=sum(s.tokens_in or 0 for s in llm_steps),
        tokens_out=sum(s.tokens_out or 0 for s in llm_steps),
        cost=sum(costs) if costs else None,
        duration_s=duration,
        step_count=len(node_steps),
    )
    input_preview = _input_preview(root, config.preview_chars)
    status = _status(root)
    final = _final_answer(root, node_steps) if status == "completed" else None
    if final is not None and not _has_text(final):
        status = "no_answer"
    return Run(
        id=run_id,
        source=trace.source,
        started_at=root.started_at,
        ended_at=root.ended_at,
        status=status,
        error=root.error,
        input_preview=input_preview,
        output_preview=_output_preview(root, node_steps, config.preview_chars),
        final_answer=(final or "")[:4_000],
        task_type=classify_task(input_preview, fallback=task_classifier),
        steps=steps,
        totals=totals,
        framework=trace.framework,
        framework_version=trace.framework_version,
    )
