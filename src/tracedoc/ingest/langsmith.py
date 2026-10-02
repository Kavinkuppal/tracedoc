"""Import traces from LangSmith: a live project via the SDK, or an exported file.

An export file is JSON (a list of runs, or `{"runs": [...]}`) or JSONL (one run
per line), where each run has the fields of `langsmith.schemas.Run`. All runs
of a trace must be present, not only the roots.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..messages import as_message, to_jsonable
from .base import RawSpan, RawTrace, SpanKind, parse_time

_KIND: dict[str, SpanKind] = {"llm": "llm", "tool": "tool", "retriever": "tool"}
_ERROR_TYPE = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*[(:]")


def _error_type(error: str) -> str | None:
    """LangSmith stores `repr(exc)` plus a traceback: `GraphRecursionError('...')`."""
    match = _ERROR_TYPE.match(error)
    return match.group(1).rsplit(".", 1)[-1] if match else None


def _first_generation_message(outputs: Any) -> tuple[Any, dict[str, Any]]:
    """Return (message-or-text, raw message kwargs) from an LLM run's outputs."""
    if not isinstance(outputs, Mapping):
        return outputs, {}
    generations = outputs.get("generations")
    if isinstance(generations, list) and generations:
        first = generations[0]
        if isinstance(first, list) and first:
            first = first[0]
        if isinstance(first, Mapping):
            raw = first.get("message")
            if raw is not None:
                kwargs = (
                    raw.get("kwargs") or raw.get("data") or raw if isinstance(raw, Mapping) else {}
                )
                return as_message(raw) or raw, dict(kwargs)
            return first.get("text"), {}
    message = as_message(outputs)
    if message is not None:
        return message, dict(outputs)
    return outputs, {}


def _llm_inputs(inputs: Any) -> Any:
    if isinstance(inputs, Mapping) and isinstance(inputs.get("messages"), list):
        messages = inputs["messages"]
        if messages and isinstance(messages[0], list) and len(messages) == 1:
            messages = messages[0]
        return {"messages": to_jsonable(messages)}
    return to_jsonable(inputs)


def _unwrap(payload: Any, key: str) -> Any:
    """LangSmith wraps non-dict tool inputs/outputs as {"input": ...} / {"output": ...}."""
    if isinstance(payload, Mapping) and set(payload) == {key}:
        return payload[key]
    return payload


def span_from_run(run: Mapping[str, Any]) -> RawSpan:
    extra = run.get("extra") or {}
    metadata = dict(extra.get("metadata") or {})
    kind = _KIND.get(str(run.get("run_type")), "chain")
    error = run.get("error")
    span = RawSpan(
        id=str(run["id"]),
        parent_id=str(run["parent_run_id"]) if run.get("parent_run_id") else None,
        kind=kind,
        name=str(run.get("name") or run.get("run_type") or "run"),
        started_at=parse_time(run.get("start_time")),
        ended_at=parse_time(run.get("end_time")),
        error=str(error) if error else None,
        error_type=_error_type(str(error)) if error else None,
        tags=list(run.get("tags") or []),
        metadata={k: v for k, v in metadata.items() if k.startswith(("langgraph_", "ls_"))},
        ended=run.get("end_time") is not None,
    )
    if run.get("total_cost") is not None:
        span.cost = float(run["total_cost"])
    inputs, outputs = run.get("inputs"), run.get("outputs")
    if kind == "llm":
        span.inputs = _llm_inputs(inputs)
        message, kwargs = _first_generation_message(outputs)
        span.outputs = to_jsonable(message)
        usage = kwargs.get("usage_metadata") or {}
        token_usage = ((outputs or {}).get("llm_output") or {}).get("token_usage") or {}
        span.tokens_in = (
            run.get("prompt_tokens")
            or usage.get("input_tokens")
            or token_usage.get("prompt_tokens")
        )
        span.tokens_out = (
            run.get("completion_tokens")
            or usage.get("output_tokens")
            or token_usage.get("completion_tokens")
        )
        params = extra.get("invocation_params") or {}
        response_meta = kwargs.get("response_metadata") or {}
        span.model = (
            response_meta.get("model_name")
            or metadata.get("ls_model_name")
            or params.get("model")
            or params.get("model_name")
        )
    elif kind == "tool":
        span.inputs = to_jsonable(_unwrap(inputs, "input"))
        output = to_jsonable(_unwrap(outputs, "output"))
        if isinstance(output, Mapping) and output.get("type") == "tool":
            output = output.get("content")
        span.outputs = output
    else:
        span.inputs = to_jsonable(inputs)
        span.outputs = to_jsonable(outputs)
    return span


def traces_from_runs(runs: Iterable[Mapping[str, Any]]) -> list[RawTrace]:
    by_trace: dict[str, list[RawSpan]] = defaultdict(list)
    for run in runs:
        trace_id = str(run.get("trace_id") or run["id"])
        by_trace[trace_id].append(span_from_run(run))
    traces = []
    for trace_id, spans in by_trace.items():
        spans.sort(key=lambda s: (s.started_at is None, s.started_at))
        is_langgraph = any("langgraph_node" in s.metadata for s in spans)
        traces.append(
            RawTrace(
                id=trace_id,
                source="langsmith",
                spans=spans,
                framework="langgraph" if is_langgraph else "langchain",
            )
        )
    return traces


def read_export(path: Path) -> list[RawTrace]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        runs = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        runs = data["runs"] if isinstance(data, Mapping) else data
    return traces_from_runs(runs)


def _run_to_dict(run: Any) -> dict[str, Any]:
    if isinstance(run, Mapping):
        return dict(run)
    if hasattr(run, "model_dump"):
        return run.model_dump()
    return run.dict()


def fetch_project(project: str, *, limit: int | None = 100, client: Any = None) -> list[RawTrace]:
    """Pull the most recent root runs of a project and all of their child runs."""
    if client is None:
        try:
            from langsmith import Client
        except ImportError as e:  # pragma: no cover
            raise ImportError("LangSmith import needs `pip install tracedoc[langsmith]`") from e
        client = Client()
    runs: list[dict[str, Any]] = []
    for root in client.list_runs(project_name=project, is_root=True, limit=limit):
        trace_id = getattr(root, "trace_id", None) or root.id
        runs.extend(
            _run_to_dict(r) for r in client.list_runs(project_name=project, trace_id=trace_id)
        )
    return traces_from_runs(runs)
