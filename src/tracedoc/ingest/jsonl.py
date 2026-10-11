"""Read traces written by `TraceCapture`."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..capture.callback import FORMAT
from .base import RawSpan, RawTrace, parse_time

_START_KIND = {
    "chain_start": "chain",
    "chat_model_start": "llm",
    "llm_start": "llm",
    "tool_start": "tool",
}


class CaptureFormatError(ValueError):
    pass


def _apply_event(spans: dict[str, RawSpan], record: dict[str, Any]) -> None:
    event = record["event"]
    run_id = record["run_id"]
    data = record.get("data") or {}
    ts = parse_time(record.get("ts"))
    if event in _START_KIND:
        metadata = record.get("metadata") or {}
        span = RawSpan(
            id=run_id,
            parent_id=record.get("parent_run_id"),
            kind=_START_KIND[event],  # type: ignore[arg-type]
            name=record.get("name") or event.removesuffix("_start"),
            started_at=ts,
            tags=list(record.get("tags") or []),
            metadata=metadata,
        )
        if event == "chain_start":
            span.inputs = data.get("inputs")
        elif event == "chat_model_start":
            span.inputs = {"messages": data.get("messages")}
            span.model = data.get("model") or metadata.get("ls_model_name")
        elif event == "llm_start":
            span.inputs = {"prompts": data.get("prompts")}
            span.model = data.get("model") or metadata.get("ls_model_name")
        else:
            span.inputs = data.get("input")
        spans[run_id] = span
        return
    span = spans.get(run_id)
    if span is None:
        return
    span.ended_at = ts
    span.ended = True
    if event.endswith("_error"):
        span.error = data.get("error")
        span.error_type = data.get("error_type")
    elif event == "chain_end":
        span.outputs = data.get("outputs")
    elif event == "llm_end":
        span.outputs = data.get("output")
        usage = data.get("usage") or {}
        span.tokens_in = usage.get("input_tokens")
        span.tokens_out = usage.get("output_tokens")
        span.model = data.get("model") or span.model
    elif event == "tool_end":
        span.outputs = data.get("output")


def read_capture_file(path: Path) -> RawTrace:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            # A crash can leave a partial last line; anything else is corruption.
            if line_no == len(lines) and line_no > 1:
                break
            raise CaptureFormatError(f"{path}:{line_no}: invalid JSON") from e
    if not records or records[0].get("event") != "trace_start":
        raise CaptureFormatError(f"{path}: missing trace_start header")
    header = records[0]
    if header.get("format") != FORMAT:
        raise CaptureFormatError(f"{path}: not a {FORMAT} file")
    spans: dict[str, RawSpan] = {}
    for record in records[1:]:
        _apply_event(spans, record)
    return RawTrace(
        id=header["run_id"],
        source="capture",
        spans=list(spans.values()),
        framework=header.get("framework"),
        framework_version=header.get("framework_version"),
    )


SKIPPED_DIRS = {"__pycache__", "node_modules", "site-packages"}


def iter_files(path: Path, suffixes: tuple[str, ...]) -> Iterator[Path]:
    """`path` itself if it is a file, else every matching file below it, in sorted order.
    Hidden folders (.venv, .git, ...) and package folders are not searched."""
    if path.is_file():
        yield path
        return
    found = []
    for folder, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIPPED_DIRS)
        found += [Path(folder) / f for f in files if f.endswith(suffixes)]
    yield from sorted(found)


def is_capture_file(path: Path) -> bool:
    """Whether the file starts with a tracedoc capture header (cheap: reads one line)."""
    try:
        with path.open(encoding="utf-8") as f:
            first = f.readline()
        header = json.loads(first)
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return isinstance(header, dict) and header.get("format") == FORMAT


def iter_capture_files(path: Path) -> Iterator[Path]:
    yield from iter_files(path, (".jsonl",))


def read_captures(path: Path) -> list[RawTrace]:
    return [read_capture_file(p) for p in iter_capture_files(path) if is_capture_file(p)]
