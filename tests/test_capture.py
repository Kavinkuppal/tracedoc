"""TraceCapture writes one well-formed JSONL file per run."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import pytest
from langgraph.errors import GraphRecursionError

from tests.fixtures.agents import AGENTS, run_agent
from tracedoc import TraceCapture
from tracedoc.capture.callback import FORMAT, FORMAT_VERSION
from tracedoc.ingest.jsonl import CaptureFormatError, read_capture_file
from tracedoc.normalize import normalize


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _capture(tmp_path: Path, name: str) -> Path:
    capture = TraceCapture(tmp_path)
    with contextlib.suppress(GraphRecursionError):
        run_agent(name, callbacks=[capture])
    (path,) = capture.files_written
    return path


def test_header_and_one_file_per_run(tmp_path: Path) -> None:
    capture = TraceCapture(tmp_path)
    run_agent("healthy", callbacks=[capture])
    run_agent("tool_output_unused", callbacks=[capture])
    assert len(capture.files_written) == 2
    assert sorted(tmp_path.glob("*.jsonl")) == sorted(capture.files_written)
    header = _events(capture.files_written[0])[0]
    assert header["event"] == "trace_start"
    assert header["format"] == FORMAT
    assert header["format_version"] == FORMAT_VERSION
    assert header["framework"] == "langgraph"
    assert header["framework_version"]


def test_records_langgraph_node_and_step_metadata(tmp_path: Path) -> None:
    events = _events(_capture(tmp_path, "healthy"))
    node_starts = [
        e
        for e in events
        if e["event"] == "chain_start" and e.get("metadata", {}).get("langgraph_node") == e["name"]
    ]
    assert [e["name"] for e in node_starts] == [
        "researcher",
        "tools",
        "researcher",
        "writer",
        "critic",
    ]
    assert [e["metadata"]["langgraph_step"] for e in node_starts] == [1, 2, 3, 4, 5]


def test_records_token_usage_and_model(tmp_path: Path) -> None:
    events = _events(_capture(tmp_path, "healthy"))
    ends = [e for e in events if e["event"] == "llm_end"]
    assert len(ends) == 4
    for e in ends:
        assert e["data"]["usage"]["input_tokens"] > 0
        assert e["data"]["model"] == "fake-scripted-1"


def test_every_event_has_ids_and_parent_links(tmp_path: Path) -> None:
    events = _events(_capture(tmp_path, "researcher_repeat_search"))[1:]
    started = {e["run_id"] for e in events if e["event"].endswith("_start")}
    roots = [e for e in events if e["event"].endswith("_start") and e["parent_run_id"] is None]
    assert len(roots) == 1
    for e in events:
        assert e["run_id"] in started
        if e["parent_run_id"] is not None:
            assert e["parent_run_id"] in started
    ended = {e["run_id"] for e in events if e["event"].endswith(("_end", "_error"))}
    assert ended == started


def test_records_tool_errors(tmp_path: Path) -> None:
    events = _events(_capture(tmp_path, "tool_retry_same_args"))
    errors = [e for e in events if e["event"] == "tool_error"]
    assert len(errors) == 3
    assert all(e["data"]["error_type"] == "ConnectionError" for e in errors)


def test_records_recursion_limit_on_root(tmp_path: Path) -> None:
    events = _events(_capture(tmp_path, "supervisor_no_end"))
    last = events[-1]
    assert last["event"] == "chain_error"
    assert last["parent_run_id"] is None
    assert last["data"]["error_type"] == "GraphRecursionError"


def test_works_with_async_invoke(tmp_path: Path) -> None:
    module = AGENTS["healthy"]
    capture = TraceCapture(tmp_path)
    asyncio.run(module.build().ainvoke(module.INPUT, config={"callbacks": [capture]}))
    (path,) = capture.files_written
    run = normalize(read_capture_file(path))
    assert run.status == "completed"
    assert run.totals.step_count == 5


def test_truncates_long_strings(tmp_path: Path) -> None:
    capture = TraceCapture(tmp_path, max_chars=40)
    run_agent("healthy", callbacks=[capture])
    for e in _events(capture.files_written[0])[1:]:
        for msg in e["data"].get("messages") or []:
            assert len(msg["content"]) <= 40 + len("... [truncated 9999 chars]")


def test_crashed_run_with_partial_last_line_is_readable(tmp_path: Path) -> None:
    path = _capture(tmp_path, "healthy")
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[: len(lines) // 2]) + "\n" + lines[len(lines) // 2][:20])
    run = normalize(read_capture_file(path))
    assert run.status == "error"
    assert run.steps


def test_rejects_files_without_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"event": "chain_start"}\n')
    with pytest.raises(CaptureFormatError):
        read_capture_file(path)
