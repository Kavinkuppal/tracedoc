"""LangSmith import via the SDK, with a fake client (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from tests.helpers import TRACES
from tracedoc.ingest.langsmith import fetch_project, read_export
from tracedoc.normalize import normalize


class FakeClient:
    def __init__(self, runs: list[dict[str, Any]]) -> None:
        self.runs = runs
        self.calls: list[dict[str, Any]] = []

    def list_runs(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        if kwargs.get("is_root"):
            roots = [r for r in self.runs if r["parent_run_id"] is None]
            return [SimpleNamespace(**r) for r in roots][: kwargs.get("limit")]
        return [dict(r) for r in self.runs if r["trace_id"] == kwargs["trace_id"]]


def _fixture_runs(*names: str) -> list[dict[str, Any]]:
    runs = []
    for name in names:
        runs.extend(json.loads((TRACES / "langsmith" / f"{name}.json").read_text()))
    return runs


def test_fetch_project_pulls_roots_then_children() -> None:
    client = FakeClient(_fixture_runs("healthy", "tool_output_unused"))
    traces = fetch_project("my-project", limit=10, client=client)
    assert len(traces) == 2
    assert client.calls[0] == {"project_name": "my-project", "is_root": True, "limit": 10}
    assert all(c["project_name"] == "my-project" for c in client.calls)
    statuses = sorted(normalize(t).status for t in traces)
    assert statuses == ["completed", "completed"]


def test_export_formats_json_list_object_and_jsonl(tmp_path) -> None:
    runs = _fixture_runs("healthy")
    as_list = tmp_path / "a.json"
    as_list.write_text(json.dumps(runs))
    as_obj = tmp_path / "b.json"
    as_obj.write_text(json.dumps({"runs": runs}))
    as_lines = tmp_path / "c.jsonl"
    as_lines.write_text("\n".join(json.dumps(r) for r in runs))
    counts = {len(normalize(read_export(p)[0]).steps) for p in (as_list, as_obj, as_lines)}
    assert counts == {12}


def test_cost_from_langsmith_is_kept() -> None:
    runs = _fixture_runs("tool_output_unused")
    for r in runs:
        if r["run_type"] == "llm":
            r["total_cost"] = "0.0015"
    (trace,) = read_export_from(runs)
    run = normalize(trace)
    assert run.totals.cost == 0.003


def read_export_from(runs: list[dict[str, Any]]):
    from tracedoc.ingest.langsmith import traces_from_runs

    return traces_from_runs(runs)
