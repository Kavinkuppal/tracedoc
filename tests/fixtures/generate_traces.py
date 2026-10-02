"""Regenerate the committed fixture traces from the fixture agents.

    python -m tests.fixtures.generate_traces

Writes, per agent:
  traces/capture/<agent>.jsonl   via TraceCapture
  traces/langsmith/<agent>.json  LangSmith run payloads from LangChain's own tracer
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.schemas import Run as TracerRun
from langgraph.errors import GraphRecursionError

from tracedoc import TraceCapture

from .agents import AGENTS, run_agent

TRACES = Path(__file__).parent / "traces"


class CollectingTracer(LangChainTracer):
    """LangChain's LangSmith tracer, but it keeps runs in memory instead of posting them."""

    def __init__(self) -> None:
        super().__init__(project_name="tracedoc-fixtures", client=object())  # type: ignore[arg-type]
        self.collected: list[TracerRun] = []

    def copy_with_metadata_defaults(self, *args: Any, **kwargs: Any) -> CollectingTracer:
        return self  # no tracer-level metadata to merge; keep one shared collection

    def _persist_run_single(self, run: TracerRun) -> None:
        run.tags = self._get_tags(run)
        self.collected.append(run)

    @staticmethod
    def _update_run_single(run: TracerRun) -> None:
        return None

    def export(self) -> list[dict[str, Any]]:
        exported = []
        for run in self.collected:
            data = json.loads(
                run.model_dump_json(exclude={"child_runs", "parent_run", "ls_client"})
            )
            usage = (data.get("extra") or {}).get("metadata", {}).get("usage_metadata")
            if usage:  # the LangSmith server derives these columns from usage metadata
                data["prompt_tokens"] = usage.get("input_tokens")
                data["completion_tokens"] = usage.get("output_tokens")
            exported.append(data)
        return exported


def generate() -> None:
    capture_dir = TRACES / "capture"
    langsmith_dir = TRACES / "langsmith"
    for d in (capture_dir, langsmith_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for name in AGENTS:
        with tempfile.TemporaryDirectory() as tmp:
            capture = TraceCapture(tmp)
            tracer = CollectingTracer()
            with contextlib.suppress(GraphRecursionError):
                run_agent(name, callbacks=[capture, tracer])
            (written,) = capture.files_written
            shutil.copy(written, capture_dir / f"{name}.jsonl")
        runs = tracer.export()
        (langsmith_dir / f"{name}.json").write_text(json.dumps(runs, indent=1) + "\n")
        print(f"{name}: {len(runs)} LangSmith runs")


if __name__ == "__main__":
    generate()
