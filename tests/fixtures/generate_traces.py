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
REPO = Path(__file__).resolve().parents[2]


def scrub_local_paths(text: str) -> str:
    """Tracebacks embed absolute paths (and so the developer's username); keep them out of git."""
    return text.replace(str(REPO), "<repo>").replace(str(Path.home()), "<home>")


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
                run.model_dump_json(
                    exclude={"child_runs", "parent_run", "ls_client"},
                    fallback=repr,  # like LangSmith's own serializer, for e.g. langgraph Send
                )
            )
            usage = (data.get("extra") or {}).get("metadata", {}).get("usage_metadata")
            if usage:  # the LangSmith server derives these columns from usage metadata
                data["prompt_tokens"] = usage.get("input_tokens")
                data["completion_tokens"] = usage.get("output_tokens")
            exported.append(data)
        return exported


def generate(names: list[str] | None = None) -> None:
    """Regenerate traces for `names` (default: all agents, replacing every trace file)."""
    capture_dir = TRACES / "capture"
    langsmith_dir = TRACES / "langsmith"
    for d in (capture_dir, langsmith_dir):
        if names is None:
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    for name in names or list(AGENTS):
        with tempfile.TemporaryDirectory() as tmp:
            capture = TraceCapture(tmp)
            tracer = CollectingTracer()
            with contextlib.suppress(GraphRecursionError):
                run_agent(name, callbacks=[capture, tracer])
            (written,) = capture.files_written
            (capture_dir / f"{name}.jsonl").write_text(scrub_local_paths(written.read_text()))
        runs = tracer.export()
        (langsmith_dir / f"{name}.json").write_text(
            scrub_local_paths(json.dumps(runs, indent=1) + "\n")
        )
        print(f"{name}: {len(runs)} LangSmith runs")


if __name__ == "__main__":
    import sys

    generate(sys.argv[1:] or None)
