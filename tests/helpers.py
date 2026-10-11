"""Shared helpers for loading fixture traces."""

from __future__ import annotations

from pathlib import Path

from tracedoc.config import Config
from tracedoc.ingest.base import RawTrace
from tracedoc.ingest.jsonl import read_capture_file
from tracedoc.ingest.langsmith import read_export
from tracedoc.normalize import normalize
from tracedoc.schema import Run

TRACES = Path(__file__).parent / "fixtures" / "traces"
SOURCES = ("capture", "langsmith")


def raw_fixture(name: str, source: str) -> RawTrace:
    if source == "capture":
        return read_capture_file(TRACES / "capture" / f"{name}.jsonl")
    (trace,) = read_export(TRACES / "langsmith" / f"{name}.json")
    return trace


def load_fixture(name: str, source: str = "capture", config: Config | None = None) -> Run:
    return normalize(raw_fixture(name, source), config)


def reports_in(out: Path) -> list[dict]:
    """The report.json files an analysis wrote: the one in `out`, or, when the traces came from
    several systems, one per system sub-folder (the fixtures are seven systems)."""
    import json

    paths = (
        [out / "report.json"]
        if (out / "report.json").is_file()
        else sorted(out.glob("*/report.json"))
    )
    assert paths, f"no report.json under {out}"
    return [json.loads(p.read_text()) for p in paths]
