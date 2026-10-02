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
