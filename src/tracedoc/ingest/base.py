"""Source-neutral intermediate form: a trace is a flat list of spans.

Each reader (our JSONL capture, LangSmith, later OpenTelemetry) produces
`RawTrace`s. `normalize.py` turns those into the schema in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from ..schema import Source

SpanKind = Literal["chain", "llm", "tool"]


@dataclass
class RawSpan:
    id: str
    parent_id: str | None
    kind: SpanKind
    name: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    inputs: Any = None
    outputs: Any = None
    error: str | None = None
    error_type: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost: float | None = None
    ended: bool = False


@dataclass
class RawTrace:
    id: str
    source: Source
    spans: list[RawSpan]
    framework: str | None = None
    framework_version: str | None = None

    @property
    def root(self) -> RawSpan | None:
        ids = {s.id for s in self.spans}
        for span in self.spans:
            if span.parent_id is None or span.parent_id not in ids:
                return span
        return None


def parse_time(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)
