"""Small helpers shared by rule checks and signals."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..messages import payload_text
from ..schema import Run, Step

_WORD = re.compile(r"[a-z0-9][a-z0-9.$%-]*[a-z0-9%]|[a-z0-9]", re.IGNORECASE)
_STOPWORDS = frozenset(
    (
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "your",
        "with",
        "this",
        "that",
        "from",
        "have",
        "has",
        "had",
        "was",
        "were",
        "will",
        "would",
        "can",
        "could",
        "should",
        "about",
        "into",
        "over",
        "than",
        "then",
        "them",
        "they",
        "their",
        "there",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "all",
        "any",
        "some",
        "its",
        "our",
        "out",
        "per",
        "via",
        "also",
        "just",
        "only",
        "very",
        "more",
        "most",
        "such",
        "each",
        "other",
        "been",
        "being",
        "does",
        "did",
        "doing",
        "here",
    )
)
_IGNORED_KEYS = {"id", "tool_call_id"}


HANDOFF_TOOL_PREFIXES = ("transfer_to_", "transfer_back_to_")


def is_handoff_tool(name: str | None) -> bool:
    """Tools that pass control to another agent (LangGraph supervisor/swarm handoffs). Their
    output is a transfer confirmation, not data for the agent to use."""
    return bool(name) and str(name).startswith(HANDOFF_TOOL_PREFIXES)


def content_tokens(text: str) -> set[str]:
    """Distinctive lowercase tokens: words of 3+ letters minus stopwords, and anything numeric."""
    tokens = set()
    for raw in _WORD.findall(text.lower()):
        word = raw.strip(".-")
        if any(ch.isdigit() for ch in word) or (len(word) >= 3 and word not in _STOPWORDS):
            tokens.add(word)
    return tokens


def coverage(source: set[str], target: set[str]) -> float:
    """Share of `source` tokens that appear in `target`."""
    return len(source & target) / len(source) if source else 1.0


def canonical(value: Any) -> str:
    """Comparable form of a payload: sorted keys, call IDs dropped, strings trimmed/lowercased."""

    def clean(v: Any) -> Any:
        if isinstance(v, Mapping):
            return {k: clean(x) for k, x in v.items() if k not in _IGNORED_KEYS}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, str):
            return " ".join(v.lower().split())
        return v

    return json.dumps(clean(value), sort_keys=True, default=str)


def excerpt(value: Any, limit: int = 160) -> str:
    text = " ".join(payload_text(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def llm_output_text(step: Step) -> str:
    return payload_text(step.output)


def preceding_llm(run: Run, step: Step) -> Step | None:
    """The model call that most recently finished before `step` started."""
    best = None
    for s in run.steps:
        if s.kind == "llm" and s.ended_at and step.started_at and s.ended_at <= step.started_at:
            best = s
    return best


def step_tokens(steps: Iterable[Step]) -> int:
    return sum((s.tokens_in or 0) + (s.tokens_out or 0) for s in steps)


def step_cost(steps: Iterable[Step]) -> float:
    return sum(s.cost or 0.0 for s in steps)


def step_latency(steps: Iterable[Step]) -> float:
    return sum(s.duration_s or 0.0 for s in steps)


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already sorted list."""
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


# ---- does a suggestion fit the trace? -------------------------------------------------------

_QUOTED = re.compile(r'"([^"]{12,})"')
_FAMILY = re.compile(r"[a-z]+")


def model_family(model: str) -> str:
    """The model line a model belongs to, from its name: claude-haiku-4-5 -> claude."""
    found = _FAMILY.match(model.split("/")[-1].lower())
    return found.group(0) if found else model


def cheaper_model_exists(model: str | None, prices: dict[str, Any]) -> bool:
    """True only if the price table lists a model of the same family that costs less. An
    unpriced model gets no "use a cheaper model" advice: nothing shows a cheaper one exists."""
    from ..llm.cost import price_of

    if not model:
        return False
    own = price_of(prices, model, 1_000_000, 0)
    if own is None:
        return False
    family = model_family(model)
    return any(model_family(name) == family and price.input < own for name, price in prices.items())


def agent_instructions(run: Run, agent: str | None) -> str:
    """The system prompts an agent ran with in this run, as text."""
    texts = []
    for step in run.steps_of("llm"):
        if step.node != agent or not isinstance(step.input, list):
            continue
        for message in step.input:
            if isinstance(message, dict) and message.get("type") == "system":
                texts.append(payload_text(message))
    return " ".join(dict.fromkeys(texts))  # each distinct prompt once, in order


def _plain(text: str) -> str:
    text = text.lower().replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text).split())


def instruction_in_place(fix: str, instructions: str) -> bool:
    """Whether a quoted prompt instruction in `fix` is already in `instructions`."""
    plain = _plain(instructions)
    return bool(plain) and any(
        _plain(quote) and _plain(quote) in plain for quote in _QUOTED.findall(fix)
    )
