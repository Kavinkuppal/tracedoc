"""The only text a model writes into a report: three short slots per AI judge finding.

What happened, what to change, and one evidence quote, each with a character limit from the
catalog. An answer that breaks a limit, or quotes something that is not in the trace window
word for word, is sent back once with the problem named. Whatever is still too long after
that is cut at a word boundary, and a quote that still isn't in the window is dropped (the
finding then quotes the cited step instead).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

from ...catalog import limit
from ..util import instruction_in_place

SLOTS = ("what_happened", "what_to_change", "evidence_quote")
_QUOTE_MARKS = " \"'\u201c\u201d\u2018\u2019"  # straight and curly quotation marks
_TRAILING = " ,;:.-\u2013\u2014"  # punctuation not worth keeping before an ellipsis


class HasSlots(Protocol):
    what_happened: str
    what_to_change: str
    evidence_quote: str


def squash(text: str) -> str:
    """Collapse runs of whitespace (newlines included) to single spaces."""
    return " ".join(text.split())


def truncate_words(text: str, max_chars: int) -> str:
    """At most `max_chars` characters, cut at a word boundary and marked with an ellipsis."""
    text = squash(text)
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    space = cut.rfind(" ")
    if space >= max_chars // 2:
        cut = cut[:space]
    return cut.rstrip(_TRAILING) + "…"


_SENTENCE_BREAK = re.compile(r"[.!?][\"'\u201d\u2019)]*\s+(?=[A-Z0-9\"'\u201c(\[])")
"""The end of a sentence: an end mark (perhaps inside closing quotes), a space, and a capital,
a digit or an opening quote. "1.5 s" and "e.g. this" are not sentence ends."""


def sentences(text: str) -> list[str]:
    text = squash(text)
    parts, start = [], 0
    for end in _SENTENCE_BREAK.finditer(text):
        parts.append(text[start : end.end()].strip())
        start = end.end()
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def fit_sentences(text: str, max_chars: int, frame: Callable[[str], str] = str) -> str:
    """`frame(text)` in at most `max_chars` characters, keeping as many whole sentences of
    `text` as fit, so it ends at a sentence boundary. Only when the first sentence alone is too
    long is it cut at a word boundary, with an ellipsis."""
    parts = sentences(text)
    for k in range(len(parts), 0, -1):
        candidate = frame(" ".join(parts[:k]))
        if len(candidate) <= max_chars:
            return candidate
    return truncate_words(frame(parts[0] if parts else ""), max_chars)


def quote_in_window(quote: str, window_text: str) -> bool:
    core = squash(quote).strip(_QUOTE_MARKS)
    return bool(core) and core in squash(window_text)


def window_instructions(window_text: str) -> str:
    """The agents' system prompts as the window shows them (see windows.py)."""
    head = window_text.split("\nSTEPS:", 1)[0]
    return (
        head.split("AGENT INSTRUCTIONS (system prompts):", 1)[1]
        if "AGENT INSTRUCTIONS" in head
        else ""
    )


def slot_problems(answer: HasSlots, window_text: str) -> list[str]:
    """Each broken rule, in words the model can act on. Empty if the slots are fine."""
    problems = []
    for name in SLOTS:
        value, max_chars = squash(getattr(answer, name)), limit(name)
        if not value:
            problems.append(f"{name} is empty")
        elif len(value) > max_chars:
            problems.append(f"{name} is {len(value)} characters; the limit is {max_chars}")
    if instruction_in_place(answer.what_to_change, window_instructions(window_text)):
        problems.append(
            "what_to_change suggests an instruction the agent's prompt already has; suggest a "
            "change that the steps show is not in place yet"
        )
    quote = squash(answer.evidence_quote)
    if quote and not quote_in_window(quote, window_text):
        problems.append(
            "evidence_quote is not copied exactly from the steps shown; copy it character "
            "for character"
        )
    return problems


def fitted(answer: HasSlots, window_text: str) -> dict[str, str]:
    """Slot values that respect every limit: cut at a word boundary where too long, and the
    quote dropped if it is not in the window."""
    values = {name: truncate_words(getattr(answer, name), limit(name)) for name in SLOTS}
    quote = squash(answer.evidence_quote).strip(_QUOTE_MARKS)
    if not quote_in_window(quote, window_text):
        values["evidence_quote"] = ""
    elif len(quote) > limit("evidence_quote"):
        values["evidence_quote"] = truncate_words(quote, limit("evidence_quote"))
    else:
        values["evidence_quote"] = quote
    return values
