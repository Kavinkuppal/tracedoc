"""The only text a model writes into a report: three short slots per AI judge finding.

What happened, what to change, and one evidence quote, each with a character limit from the
catalog. An answer that breaks a limit, or quotes something that is not in the trace window
word for word, is sent back once with the problem named. Whatever is still too long after
that is cut at a word boundary, and a quote that still isn't in the window is dropped (the
finding then quotes the cited step instead).
"""

from __future__ import annotations

from typing import Protocol

from ...catalog import limit

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


def quote_in_window(quote: str, window_text: str) -> bool:
    core = squash(quote).strip(_QUOTE_MARKS)
    return bool(core) and core in squash(window_text)


def slot_problems(answer: HasSlots, window_text: str) -> list[str]:
    """Each broken rule, in words the model can act on. Empty if the slots are fine."""
    problems = []
    for name in SLOTS:
        value, max_chars = squash(getattr(answer, name)), limit(name)
        if not value:
            problems.append(f"{name} is empty")
        elif len(value) > max_chars:
            problems.append(f"{name} is {len(value)} characters; the limit is {max_chars}")
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
