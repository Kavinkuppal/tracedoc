"""Classify a run's task into a fixed list, from its input preview.

Weighted keyword rules: each task type has patterns with weights (specific cues weigh more than
generic question words); the highest total wins, ties go to the earlier type in PRIORITY, and no
match at all means "other" (or the optional model fallback). Definitions:

- research: find or explain facts ("who founded...", "compare...", "how many employees...").
- coding: write, fix, test or explain code.
- data_extraction: pull structured fields out of given documents or text.
- customer_support: the user's own orders, accounts, bills and bookings, including booking,
  changing or cancelling travel or reservations through an assistant.
- planning: plans, itineraries, schedules, roadmaps, step-by-step plans.
- writing: produce or edit prose (posts, emails, summaries, translations, taglines).
- other: everything else, including pure arithmetic and chit-chat.

Labeled examples: tests/fixtures/task_types.jsonl.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ..schema import TASK_TYPES, TaskType

TaskClassifier = Callable[[str], "TaskType | None"]
"""Optional model fallback: given a truncated input preview, return a task type or None."""

PRIORITY: tuple[TaskType, ...] = (
    "coding",
    "data_extraction",
    "customer_support",
    "planning",
    "writing",
    "research",
    "other",
)

_RULES: dict[TaskType, list[tuple[str, int]]] = {
    "coding": [
        (r"\b(python|javascript|typescript|java|golang|rust|c\+\+|react|sql|regex|html|css)\b", 3),
        (r"\b(go|my) program\b", 3),
        (
            r"\b(function|refactor\w*|bug|stack ?trace|unit tests?|compile\w*|panics?|exception)\b",
            2,
        ),
        (r"\b(implement\w*|debug\w*|component|module|async|await|nil pointer|endpoint)\b", 2),
        (r"\b(code|script|query|loop)\b", 1),
    ],
    "data_extraction": [
        (r"\b(extract\w*|scrape\w*|parse\w*)\b", 3),
        (r"\bpull\b.*\b(out of|from)\b", 3),
        (r"\b(into|as) (json|csv|a table|structured)\b", 3),
        (r"\b(fields?|structured)\b", 2),
        (r"\b(invoices?|receipts?|resumes?|contract|pdf)\b", 1),
        (r"\bget the\b.*\bfrom (each|these|this)\b", 3),
    ],
    "customer_support": [
        (r"\bmy (order|package|account|subscription|booking|reservation|password|bill|card)\b", 3),
        (r"\b(refund|charged|cancel my|log ?in|password|order number)\b", 3),
        (r"\b(book|reserve)( me)? (a|an|the) (flight|hotel|room|table|seat|ticket)\b", 3),
        (r"\b(flight|flights) from\b.*\bto\b", 3),
        (r"\b(find|get) me a (flight|hotel|room)\b", 3),
        (r"\bhotel room\b|\bi need a (hotel|flight|room)\b", 3),
        (r"\b(reservation|booking)\b", 2),
        (r"\bchange my\b", 2),
    ],
    "planning": [
        (r"\b(itinerary|roadmap|milestones?|meal prep)\b", 3),
        (r"\b(plan|planning)\b", 2),
        (r"\bschedule\b", 2),
        (r"\bsteps to\b", 3),
        (r"\bbreak\b.*\binto\b", 2),
    ],
    "writing": [
        (r"\b(blog|essay|tagline|slogan|cover letter|announcement|poem|story)\b", 3),
        (r"\b(proofread|rewrite|translate|summari[sz]e|summary)\b", 3),
        (r"\b(write|draft)\b", 1),
        (r"\b(email|paragraph|brief|post|bullet points?)\b", 2),
    ],
    "research": [
        (r"\b(research|studies|sources?|look up|compare|founded|history|population)\b", 2),
        (r"\b(headcount|employees|market cap|revenue|ceo)\b", 3),
        (r"\b(who|which|why|when)\b", 1),
        (r"\bwhat (is|are|was|were|'s)\b|\bwhat's\b", 1),
        (r"\bhow (many|much|does|did|do)\b", 1),
        (r"\bfind\b", 1),
        (r"^\s*(did|does|do|is|are|was|were|has|have|can)\b.*\?\s*$", 1),
    ],
    "other": [
        (r"\b(multiplied by|divided by|times|plus|minus|square root|power of)\b", 3),
        (r"\d+(\.\d+)?\s*%\s*of\b", 3),
        (r"\b(compute|calculate|add|subtract)\s+[\d$]", 3),
        (r"\bconvert\b.*\b(celsius|fahrenheit|miles|km|kg|pounds|inches|cm)\b", 3),
        (r"\b(joke|riddle)\b|^\s*(hi|hello|hey)\b", 3),
    ],
}
_COMPILED = {
    t: [(re.compile(p, re.IGNORECASE), w) for p, w in rules] for t, rules in _RULES.items()
}


def task_scores(text: str) -> dict[TaskType, int]:
    return {t: sum(w for p, w in rules if p.search(text)) for t, rules in _COMPILED.items()}


def task_from_text(text: str) -> TaskType | None:
    scores = task_scores(text)
    best = max(scores.values())
    if best == 0:
        return None
    return next(t for t in PRIORITY if scores[t] == best)


def classify_task(
    input_preview: str, *, fallback: TaskClassifier | None = None, max_chars: int = 500
) -> TaskType:
    task = task_from_text(input_preview)
    if task is None and fallback is not None and input_preview.strip():
        guess = fallback(input_preview[:max_chars])
        task = guess if guess in TASK_TYPES else None
    return task or "other"
