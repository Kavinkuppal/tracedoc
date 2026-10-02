"""Classify a run's task into a fixed list, from its input preview."""

from __future__ import annotations

import re
from collections.abc import Callable

from ..schema import TASK_TYPES, TaskType

TaskClassifier = Callable[[str], "TaskType | None"]
"""Optional model fallback: given a truncated input preview, return a task type or None."""

# Higher-specificity types first so ties break toward them.
_PATTERNS: list[tuple[TaskType, tuple[str, ...]]] = [
    (
        "coding",
        (
            r"\bcode\b",
            r"\bfunction\b",
            r"\bbug\b",
            r"\bpython\b",
            r"\bjavascript\b",
            r"\btypescript\b",
            r"\brefactor",
            r"\bstack ?trace\b",
            r"\bcompile",
            r"\bunit tests?\b",
            r"\bimplement",
            r"\bdebug",
            r"\bsql\b",
            r"\bregex\b",
        ),
    ),
    (
        "customer_support",
        (
            r"\bmy order\b",
            r"\border #?\w*\d",
            r"\brefund",
            r"\bmy account\b",
            r"\bpassword\b",
            r"\bshipping\b",
            r"\bsubscription\b",
            r"\bdelivery\b",
            r"\bcancel my\b",
            r"\bcharged\b",
            r"\bwhere is my\b",
        ),
    ),
    (
        "data_extraction",
        (
            r"\bextract",
            r"\bparse\b",
            r"\bfields?\b",
            r"\binvoices?\b",
            r"\bscrape",
            r"\bjson\b",
            r"\bcsv\b",
            r"\bstructured data\b",
        ),
    ),
    (
        "planning",
        (
            r"\bplan\b",
            r"\bitinerary\b",
            r"\bschedule\b",
            r"\broadmap\b",
            r"\bsteps to\b",
            r"\bmilestones?\b",
        ),
    ),
    (
        "writing",
        (
            r"\bwrite\b",
            r"\bdraft\b",
            r"\bessay\b",
            r"\bblog\b",
            r"\bemail\b",
            r"\bbrief\b",
            r"\bsummar(y|ize|ise)\b",
            r"\brewrite\b",
            r"\bparagraph\b",
            r"\bpost\b",
        ),
    ),
    (
        "research",
        (
            r"\bresearch\b",
            r"\bwhat (is|are|was|were)\b",
            r"\bfind\b",
            r"\bcompare\b",
            r"\bsources?\b",
            r"\blook up\b",
            r"\bapproaches\b",
            r"\bhow does\b",
            r"\bwhy\b",
            r"\bdid\b.*\?",
        ),
    ),
]
_COMPILED = [(t, [re.compile(p, re.IGNORECASE) for p in ps]) for t, ps in _PATTERNS]


def task_from_text(text: str) -> TaskType | None:
    best: TaskType | None = None
    best_score = 0
    for task, patterns in _COMPILED:
        score = sum(1 for p in patterns if p.search(text))
        if score > best_score:
            best, best_score = task, score
    return best


def classify_task(
    input_preview: str, *, fallback: TaskClassifier | None = None, max_chars: int = 500
) -> TaskType:
    task = task_from_text(input_preview)
    if task is None and fallback is not None and input_preview.strip():
        guess = fallback(input_preview[:max_chars])
        task = guess if guess in TASK_TYPES else None
    return task or "other"
