"""verification.contradicted_result: the final answer states facts no tool returned.

A serious problem: the agent made up (or misread) flight numbers, booking references, routes,
dates, times or numbers, and the user can't tell. Only facts that can be checked against the
trace count: identifiers and codes, route codes (in a route: "BOS to LAX"), dates, times, and
numbers. A fact is made up when no tool result, the user's request, or the agents' prompts
contain it. A number is not counted when it can be computed from the numbers the tools
returned (a sum, difference, product or ratio of two of them, or a few of one) or is one of
them rounded, and a bare year never is.

Fires only on runs that delivered an answer and called at least one tool that returned data.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import combinations
from typing import Any

from ...messages import payload_text
from ...schema import Finding, Run
from ..base import Check, register
from ..util import excerpt, is_handoff_tool

_MONTHS = {
    m: i + 1
    for i, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        )
    )
    for m in names
}
_MONTH_NAMES = [n.capitalize() for n, i in _MONTHS.items() if len(n) > 3 or n == "may"]
_MONTH_NAMES = list(dict.fromkeys(_MONTH_NAMES))  # January ... December, in order
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_TEXT = re.compile(
    rf"\b(?:(?P<m1>{_MONTH})\.?\s+(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"|(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m2>{_MONTH})\b)",
    re.IGNORECASE,
)
_DATE_ISO = re.compile(r"\b(?:19|20)\d\d-(\d\d)-(\d\d)\b")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?:\s*([ap])\.?m\.?)?", re.IGNORECASE)
_FLIGHT = re.compile(r"\b([A-Z]{2}|[A-Z]\d|\d[A-Z])\s?(\d{2,4})\b")
_CODE = re.compile(r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{5,12}\b")
_PREFIXED_ID = re.compile(r"\b[A-Z]{2,}-\d[A-Z0-9-]*\b")
_ROUTE = re.compile(r"\b([A-Z]{3})\s*(?:-|\u2013|\u2014|\u2192|->|to)\s*([A-Z]{3})\b")
_NUMBER = re.compile(r"(?<![\w.:/-])\$?\d[\d,]*(?:\.\d+)?%?(?![\w:/-])")
_YEAR = re.compile(r"^(?:19|20)\d\d$")


def _iso_dates(text: str) -> set[str]:
    return {f"{int(m):02d}-{int(d):02d}" for m, d in _DATE_ISO.findall(text)}


def _dates(text: str) -> set[str]:
    """Month and day, "06-03" for "June 3", "3 June" and "2026-06-03" alike."""
    found = _iso_dates(text)
    for m in _DATE_TEXT.finditer(text):
        month = _MONTHS[(m["m1"] or m["m2"]).lower()]
        day = int(m["d1"] or m["d2"])
        if 1 <= day <= 31:
            found.add(f"{month:02d}-{day:02d}")
    return found


def _times(text: str) -> set[str]:
    found = set()
    for hour, minute, half in _TIME.findall(text):
        h = int(hour) % 12 + (12 if half.lower() == "p" else 0) if half else int(hour)
        if h < 24 and int(minute) < 60:
            found.add(f"{h:02d}:{minute}")
    return found


def _flights(text: str) -> set[str]:
    """Flight numbers ("UA 1532" and "UA1532" alike); never a quarter and a year ("Q3 2024")."""
    return {
        f"{a}{n}"
        for a, n in _FLIGHT.findall(text)
        if not _YEAR.match(n) and not re.fullmatch(r"Q[1-4]", a)
    }


def _codes(text: str) -> set[str]:
    """Identifiers: booking references, order and invoice numbers ("K7Q2LM", "INV-2041")."""
    flights = {m.group(0).replace(" ", "") for m in _FLIGHT.finditer(text)}
    codes = set(_CODE.findall(text)) | set(_PREFIXED_ID.findall(text))
    return {c for c in codes if c not in flights}


def _routes(text: str) -> set[str]:
    """Codes the answer gives as a route ("BOS to LAX"): never a lone acronym like "USA"."""
    return {code for pair in _ROUTE.findall(text) for code in pair}


def _all_codes(text: str) -> set[str]:
    """Every three-letter code in tool results and the request ({"from": "BOS"} too)."""
    return set(re.findall(r"\b[A-Z]{3}\b", text))


def _numbers(text: str) -> set[float]:
    """Amounts and counts: three or more digits, or a decimal, percentage or dollar amount.
    Years, and the digits of dates, times and identifiers, are left out."""
    for pattern in (_DATE_ISO, _TIME, _FLIGHT, _CODE, _PREFIXED_ID, _DATE_TEXT):
        text = pattern.sub(" ", text)
    found = set()
    for raw in _NUMBER.findall(text):
        value = raw.strip("$%").replace(",", "")
        digits = sum(ch.isdigit() for ch in value)
        marked = "%" in raw or "$" in raw or "." in value
        if _YEAR.match(value) and not marked:
            continue
        if digits >= 3 or (digits >= 2 and marked):
            found.add(float(value))
    return found


def _rounded(n: float, k: float) -> bool:
    """`n` is `k` rounded (to the cent, or to one, two or three significant figures)."""
    if abs(n - k) <= max(0.005 * abs(k), 0.5):
        return True
    digits = len(str(int(abs(k)))) if k else 1
    return any(round(k, sig - digits) == n for sig in (1, 2, 3) if sig < digits)


def _derived(n: float, known: Iterable[float]) -> bool:
    """`n` is one of `known` rounded, a few of one ("2 tickets at $214" is $428), or a sum,
    difference, product or ratio of two of them."""
    known = sorted(set(known))
    if any(_rounded(n, k) for k in known):
        return True
    if any(k and abs(n - k * m) <= 0.01 for k in known for m in range(2, 13)):
        return True
    for a, b in combinations(known, 2):
        results = [a + b, abs(a - b), a * b]
        results += [a / b, b / a, 100 * a / b, 100 * b / a] if a and b else []
        if any(abs(n - r) <= max(0.005 * abs(r), 0.01) for r in results):
            return True
    return False


def _data_tools(run: Run) -> list:
    return [s for s in run.steps_of("tool") if not s.error and not is_handoff_tool(s.name)]


def _known_text(run: Run) -> str:
    """Everything the answer may rightly repeat: tool results, the request, the prompts."""
    parts = [payload_text(s.output) for s in _data_tools(run)]
    parts.append(run.input_preview)
    for step in run.steps_of("llm"):
        if isinstance(step.input, list):
            parts += [
                payload_text(m)
                for m in step.input
                if isinstance(m, dict) and m.get("type") in ("system", "human")
            ]
    return "\n".join(parts)


def _fmt(kind: str, value: Any) -> str:
    if kind == "dates":
        month, day = value.split("-")
        return f"{_MONTH_NAMES[int(month) - 1]} {int(day)}"
    if kind == "flights":
        return re.sub(r"^([A-Z0-9]{2})(\d)", r"\1 \2", value)
    if kind == "numbers":
        return f"{value:,.2f}".rstrip("0").rstrip(".") if value % 1 else f"{int(value):,}"
    return str(value)


_KINDS = {
    "flights": _flights,
    "codes": _codes,
    "routes": _routes,
    "dates": _dates,
    "times": _times,
}


def made_up_facts(run: Run) -> tuple[list[str], list[str]]:
    """(facts in the final answer that no tool returned, what the tools returned of the same
    kinds), formatted for the report. Both empty when the check doesn't apply."""
    tools = _data_tools(run)
    answer = run.final_answer
    if run.status != "completed" or not answer.strip() or not tools:
        return [], []
    known_text = _known_text(run)
    tool_text = "\n".join(payload_text(s.output) for s in tools)
    made_up: list[str] = []
    returned: list[str] = []
    for kind, extract in _KINDS.items():
        known = _all_codes(known_text) if kind == "routes" else extract(known_text)
        missing = sorted(extract(answer) - known)
        made_up += [_fmt(kind, v) for v in missing]
        if missing:
            returned += [_fmt(kind, v) for v in sorted(extract(tool_text))]
    numbers = _numbers(known_text)
    missing_numbers = sorted(n for n in _numbers(answer) if not _derived(n, numbers))
    made_up += [_fmt("numbers", n) for n in missing_numbers]
    if missing_numbers:
        returned += [_fmt("numbers", n) for n in sorted(_numbers(tool_text))]
    return made_up, list(dict.fromkeys(returned))


@register
class ContradictedResult(Check):
    id = "verification.contradicted_result"
    group = "verification"
    method = "rule"
    default_severity = "high"

    def run(self, runs: list[Run]) -> list[Finding]:
        findings = []
        for run in runs:
            made_up, returned = made_up_facts(run)
            if not made_up:
                continue
            llm_steps = run.steps_of("llm")
            last = llm_steps[-1] if llm_steps else run.steps_of("node")[-1]
            facts = {
                "agent": last.node or "the agent",
                "facts": ", ".join(made_up[:6]),
                "returned": ", ".join(returned[:6]),
            }
            findings.append(
                self.finding(
                    run,
                    key=last.node or last.name,
                    node=last.node,
                    role=last.role,
                    evidence_step_ids=[last.id, *(s.id for s in _data_tools(run))][:4],
                    evidence_excerpts=[
                        f"Final answer: {excerpt(run.final_answer, 160)}",
                        f"Not in any tool result: {facts['facts']}",
                        *([f"The tools returned: {facts['returned']}"] if returned else []),
                    ],
                    facts=facts,
                    variant="contradicts" if returned else "not_returned",
                )
            )
        return findings
