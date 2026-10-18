"""verification.contradicted_result: the final answer states facts no tool returned.

A serious problem when real: the agent made up (or misread) flight numbers, booking
references, routes, dates, times or numbers, and the user can't tell. Only facts that can be
checked against the trace count: identifiers and codes, route codes (in a route: "BOS to LAX"),
dates, times, and numbers. A fact is a candidate when no tool result, the user's request, or
the agents' prompts contain it, and quick filters don't explain it:

- a number computed from the numbers given (a sum or average of any of them, a difference,
  product or ratio of two, a few of one plus another, a percentage or percentage difference),
  or rounded the way it is written ("1.28 million" for 1,284,000), or the result of working
  the answer itself shows ("17.5 x 24 = 408 + 12 = 420") from numbers it was given;
- an airport code for a city that was named ("SEA" for Seattle; see airports.py);
- a term that only looks like an identifier ("COVID-19", "A320", "FY2024");
- a bare year.

What is left is a rule candidate: low confidence until the AI judge confirms it, so it never
reaches page 1 unconfirmed. Fires only on runs that delivered an answer and called at least one
tool that returned data.
"""

from __future__ import annotations

import ast
import operator
import re
from bisect import bisect_left
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from ...messages import payload_text
from ...schema import Finding, Run
from ..base import Check, register
from ..util import excerpt, is_handoff_tool
from .airports import airport_codes

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
_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip
"""How the report writes a month: _MONTH_NAMES[month - 1]."""
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
_YEAR = re.compile(r"^(?:19|20)\d\d$")
_TERMS = {
    "COVID19", "SARSCOV2", "MERSCOV", "H1N1", "H5N1", "MP3", "MP4", "CO2", "B2B", "B2C",
    "P2P", "Y2K", "IPV4", "IPV6", "UTF8", "UTF16", "SHA1", "SHA256", "SHA512", "MD5",
    "ISO8601", "ISO9001", "W2", "W4", "W9", "I9", "401K", "4K", "8K", "3G", "4G", "5G", "3D",
    "2FA", "GPT4", "GPT5", "F35", "A220", "A300", "A319", "A320", "A321", "A330", "A340",
    "A350", "A380", "B737", "B747", "B757", "B767", "B777", "B787", "E175", "E190", "E195",
}  # fmt: skip
"""Terms that only look like identifiers or flight numbers (written without hyphens)."""
_PERIOD = re.compile(r"^(?:FY|CY|Q[1-4]|H[12])\d{2,4}$")
"""Fiscal years and quarters, "FY2024", "Q3 2024"."""


def _term(code: str) -> bool:
    plain = code.replace("-", "").replace(" ", "").upper()
    return plain in _TERMS or bool(_PERIOD.match(plain))


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
    """Flight numbers ("UA 1532" and "UA1532" alike); never a quarter and a year ("Q3 2024"),
    or an aircraft ("A320")."""
    return {
        f"{a}{n}"
        for a, n in _FLIGHT.findall(text)
        if not _YEAR.match(n) and not re.fullmatch(r"Q[1-4]", a) and not _term(f"{a}{n}")
    }


def _codes(text: str) -> set[str]:
    """Identifiers: booking references, order and invoice numbers ("K7Q2LM", "INV-2041"), but
    not terms that only look like one ("COVID-19")."""
    flights = {m.group(0).replace(" ", "") for m in _FLIGHT.finditer(text)}
    codes = set(_CODE.findall(text)) | set(_PREFIXED_ID.findall(text))
    return {c for c in codes if c not in flights and not _term(c)}


def _routes(text: str) -> set[str]:
    """Codes the answer gives as a route ("BOS to LAX"): never a lone acronym like "USA"."""
    return {code for pair in _ROUTE.findall(text) for code in pair}


def _all_codes(text: str) -> set[str]:
    """Every three-letter code the text gives or names: {"from": "BOS"} gives BOS, and "a
    flight to Seattle" gives SEA."""
    return set(re.findall(r"\b[A-Z]{3}\b", text)) | airport_codes(text)


# ---- numbers --------------------------------------------------------------------------------

_SCALES = {
    "thousand": 1e3, "k": 1e3, "K": 1e3,
    "million": 1e6, "mn": 1e6, "M": 1e6,
    "billion": 1e9, "bn": 1e9, "B": 1e9,
    "trillion": 1e12, "tn": 1e12, "T": 1e12,
}  # fmt: skip
_NUMBER = re.compile(
    r"(?<![\w.:/-])(?P<dollar>\$)?(?P<digits>\d[\d,]*(?:\.\d+)?)"
    r"(?:(?P<pct>\s?%|\s+percent\b|\s+per\s+cent\b)"
    r"|\s+(?P<word>thousand|million|billion|trillion)\b"
    r"|(?P<abbr>k|K|mn|M|bn|B|tn|T)\b)?"
    r"(?![\w:/]|-\d)",
)
_RANGE = re.compile(r"(?<=\d)\s?[-\u2013]\s?(?=\$?\d)")
"""A range between two numbers, "220,000-230,000": two numbers, not one with a dash."""


@dataclass(frozen=True)
class _Number:
    value: float
    tolerance: float
    """Half a unit of the last digit as written: 0.5 for 537,210, 5,000 for 1.28 million."""
    percent: bool
    marked: bool
    """Written as a decimal, percentage, dollar or scaled amount."""
    digits: int
    text: str
    """As written, for the report: "$134.9 billion", "71.3%"."""

    @property
    def loose(self) -> bool:
        """Written to about two significant digits or fewer ("220,000", "about 2 million"), so
        close to many values by chance: only sums, averages and one-step results explain it."""
        return self.tolerance >= 0.005 * abs(self.value) and self.value != 0

    @property
    def is_fact(self) -> bool:
        """Amounts and counts worth checking: three or more digits, or two in a decimal,
        percentage, dollar or scaled amount. A bare year isn't one."""
        if not self.marked and _YEAR.match(self.text):
            return False
        return self.digits >= 3 or (self.digits >= 2 and self.marked)


def _parse(text: str) -> list[_Number]:
    found = []
    for m in _NUMBER.finditer(_RANGE.sub(" to ", text)):
        digits = m["digits"].rstrip(".,")
        if "," in digits and not re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", digits):
            continue  # "1,2,3" is a list, not a number
        plain = digits.replace(",", "")
        scale = _SCALES.get(m["word"] or m["abbr"] or "", 1.0)
        if "." in plain:
            unit = 10.0 ** -len(plain.split(".")[1])
        else:
            unit = 10.0 ** (len(plain) - len(plain.rstrip("0"))) if plain.strip("0") else 1.0
        percent = bool(m["pct"])
        found.append(
            _Number(
                value=float(plain) * scale,
                tolerance=unit / 2 * scale,
                percent=percent,
                marked=percent or bool(m["dollar"]) or "." in plain or scale != 1.0,
                digits=sum(ch.isdigit() for ch in plain),
                text=m.group(0).strip().rstrip(".,"),
            )
        )
    return found


def _without_other_facts(text: str) -> str:
    """The text with dates, times, flight numbers and identifiers blanked out, so their digits
    aren't read as amounts."""
    for pattern in (_DATE_ISO, _TIME, _FLIGHT, _CODE, _PREFIXED_ID, _DATE_TEXT):
        text = pattern.sub(" ", text)
    return text


def _numbers(text: str) -> list[_Number]:
    """The amounts and counts in a text that are facts worth checking."""
    return [n for n in _parse(_without_other_facts(text)) if n.is_fact]


class _Derived:
    """Every value the given numbers lead to in one simple step, sorted for range lookups."""

    ALL_SUBSETS = 12
    """Sums and averages of every subset up to this many numbers; above it, of any two, any
    three of the TOP largest, and all of them."""
    TOP = 30
    PAIRS = 60
    """Differences, products, ratios and percentages of pairs among the PAIRS largest only: a
    tool result with hundreds of numbers would otherwise take seconds."""
    SHARES = 8
    """A number's percentage share of every group it is in, up to this many numbers."""

    def __init__(self, known: list[float], percents: list[float]) -> None:
        known = sorted({k for k in known if k}, reverse=True)
        top = known[: self.TOP]
        plain: set[float] = set(known)
        if len(known) <= self.ALL_SUBSETS:
            groups = [c for r in range(2, len(known) + 1) for c in combinations(known, r)]
        else:
            pairs = known[: self.PAIRS]
            groups = [*combinations(pairs, 2), *combinations(top, 3), tuple(known)]
        sums = {g: sum(g) for g in groups}
        plain |= set(sums.values()) | {s / len(g) for g, s in sums.items()}  # sums, averages
        for a, b in combinations(known[: self.PAIRS], 2):
            plain |= {abs(a - b), a * b, a / b, b / a}
        more = {k * m for k in known for m in range(2, 13)}  # a few of one
        for a, b in combinations(top, 2):  # a few of one plus a few of another
            more |= {a * i + b * j for i in range(1, 13) for j in range(1, 13)}
        pct: set[float] = set(percents)
        for a in known[: self.PAIRS]:
            for b in known[: self.PAIRS]:
                if a != b:
                    pct |= {100 * a / b, 100 * abs(a - b) / b}  # a ratio, a difference
        if len(known) <= self.SHARES:
            pct |= {100 * a / s for g, s in sums.items() for a in g}  # a share of a total
        self.plain = sorted(plain)
        self.plain_more = sorted(plain | more)
        self.percent = sorted(pct)

    @staticmethod
    def _near(values: list[float], n: _Number) -> bool:
        slack = n.tolerance + 1e-9 * abs(n.value)
        i = bisect_left(values, n.value - slack)
        return i < len(values) and values[i] <= n.value + slack

    def explains(self, n: _Number) -> bool:
        if n.percent:
            return self._near(self.percent, n)
        return self._near(self.plain if n.loose else self.plain_more, n)


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_WORKING = re.compile(r"[\d(][\d\s.,()+\-\u2212\u00d7x*/\u00f7=$]*[\d)]")


def _evaluate(expr: str) -> float | None:
    """The value of plain arithmetic ("(17 x 24) + (0.5 x 24)"), or None for anything else."""
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr.replace("$", ""))
    for sign, op in (("\u00d7", "*"), ("x", "*"), ("\u00f7", "/"), ("\u2212", "-")):
        expr = expr.replace(sign, op)
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None

    def value(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](value(node.left), value(node.right))
        raise ValueError(node)

    try:
        return value(tree)
    except (ValueError, ZeroDivisionError):
        return None


def _shown_working(answer: str, given: set[float], derived: _Derived) -> set[float]:
    """Numbers in correct working the answer shows, starting from numbers it was given:
    "17.5 x 24 = (17 x 24) + (0.5 x 24) = 408 + 12 = 420" explains 17, 0.5, 408, 12 and 420."""
    text = re.sub(r"\*\*|__|`", "", answer)
    text = re.sub(r"-\s*=", "=", text)  # "- = 408 + 12": a list marker, not a minus
    explained: set[float] = set()
    for span in _WORKING.finditer(text):
        parts = [p for p in span.group(0).split("=") if p.strip()]
        if len(parts) < 2:
            continue
        values = [_evaluate(p) for p in parts]
        if any(v is None for v in values):
            continue
        first = values[0]
        numbers = [_parse(p) for p in parts]
        if not all(
            abs(v - first) <= (ns[0].tolerance if len(ns) == 1 else 1e-6 * abs(first)) + 1e-9
            for v, ns in zip(values, numbers, strict=True)  # type: ignore[arg-type]
        ):
            continue  # wrong working: its numbers are not explained by it
        if all(n.value in given or derived.explains(n) for n in numbers[0]):
            explained |= {n.value for ns in numbers for n in ns}
    return explained


# ---- the check ------------------------------------------------------------------------------


def _fmt(kind: str, value: Any) -> str:
    if kind == "dates":
        month, day = value.split("-")
        return f"{_MONTH_NAMES[int(month) - 1]} {int(day)}"
    if kind == "flights":
        return re.sub(r"^([A-Z0-9]{2})(\d)", r"\1 \2", value)
    return str(value)


_KINDS = {
    "flights": _flights,
    "codes": _codes,
    "routes": _routes,
    "dates": _dates,
    "times": _times,
}


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


def made_up_facts(run: Run) -> tuple[list[str], list[str]]:
    """(facts in the final answer that nothing given explains, what the tools returned of the
    same kinds), formatted for the report. Both empty when the check doesn't apply."""
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
    # The facts given, and every number of the request (arithmetic asks about small ones too).
    given = [*_numbers(known_text), *_parse(_without_other_facts(run.input_preview))]
    derived = _Derived(
        [n.value for n in given if not n.percent], [n.value for n in given if n.percent]
    )
    working = _shown_working(
        answer, {n.value for n in _parse(_without_other_facts(known_text))}, derived
    )
    missing_numbers = [
        n for n in _numbers(answer) if n.value not in working and not derived.explains(n)
    ]
    made_up += list(dict.fromkeys(n.text for n in missing_numbers))
    if missing_numbers:
        returned += [n.text for n in _numbers(tool_text)]
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
                    # A candidate: serious once the AI judge confirms it, low until then.
                    confidence=0.5,
                    needs_confirmation=True,
                )
            )
        return findings
