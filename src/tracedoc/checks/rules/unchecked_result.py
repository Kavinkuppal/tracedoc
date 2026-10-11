"""verification.unchecked_result: the final answer states facts from tool output, unchecked.

Fires only when the final answer reuses numbers or factual terms that came from a tool result
(not from the user's own request) and no critic or verifier step ran after the last tool result.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ...messages import payload_text
from ...schema import Finding, Run
from ..base import Check, register
from ..util import content_tokens, excerpt, is_handoff_tool

_NUMBER = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")
MIN_SHARED_TERMS = 3
"""Distinctive terms shared by the answer and tool results that count as a factual claim."""


def _numbers(text: str) -> set[str]:
    """Distinctive numbers, normalized ("$1,977,586" -> "1977586"): three or more digits, or a
    decimal, percentage or dollar amount. Bare one- or two-digit integers are too common (list
    numbering, parts of dates) to count as a claim."""
    found = set()
    for raw in _NUMBER.findall(text):
        value = raw.strip("$%").replace(",", "").rstrip(".")
        digits = sum(ch.isdigit() for ch in value)
        marked = "%" in raw or "$" in raw or "." in value
        if digits >= 3 or (digits >= 2 and marked):
            found.add(value)
    return found


def _data_tools(run: Run) -> list:
    """Successful tool calls that returned data (handoff confirmations excluded)."""
    return [
        s
        for s in run.steps_of("tool")
        if s.ended_at and not s.error and not is_handoff_tool(s.name)
    ]


def _values_text(output: Any) -> str:
    """The values in a tool result, without the field names of structured (JSON) output:
    {"departure_airport": "BOS"} contributes "BOS", not "departure_airport"."""
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except ValueError:
            return output
        if not isinstance(parsed, (dict, list)):
            return output
        output = parsed
    if isinstance(output, dict):
        return " ".join(_values_text(v) for v in output.values())
    if isinstance(output, list):
        return " ".join(_values_text(v) for v in output)
    return payload_text(output) if output is not None else ""


def claims_from_tools(run: Run) -> tuple[list[str], list[str]]:
    """(numbers, terms) the final answer took from tool results, excluding the request's own."""
    tool_text = " ".join(_values_text(s.output) for s in _data_tools(run))
    answer, request = run.final_answer, run.input_preview
    numbers = (_numbers(answer) & _numbers(tool_text)) - _numbers(request)
    terms = (content_tokens(answer) & content_tokens(tool_text)) - content_tokens(request)
    terms = {t for t in terms if not any(ch.isdigit() for ch in t)}
    return sorted(numbers), sorted(terms)


def verified_after_last_tool(run: Run) -> bool | None:
    """True/False for completed runs (is there a critic after the last tool output?), else None."""
    if run.status != "completed" or not run.steps_of("node"):
        return None
    tools = _data_tools(run)
    anchor = max((s.ended_at for s in tools), default=run.started_at)  # type: ignore[type-var]
    return any(
        s.role == "critic" and (anchor is None or (s.started_at and s.started_at >= anchor))
        for s in run.steps_of("node")
    )


@register
class UncheckedResult(Check):
    id = "verification.unchecked_result"
    group = "verification"
    method = "rule"
    default_severity = "low"

    def run(self, runs: list[Run]) -> list[Finding]:
        findings = []
        for run in runs:
            if verified_after_last_tool(run) is not False:
                continue
            numbers, terms = claims_from_tools(run)
            if not numbers and len(terms) < MIN_SHARED_TERMS:
                continue  # the answer states nothing taken from tool output
            llm_steps = run.steps_of("llm")
            last = llm_steps[-1] if llm_steps else run.steps_of("node")[-1]
            shown = [*numbers[:4], *terms[: max(0, 6 - min(len(numbers), 4))]]
            critics = [s.node for s in run.steps_of("node") if s.role == "critic" and s.node]
            findings.append(
                self.finding(
                    run,
                    key=last.node or last.name,
                    node=last.node,
                    role=last.role,
                    evidence_step_ids=[last.id],
                    evidence_excerpts=[
                        f"Final answer: {excerpt(run.final_answer, 140)}",
                        f"Taken from tool results: {', '.join(shown)}",
                    ],
                    facts={
                        "agent": last.node or "the agent",
                        "facts": ", ".join(shown),
                        "critic": critics[0] if critics else "",
                    },
                    variant="has_critic" if critics else None,
                    confidence=0.5,
                    needs_confirmation=True,
                )
            )
        return findings
