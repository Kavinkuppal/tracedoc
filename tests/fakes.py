"""A scriptable stand-in for the model client. No network."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from tracedoc.checks.judges.runner import Confirmation, TriageItem, TriageResult, Verdict
from tracedoc.llm.client import LLMError, LLMResult

_TRIAGE_CHECK = re.compile(r"^- ([a-z_]+\.[a-z_.]+): ", re.MULTILINE)
_CONFIRM_CHECK = re.compile(r"^CHECK: ([a-z_]+\.[a-z_.]+) \(prompt v\d+\)$", re.MULTILINE)

TriageFn = Callable[[str, str], "tuple[bool, float]"]
ConfirmFn = Callable[[str, str], Verdict]


def always(flag: bool = True, confidence: float = 0.9) -> TriageFn:
    return lambda prompt, check_id: (flag, confidence)


def verdict(
    ok: bool,
    confidence: float = 0.9,
    evidence: list[str] | None = None,
    *,
    what: str = "judge explanation",
    change: str = "judge fix",
    quote: str | None = None,
) -> Verdict:
    """A judge answer. With quote=None the fake quotes the window's first line (always exact)."""
    return Verdict(
        verdict=ok,
        confidence=confidence,
        evidence_step_ids=evidence or ["S1"],
        what_happened=what if ok else "",
        what_to_change=change if ok else "",
        change_type="prompt instruction",
        evidence_quote="" if quote is None or not ok else quote,
    )


class FakeLLM:
    def __init__(
        self,
        triage: TriageFn | None = None,
        confirm: ConfirmFn | None = None,
        fail_on: Callable[[str], bool] | None = None,
    ) -> None:
        self.triage = triage or always(False)
        self.confirm = confirm or (lambda prompt, check_id: verdict(False))
        self.fail_on = fail_on
        self.calls: list[tuple[str, str]] = []  # (model, schema name)
        self.temperatures: list[float | None] = []
        self.prompts: list[str] = []

    def complete_json(
        self,
        system: str,
        prompt: str,
        schema: type[BaseModel],
        *,
        model: str,
        max_tokens: int = 2_000,
        effort: str | None = None,
        temperature: float | None = None,
    ) -> LLMResult[Any]:
        self.calls.append((model, schema.__name__))
        self.temperatures.append(temperature)
        self.prompts.append(prompt)
        if self.fail_on and self.fail_on(prompt):
            raise LLMError("fake failure")
        if schema is TriageResult:
            items = []
            for check_id in _TRIAGE_CHECK.findall(prompt.split("For each check below")[-1]):
                flagged, conf = self.triage(prompt, check_id)
                items.append(TriageItem(check_id=check_id, flagged=flagged, confidence=conf))
            value: BaseModel = TriageResult(items=items)
        elif schema in (Verdict, Confirmation):
            (check_id,) = _CONFIRM_CHECK.findall(prompt)
            answer = self.confirm(prompt, check_id)
            if schema is Confirmation:
                value = Confirmation(
                    verdict=answer.verdict,
                    confidence=answer.confidence,
                    evidence_step_ids=answer.evidence_step_ids,
                )
            elif answer.verdict and not answer.evidence_quote:
                value = answer.model_copy(update={"evidence_quote": prompt.splitlines()[0][:60]})
            else:
                value = answer
        else:  # classifier fallbacks: first allowed value of the single field
            (name, field) = next(iter(schema.model_fields.items()))
            value = schema(**{name: field.annotation.__args__[-1]})  # type: ignore[union-attr]
        return LLMResult(value=value, model=model, tokens_in=len(prompt) // 4, tokens_out=60)
