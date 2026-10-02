"""A scriptable stand-in for the model client. No network."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from tracedoc.checks.judges.runner import TriageItem, TriageResult, Verdict
from tracedoc.llm.client import LLMError, LLMResult

_TRIAGE_CHECK = re.compile(r"^- ([a-z_]+\.[a-z_.]+): ", re.MULTILINE)
_CONFIRM_CHECK = re.compile(r"^CHECK: .* \(([a-z_]+\.[a-z_.]+)\)$", re.MULTILINE)

TriageFn = Callable[[str, str], "tuple[bool, float]"]
ConfirmFn = Callable[[str, str], Verdict]


def always(flag: bool = True, confidence: float = 0.9) -> TriageFn:
    return lambda prompt, check_id: (flag, confidence)


def verdict(ok: bool, confidence: float = 0.9, evidence: list[str] | None = None) -> Verdict:
    return Verdict(
        verdict=ok,
        confidence=confidence,
        evidence_step_ids=evidence or ["S1"],
        explanation="judge explanation" if ok else "",
        suggested_fix="judge fix" if ok else "",
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

    def complete_json(
        self,
        system: str,
        prompt: str,
        schema: type[BaseModel],
        *,
        model: str,
        max_tokens: int = 2_000,
        effort: str | None = None,
    ) -> LLMResult[Any]:
        self.calls.append((model, schema.__name__))
        if self.fail_on and self.fail_on(prompt):
            raise LLMError("fake failure")
        if schema is TriageResult:
            items = []
            for check_id in _TRIAGE_CHECK.findall(prompt.split("For each check below")[-1]):
                flagged, conf = self.triage(prompt, check_id)
                items.append(TriageItem(check_id=check_id, flagged=flagged, confidence=conf))
            value: BaseModel = TriageResult(items=items)
        elif schema is Verdict:
            (check_id,) = _CONFIRM_CHECK.findall(prompt)
            value = self.confirm(prompt, check_id)
        else:  # classifier fallbacks: first allowed value of the single field
            (name, field) = next(iter(schema.model_fields.items()))
            value = schema(**{name: field.annotation.__args__[-1]})  # type: ignore[union-attr]
        return LLMResult(value=value, model=model, tokens_in=len(prompt) // 4, tokens_out=60)
