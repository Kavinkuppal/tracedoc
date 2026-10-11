"""Model fallbacks for task typing and role mapping, used only when heuristics give no answer."""

from __future__ import annotations

from functools import cache

from pydantic import BaseModel

from ..schema import Role, TaskType
from .client import LLMClient, LLMError, LLMResult
from .cost import Spend, estimate_tokens

_SYSTEM = "You classify LLM agent runs. Answer with the single best category."


def _resolve(client: LLMClient, model: str) -> str:
    resolve = getattr(client, "resolve", None)
    return resolve(model) if resolve else model


def _ask(
    client: LLMClient,
    spend: Spend,
    prompt: str,
    schema: type[BaseModel],
    model: str,
    temperature: float | None,
) -> LLMResult | None:  # type: ignore[type-arg]
    """A cached answer if there is one; otherwise a budget-checked call. None on model errors."""
    settings = {"model": model, "max_tokens": 100, "temperature": temperature}
    lookup = getattr(client, "lookup", None)
    try:
        result = lookup(_SYSTEM, prompt, schema, **settings) if lookup else None
        if result is None:
            spend.check(_resolve(client, model), estimate_tokens(prompt) + 50, 20)
            result = client.complete_json(_SYSTEM, prompt, schema, **settings)
    except LLMError:
        return None
    spend.record_result(result)
    return result


class _TaskAnswer(BaseModel):
    task_type: TaskType


class _RoleAnswer(BaseModel):
    role: Role


def task_classifier(  # noqa: ANN201
    client: LLMClient, model: str, spend: Spend, temperature: float | None = None
):
    def classify(preview: str) -> TaskType | None:
        prompt = f"What kind of task is this user request?\n\nREQUEST: {preview}"
        result = _ask(client, spend, prompt, _TaskAnswer, model, temperature)
        return result.value.task_type if result else None

    return classify


def role_classifier(  # noqa: ANN201
    client: LLMClient, model: str, spend: Spend, temperature: float | None = None
):
    @cache
    def classify(node_name: str) -> Role | None:
        prompt = (
            "Which role best describes a LangGraph node with this name? Use tool_executor only "
            f"for nodes that run tools, and other if unclear.\n\nNODE NAME: {node_name}"
        )
        result = _ask(client, spend, prompt, _RoleAnswer, model, temperature)
        return result.value.role if result else None

    return classify
