"""Model fallbacks for task typing and role mapping, used only when heuristics give no answer."""

from __future__ import annotations

from functools import cache

from pydantic import BaseModel

from ..schema import Role, TaskType
from .client import LLMClient, LLMError
from .cost import Spend, estimate_tokens

_SYSTEM = "You classify LLM agent runs. Answer with the single best category."


class _TaskAnswer(BaseModel):
    task_type: TaskType


class _RoleAnswer(BaseModel):
    role: Role


def task_classifier(client: LLMClient, model: str, spend: Spend):  # noqa: ANN201
    def classify(preview: str) -> TaskType | None:
        prompt = f"What kind of task is this user request?\n\nREQUEST: {preview}"
        try:
            spend.check(model, estimate_tokens(prompt) + 50, 20)
            result = client.complete_json(_SYSTEM, prompt, _TaskAnswer, model=model, max_tokens=100)
        except LLMError:
            return None
        spend.record(model, result.tokens_in, result.tokens_out)
        return result.value.task_type

    return classify


def role_classifier(client: LLMClient, model: str, spend: Spend):  # noqa: ANN201
    @cache
    def classify(node_name: str) -> Role | None:
        prompt = (
            "Which role best describes a LangGraph node with this name? Use tool_executor only "
            f"for nodes that run tools, and other if unclear.\n\nNODE NAME: {node_name}"
        )
        try:
            spend.check(model, estimate_tokens(prompt) + 50, 20)
            result = client.complete_json(_SYSTEM, prompt, _RoleAnswer, model=model, max_tokens=100)
        except LLMError:
            return None
        spend.record(model, result.tokens_in, result.tokens_out)
        return result.value.role

    return classify
