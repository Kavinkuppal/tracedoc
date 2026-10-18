"""A scripted chat model for deterministic fixture agents.

Extends LangChain's fake-model idea with what the fixtures need:
`bind_tools` (so agents can be written the normal way), `usage_metadata`
(so traces carry token counts), and, on request, the provider-side check that
every tool call in the history has a result. Token counts are derived from text
length, so they are deterministic and grow with context like a real model's would.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

MODEL_NAME = "fake-scripted-1"


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _message_text(message: BaseMessage) -> str:
    text = message.content if isinstance(message.content, str) else str(message.content)
    for call in getattr(message, "tool_calls", None) or []:
        text += f" {call['name']} {call['args']}"
    return text


class InvalidRequestError(Exception):
    """What a provider's API raises for a malformed request (an HTTP 400)."""


def _unanswered_calls(messages: list[BaseMessage]) -> list[str]:
    pending: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage):
            pending += [call["id"] for call in message.tool_calls if call.get("id")]
        elif isinstance(message, ToolMessage) and message.tool_call_id in pending:
            pending.remove(message.tool_call_id)
    return pending


def ai(content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> AIMessage:
    """Build a scripted reply. Tool calls are given as {"name", "args", "id"}."""
    calls = [{"type": "tool_call", **call} for call in tool_calls or []]
    return AIMessage(content=content, tool_calls=calls)


class ScriptedChatModel(BaseChatModel):
    """Returns `responses` in order. With `cycle=True` it loops forever. With
    `require_tool_results=True` it rejects a history with a tool call that has no result, as
    Anthropic's API does."""

    responses: list[AIMessage]
    cycle: bool = False
    require_tool_results: bool = False
    model_name: str = MODEL_NAME
    i: int = 0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        unanswered = _unanswered_calls(messages) if self.require_tool_results else []
        if unanswered:
            raise InvalidRequestError(
                "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
                "'message': '`tool_use` ids were found without `tool_result` blocks immediately "
                f"after: {', '.join(unanswered)}. Each `tool_use` block must have a corresponding "
                "`tool_result` block in the next message.'}}"
            )
        if self.i >= len(self.responses):
            if not self.cycle:
                raise RuntimeError(f"ScriptedChatModel ran out of responses after {self.i}")
            self.i = 0
        scripted = self.responses[self.i]
        self.i += 1
        tokens_in = sum(_count_tokens(_message_text(m)) for m in messages)
        tokens_out = _count_tokens(_message_text(scripted))
        message = scripted.model_copy(
            update={
                "usage_metadata": {
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_out,
                    "total_tokens": tokens_in + tokens_out,
                },
                "response_metadata": {"model_name": self.model_name},
            }
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        # A binding, like a real chat model's, so LangGraph sees the tools as bound (the
        # scripted replies ignore them).
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}
