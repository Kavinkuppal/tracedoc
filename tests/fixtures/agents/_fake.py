"""A scripted chat model for deterministic fixture agents.

Extends LangChain's fake-model idea with two things the fixtures need:
`bind_tools` (so agents can be written the normal way) and `usage_metadata`
(so traces carry token counts). Token counts are derived from text length,
so they are deterministic and grow with context like a real model's would.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

MODEL_NAME = "fake-scripted-1"


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _message_text(message: BaseMessage) -> str:
    text = message.content if isinstance(message.content, str) else str(message.content)
    for call in getattr(message, "tool_calls", None) or []:
        text += f" {call['name']} {call['args']}"
    return text


def ai(content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> AIMessage:
    """Build a scripted reply. Tool calls are given as {"name", "args", "id"}."""
    calls = [{"type": "tool_call", **call} for call in tool_calls or []]
    return AIMessage(content=content, tool_calls=calls)


class ScriptedChatModel(BaseChatModel):
    """Returns `responses` in order. With `cycle=True` it loops forever."""

    responses: list[AIMessage]
    cycle: bool = False
    model_name: str = MODEL_NAME
    i: int = 0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
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
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}
