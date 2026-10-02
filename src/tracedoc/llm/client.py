"""One interface over Anthropic, OpenAI, and OpenAI-compatible endpoints.

`complete_json(system, prompt, schema, model=...)` returns a validated instance of
the Pydantic `schema` plus token usage. Anthropic and OpenAI use their SDKs' native
structured-output parsing; OpenAI-compatible endpoints (local models) get the JSON
schema in the prompt and a validation retry, since support for schema-constrained
decoding varies between servers.

HTTP-level retries with backoff (429, 5xx, timeouts, connection errors) are done by
the SDKs, configured with `max_retries`. Invalid JSON is retried here.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import LLMConfig

T = TypeVar("T", bound=BaseModel)

_DEFAULT_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai_compatible": "OPENAI_API_KEY",
}
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMError(RuntimeError):
    """The model could not produce a valid answer (refusal, bad JSON after retries, API error)."""


@dataclass
class LLMResult(Generic[T]):
    value: T
    model: str
    tokens_in: int
    tokens_out: int


class LLMClient(Protocol):
    def complete_json(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        model: str,
        max_tokens: int = 2_000,
        effort: str | None = None,
    ) -> LLMResult[T]: ...


def _api_key(config: LLMConfig) -> str | None:
    env = config.api_key_env or _DEFAULT_KEY_ENV[config.provider]
    return os.environ.get(env) or config.api_key


class AnthropicClient:
    def __init__(self, config: LLMConfig, client: Any = None) -> None:
        if client is None:
            import anthropic

            key = _api_key(config)
            client = anthropic.Anthropic(
                **({"api_key": key} if key else {}),  # else the SDK resolves profiles/tokens
                max_retries=config.max_retries,
                timeout=config.timeout_s,
            )
        self.client = client

    def complete_json(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        model: str,
        max_tokens: int = 2_000,
        effort: str | None = None,
    ) -> LLMResult[T]:
        import anthropic

        kwargs: dict[str, Any] = {}
        if effort is not None:
            kwargs["output_config"] = {"effort": effort}
        try:
            response = self.client.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
                **kwargs,
            )
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}") from e
        if response.stop_reason == "refusal":
            raise LLMError("the model declined to answer")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise LLMError(f"no parsable output (stop_reason={response.stop_reason})")
        return LLMResult(
            value=response.parsed_output,
            model=model,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
        )


class OpenAIClient:
    def __init__(self, config: LLMConfig, client: Any = None) -> None:
        self.native = config.provider == "openai"
        if client is None:
            import openai

            client = openai.OpenAI(
                api_key=_api_key(config) or ("not-needed" if not self.native else None),
                base_url=config.base_url,
                max_retries=config.max_retries,
                timeout=config.timeout_s,
            )
        self.client = client

    def complete_json(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        model: str,
        max_tokens: int = 2_000,
        effort: str | None = None,
    ) -> LLMResult[T]:
        import openai

        try:
            if self.native:
                return self._native(system, prompt, schema, model, max_tokens)
            return self._compatible(system, prompt, schema, model, max_tokens)
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e

    def _native(
        self, system: str, prompt: str, schema: type[T], model: str, max_tokens: int
    ) -> LLMResult[T]:
        completion = self.client.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format=schema,
            max_completion_tokens=max_tokens,
        )
        message = completion.choices[0].message
        if getattr(message, "refusal", None) or message.parsed is None:
            raise LLMError("the model declined to answer or returned no parsable output")
        usage = completion.usage
        return LLMResult(
            value=message.parsed,
            model=model,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
        )

    def _compatible(
        self, system: str, prompt: str, schema: type[T], model: str, max_tokens: int
    ) -> LLMResult[T]:
        instructions = (
            f"{system}\n\nRespond with only a JSON object (no prose, no code fences) that "
            f"matches this JSON schema:\n{json.dumps(schema.model_json_schema())}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ]
        tokens_in = tokens_out = 0
        last_error = ""
        for _ in range(2):
            completion = self.client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens
            )
            if completion.usage:
                tokens_in += completion.usage.prompt_tokens
                tokens_out += completion.usage.completion_tokens
            text = completion.choices[0].message.content or ""
            try:
                value = schema.model_validate_json(_FENCE.sub("", text.strip()))
                return LLMResult(
                    value=value, model=model, tokens_in=tokens_in, tokens_out=tokens_out
                )
            except ValidationError as e:
                last_error = str(e)[:500]
                messages += [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": f"That was not valid. Errors: {last_error}. "
                        "Reply with only the corrected JSON object.",
                    },
                ]
        raise LLMError(f"invalid JSON from {model} after retry: {last_error}")


def make_client(config: LLMConfig) -> LLMClient:
    if config.provider == "anthropic":
        return AnthropicClient(config)
    if config.provider == "openai_compatible" and not config.base_url:
        raise ValueError("llm.base_url is required for the openai_compatible provider")
    return OpenAIClient(config)


def has_credentials(config: LLMConfig) -> bool:
    """Best effort: whether a key is configured. Anthropic can also use `ant auth` profiles."""
    return config.provider == "openai_compatible" or _api_key(config) is not None
