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
import threading
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import LLMConfig

T = TypeVar("T", bound=BaseModel)

DEFAULT_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
README_KEYS = "README: 'Set up your API key'"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMError(RuntimeError):
    """The model could not produce a valid answer (refusal, bad JSON after retries, API error)."""


class MissingKeyError(RuntimeError):
    """No key in the environment for the configured provider."""

    def __init__(
        self, env_var: str, hint: str = "or run with --no-judge for rule checks only"
    ) -> None:
        self.env_var = env_var
        super().__init__(
            f"{env_var} is not set. Set it to your own API key (see {README_KEYS}), {hint}."
        )


class ModelUnavailableError(RuntimeError):
    """The provider says the requested model does not exist, was retired, or is not available
    to this key (for example HTTP 404 not_found / model_not_found)."""

    def __init__(self, model: str, detail: str) -> None:
        self.model = model
        self.detail = detail
        super().__init__(f"model {model} is unavailable ({detail})")


_MODEL_GONE = re.compile(
    r"model.{0,80}(not[ _]found|does not exist|not exist|retired|deprecated|no longer "
    r"(available|supported)|is not available|not supported|unknown)|"
    r"(unknown|invalid|unsupported)[ _]model|model_not_found",
    re.IGNORECASE | re.DOTALL,
)


def _status_error_detail(error: Any) -> str:
    """A short, value-free description of an SDK status error, e.g. 'HTTP 404 not_found_error'."""
    body = getattr(error, "body", None)
    kind = ""
    if isinstance(body, dict):
        inner = body.get("error") if isinstance(body.get("error"), dict) else body
        kind = str(inner.get("code") or inner.get("type") or "")  # specific code beats generic type
    return f"HTTP {getattr(error, 'status_code', '?')}{' ' + kind if kind else ''}"


def is_model_unavailable(error: Any) -> bool:
    """404s from a model endpoint, and 400s whose error says the model is gone or unknown."""
    status = getattr(error, "status_code", None)
    if status == 404:
        return True
    if status == 400:
        body = getattr(error, "body", None)
        text = f"{getattr(error, 'message', '')} {getattr(error, 'code', '') or ''} {body}"
        return bool(_MODEL_GONE.search(text))
    return False


class KeyRejectedError(RuntimeError):
    """The provider rejected the key (401/403). Fatal: stop instead of retrying every call."""

    def __init__(self, env_var: str | None, detail: str) -> None:
        name = env_var or "the API key"
        super().__init__(
            f"The provider rejected {name} ({detail}). Check the key; see {README_KEYS}."
        )


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


def key_env_var(config: LLMConfig) -> str | None:
    """The environment variable this provider's key is read from (None: no key is sent)."""
    return config.api_key_env or DEFAULT_KEY_ENV.get(config.provider)


def require_key(config: LLMConfig) -> str | None:
    """Return the key from the environment, or raise MissingKeyError. Keys never come from
    config files, and there is no built-in or fallback key."""
    env_var = key_env_var(config)
    if env_var is None:
        return None
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise MissingKeyError(env_var)
    return key


class AnthropicClient:
    def __init__(self, config: LLMConfig, client: Any = None) -> None:
        self.env_var = key_env_var(config)
        if client is None:
            import anthropic

            client = anthropic.Anthropic(
                api_key=require_key(config),  # explicit: never a profile or other ambient token
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
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise KeyRejectedError(self.env_var, f"HTTP {e.status_code}") from e
        except anthropic.APIStatusError as e:
            if is_model_unavailable(e):
                raise ModelUnavailableError(model, _status_error_detail(e)) from e
            raise LLMError(f"Anthropic API error: {e}") from e
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
        self.env_var = key_env_var(config)
        if client is None:
            import openai

            client = openai.OpenAI(
                # A local endpoint gets no key unless api_key_env names one; the SDK needs a value.
                api_key=require_key(config) or "not-needed",
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
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            raise KeyRejectedError(self.env_var, f"HTTP {e.status_code}") from e
        except openai.APIStatusError as e:
            if is_model_unavailable(e):
                raise ModelUnavailableError(model, _status_error_detail(e)) from e
            raise LLMError(f"OpenAI API error: {e}") from e
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


@dataclass
class ModelFallback:
    unavailable: str
    replacement: str
    detail: str

    def notice(self) -> str:
        return (
            f"Notice: triage model {self.unavailable} is unavailable ({self.detail}); used "
            f"{self.replacement} instead. To set a different default, change [llm] "
            "triage_model in your config file (`tracedoc paths` shows where)."
        )


class ConfirmModelUnavailableError(RuntimeError):
    """The confirmation model is unavailable: there is nothing left to fall back to."""

    def __init__(self, model: str, detail: str) -> None:
        self.model = model
        super().__init__(
            f"confirmation model {model} is unavailable ({detail}). Set a different model "
            "under [llm] confirm_model in your config file (`tracedoc paths` shows where), "
            "or pass --model."
        )


class FallbackClient:
    """Wraps a client so a retired or unknown triage model falls back to the confirmation model.

    The first time the triage model is unavailable, that call is retried on the confirmation
    model, and every later triage call goes straight there for the rest of the run. Triage
    calls on the fallback get the confirmation effort and a larger token cap, since stronger
    models may think before answering. If the confirmation model is unavailable, raise
    ConfirmModelUnavailableError. Decided purely from the errors calls already return: no
    extra network requests, no model lists.
    """

    def __init__(
        self,
        client: LLMClient,
        config: LLMConfig,
        *,
        fallback_min_tokens: int = 8_000,
    ) -> None:
        self.client = client
        self.triage_model = config.triage_model
        self.confirm_model = config.confirm_model
        self.confirm_effort = config.confirm_effort
        self.fallback_min_tokens = fallback_min_tokens
        self.fallback: ModelFallback | None = None
        self._lock = threading.Lock()

    def resolve(self, model: str) -> str:
        """The model a call for `model` will actually use (for budget checks and accounting)."""
        if self.fallback is not None and model == self.triage_model:
            return self.confirm_model
        return model

    def _confirm_call(
        self, system: str, prompt: str, schema: type[T], max_tokens: int, effort: str | None
    ) -> LLMResult[T]:
        try:
            return self.client.complete_json(
                system,
                prompt,
                schema,
                model=self.confirm_model,
                max_tokens=max_tokens,
                effort=effort,
            )
        except ModelUnavailableError as e:
            raise ConfirmModelUnavailableError(self.confirm_model, e.detail) from None

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
        is_triage = model == self.triage_model and model != self.confirm_model
        if model == self.confirm_model:
            return self._confirm_call(system, prompt, schema, max_tokens, effort)
        if is_triage and self.fallback is not None:
            return self._confirm_call(
                system,
                prompt,
                schema,
                max(max_tokens, self.fallback_min_tokens),
                effort or self.confirm_effort,
            )
        try:
            return self.client.complete_json(
                system, prompt, schema, model=model, max_tokens=max_tokens, effort=effort
            )
        except ModelUnavailableError as e:
            if not is_triage:
                raise
            with self._lock:
                if self.fallback is None:
                    self.fallback = ModelFallback(model, self.confirm_model, e.detail)
            return self._confirm_call(
                system,
                prompt,
                schema,
                max(max_tokens, self.fallback_min_tokens),
                effort or self.confirm_effort,
            )


def with_model_fallback(client: LLMClient, config: LLMConfig) -> FallbackClient:
    return FallbackClient(client, config)
