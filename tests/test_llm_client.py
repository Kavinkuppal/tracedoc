"""The provider wrapper, with fake SDK clients (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from tracedoc.config import LLMConfig
from tracedoc.llm.client import (
    AnthropicClient,
    LLMError,
    MissingKeyError,
    OpenAIClient,
    make_client,
    require_key,
)


class Answer(BaseModel):
    label: str
    score: float


class FakeAnthropicMessages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def parse(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def _anthropic(stop_reason: str = "end_turn", parsed: Any = None) -> tuple[AnthropicClient, Any]:
    response = SimpleNamespace(
        stop_reason=stop_reason,
        parsed_output=parsed,
        usage=SimpleNamespace(input_tokens=120, output_tokens=30),
    )
    messages = FakeAnthropicMessages(response)
    return AnthropicClient(LLMConfig(), client=SimpleNamespace(messages=messages)), messages


def test_anthropic_uses_native_structured_output() -> None:
    client, messages = _anthropic(parsed=Answer(label="x", score=0.5))
    result = client.complete_json("sys", "hi", Answer, model="claude-opus-5", effort="low")
    assert result.value == Answer(label="x", score=0.5)
    assert (result.tokens_in, result.tokens_out) == (120, 30)
    assert messages.kwargs["output_format"] is Answer
    assert messages.kwargs["output_config"] == {"effort": "low"}
    assert messages.kwargs["system"] == "sys"


def test_anthropic_omits_effort_when_unset() -> None:
    client, messages = _anthropic(parsed=Answer(label="x", score=0.5))
    client.complete_json("sys", "hi", Answer, model="claude-haiku-4-5")
    assert "output_config" not in messages.kwargs


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_anthropic_refusal_or_truncation_raises(stop_reason: str) -> None:
    client, _ = _anthropic(stop_reason=stop_reason, parsed=None)
    with pytest.raises(LLMError):
        client.complete_json("sys", "hi", Answer, model="m")


def _openai(contents: list[str], native: bool, parsed: Any = None) -> tuple[OpenAIClient, list]:
    calls: list[dict[str, Any]] = []
    replies = iter(contents)

    def respond(**kwargs: Any) -> Any:
        calls.append(kwargs)
        message = SimpleNamespace(content=next(replies, ""), parsed=parsed, refusal=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    completions = SimpleNamespace(parse=respond, create=respond)
    provider = "openai" if native else "openai_compatible"
    config = LLMConfig(provider=provider, base_url=None if native else "http://localhost:1")
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIClient(config, client=sdk), calls


def test_openai_native_parse() -> None:
    client, calls = _openai([""], native=True, parsed=Answer(label="y", score=1))
    result = client.complete_json("sys", "hi", Answer, model="gpt-x")
    assert result.value.label == "y"
    assert calls[0]["response_format"] is Answer


def test_compatible_strips_fences() -> None:
    client, calls = _openai(['```json\n{"label": "z", "score": 0.2}\n```'], native=False)
    result = client.complete_json("sys", "hi", Answer, model="llama")
    assert result.value == Answer(label="z", score=0.2)
    assert '"label"' in calls[0]["messages"][0]["content"]  # schema is in the prompt


def test_compatible_retries_invalid_json_once() -> None:
    client, calls = _openai(['{"label": "z"}', '{"label": "z", "score": 1}'], native=False)
    result = client.complete_json("sys", "hi", Answer, model="llama")
    assert result.value.score == 1
    assert len(calls) == 2
    assert result.tokens_in == 20
    assert "not valid" in calls[1]["messages"][-1]["content"]


def test_compatible_gives_up_after_retry() -> None:
    client, _ = _openai(["nope", "still nope"], native=False)
    with pytest.raises(LLMError):
        client.complete_json("sys", "hi", Answer, model="llama")


def test_make_client_requires_base_url_for_compatible() -> None:
    with pytest.raises(ValueError):
        make_client(LLMConfig(provider="openai_compatible"))


def test_keys_come_only_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MY_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(MissingKeyError) as e:
        require_key(LLMConfig())
    assert e.value.env_var == "ANTHROPIC_API_KEY"
    with pytest.raises(MissingKeyError):
        require_key(LLMConfig(provider="openai"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")  # blank counts as missing
    with pytest.raises(MissingKeyError):
        require_key(LLMConfig())
    monkeypatch.setenv("MY_KEY", "k")
    assert require_key(LLMConfig(api_key_env="MY_KEY")) == "k"


def test_config_cannot_hold_a_key() -> None:
    with pytest.raises(ValueError):
        LLMConfig(api_key="sk-anything")  # type: ignore[call-arg]


def test_local_endpoint_never_receives_the_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key-should-not-leak")
    config = LLMConfig(provider="openai_compatible", base_url="http://localhost:11434/v1")
    assert require_key(config) is None
    client = make_client(config)
    assert client.client.api_key == "not-needed"


def test_anthropic_client_uses_the_env_key_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-token-should-be-ignored")
    client = make_client(LLMConfig())
    assert client.client.api_key == "env-key"


def test_rejected_key_is_fatal_not_a_per_call_error() -> None:
    import anthropic
    import httpx2 as httpx

    from tracedoc.llm.client import KeyRejectedError

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.AuthenticationError(
        "bad key", response=httpx.Response(401, request=request), body=None
    )

    class Raising:
        def parse(self, **kwargs: Any) -> Any:
            raise error

    client = AnthropicClient(LLMConfig(), client=SimpleNamespace(messages=Raising()))
    with pytest.raises(KeyRejectedError, match="HTTP 401"):
        client.complete_json("s", "p", Answer, model="m")
