"""A retired or unknown triage model falls back to the confirmation model; an unavailable
confirmation model stops with one line. Uses the real Anthropic/OpenAI client wrappers with
stand-in SDK objects that raise the SDKs' own "model not found" errors. No network."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import openai
import pytest
from typer.testing import CliRunner

from tests.fakes import FakeLLM, always, verdict
from tests.helpers import TRACES
from tracedoc import cli
from tracedoc.config import LLMConfig
from tracedoc.llm.client import (
    AnthropicClient,
    ConfirmModelUnavailableError,
    LLMError,
    ModelUnavailableError,
    OpenAIClient,
    with_model_fallback,
)

runner = CliRunner()
CAPTURES = str(TRACES / "capture")

ANTHROPIC = LLMConfig()  # claude-haiku-4-5-20251001 -> claude-sonnet-5-5
OPENAI = LLMConfig(provider="openai", triage_model="gpt-x-mini", confirm_model="gpt-x")
CONFIGS = {"anthropic": ANTHROPIC, "openai": OPENAI}


# ---- the SDKs' own errors -----------------------------------------------------------------------


def _response(status: int, url: str) -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("POST", url))


def anthropic_not_found(model: str) -> anthropic.NotFoundError:
    body = {"type": "error", "error": {"type": "not_found_error", "message": f"model: {model}"}}
    return anthropic.NotFoundError(
        f"Error code: 404 - {body}",
        response=_response(404, "https://api.anthropic.com/v1/messages"),
        body=body,
    )


def openai_not_found(model: str) -> openai.NotFoundError:
    body = {
        "message": f"The model `{model}` does not exist or you do not have access to it.",
        "type": "invalid_request_error",
        "param": None,
        "code": "model_not_found",
    }
    return openai.NotFoundError(
        f"Error code: 404 - {body}",
        response=_response(404, "https://api.openai.com/v1/chat/completions"),
        body=body,
    )


# ---- stand-in SDK objects: answer like the API, or raise "model not found" ---------------------


class FakeAnthropicSDK:
    def __init__(self, unavailable: set[str], oracle: FakeLLM) -> None:
        self.unavailable, self.oracle, self.calls = unavailable, oracle, []
        self.messages = SimpleNamespace(parse=self.parse)

    def parse(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if kw["model"] in self.unavailable:
            raise anthropic_not_found(kw["model"])
        prompt = kw["messages"][0]["content"]
        value = self.oracle.complete_json(
            kw["system"], prompt, kw["output_format"], model=kw["model"]
        ).value
        usage = SimpleNamespace(input_tokens=len(prompt) // 4, output_tokens=40)
        return SimpleNamespace(stop_reason="end_turn", parsed_output=value, usage=usage)


class FakeOpenAISDK:
    def __init__(self, unavailable: set[str], oracle: FakeLLM) -> None:
        self.unavailable, self.oracle, self.calls = unavailable, oracle, []
        completions = SimpleNamespace(parse=self.parse)
        self.chat = SimpleNamespace(completions=completions)

    def parse(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if kw["model"] in self.unavailable:
            raise openai_not_found(kw["model"])
        system, prompt = kw["messages"][0]["content"], kw["messages"][1]["content"]
        value = self.oracle.complete_json(
            system, prompt, kw["response_format"], model=kw["model"]
        ).value
        message = SimpleNamespace(parsed=value, refusal=None, content=None)
        usage = SimpleNamespace(prompt_tokens=len(prompt) // 4, completion_tokens=40)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _client(provider: str, unavailable: set[str]) -> tuple[Any, Any]:
    oracle = FakeLLM(triage=always(True), confirm=lambda p, c: verdict(True, 0.8))
    if provider == "anthropic":
        sdk: Any = FakeAnthropicSDK(unavailable, oracle)
        return AnthropicClient(ANTHROPIC, client=sdk), sdk
    sdk = FakeOpenAISDK(unavailable, oracle)
    return OpenAIClient(OPENAI, client=sdk), sdk


PROVIDERS = ["anthropic", "openai"]


# ---- error mapping ------------------------------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
def test_not_found_becomes_model_unavailable(provider: str) -> None:
    config = CONFIGS[provider]
    client, _ = _client(provider, {config.triage_model})
    with pytest.raises(ModelUnavailableError) as e:
        client.complete_json("s", "p", SimpleNamespace, model=config.triage_model)
    assert e.value.model == config.triage_model
    expected = "not_found_error" if provider == "anthropic" else "model_not_found"
    assert e.value.detail == f"HTTP 404 {expected}"


@pytest.mark.parametrize(
    ("status", "message", "unavailable"),
    [
        (400, "model claude-x has been retired", True),
        (400, "Model claude-x is deprecated and no longer available", True),
        (400, "invalid model: claude-x", True),
        (400, "max_tokens: must be at most 64000", False),
        (500, "internal error", False),
        (429, "rate limited", False),
    ],
)
def test_other_status_errors(status: int, message: str, unavailable: bool) -> None:
    error_cls = {
        400: anthropic.BadRequestError,
        500: anthropic.InternalServerError,
        429: anthropic.RateLimitError,
    }[status]
    error = error_cls(
        message,
        response=_response(status, "https://api.anthropic.com/v1/messages"),
        body={"type": "error", "error": {"type": "invalid_request_error", "message": message}},
    )

    class Raising:
        def parse(self, **kw: Any) -> Any:
            raise error

    client = AnthropicClient(ANTHROPIC, client=SimpleNamespace(messages=Raising()))
    expected = ModelUnavailableError if unavailable else LLMError
    with pytest.raises(expected):
        client.complete_json("s", "p", SimpleNamespace, model="claude-x")


# ---- the fallback wrapper -----------------------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
def test_triage_falls_back_then_goes_straight_to_confirm(provider: str) -> None:
    from tracedoc.checks.judges.runner import TriageResult

    config = CONFIGS[provider]
    base, sdk = _client(provider, {config.triage_model})
    client = with_model_fallback(base, config)
    prompt = "x\n\nFor each check below ...\n- design.loop: q?"
    first = client.complete_json(
        "s", prompt, TriageResult, model=config.triage_model, max_tokens=1_000
    )
    second = client.complete_json(
        "s", prompt, TriageResult, model=config.triage_model, max_tokens=1_000
    )
    assert first.model == second.model == config.confirm_model
    assert [c["model"] for c in sdk.calls] == [
        config.triage_model,
        config.confirm_model,
        config.confirm_model,
    ]
    assert client.fallback is not None
    assert client.fallback.unavailable == config.triage_model
    assert client.resolve(config.triage_model) == config.confirm_model
    # Fallback triage calls get room to think and the confirmation effort.
    token_key = "max_tokens" if provider == "anthropic" else "max_completion_tokens"
    assert all(c[token_key] >= 8_000 for c in sdk.calls[1:])
    if provider == "anthropic":
        assert sdk.calls[1]["output_config"] == {"effort": config.confirm_effort}


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("which", ["confirm", "both"])
def test_unavailable_confirm_model_is_fatal(provider: str, which: str) -> None:
    from tracedoc.checks.judges.runner import TriageResult, Verdict

    config = CONFIGS[provider]
    gone = {config.confirm_model} | ({config.triage_model} if which == "both" else set())
    base, _ = _client(provider, gone)
    client = with_model_fallback(base, config)
    with pytest.raises(ConfirmModelUnavailableError, match=config.confirm_model):
        client.complete_json("s", "CHECK: t (design.loop)", Verdict, model=config.confirm_model)
    if which == "both":
        with pytest.raises(ConfirmModelUnavailableError):
            client.complete_json("s", "- design.loop: q?", TriageResult, model=config.triage_model)


# ---- through the CLI ----------------------------------------------------------------------------


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from tracedoc import config as config_module

    monkeypatch.setattr(config_module, "user_config_path", lambda: tmp_path / "none.toml")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    return tmp_path


def _run(provider: str, unavailable: set[str], tmp_path: Path, monkeypatch: Any) -> tuple:
    base, sdk = _client(provider, unavailable)
    monkeypatch.setattr(cli, "make_client", lambda config: base)
    args = ["analyze", CAPTURES, "--out", str(tmp_path / "out"), "--yes", "--include-unvalidated"]
    if provider == "openai":
        config = tmp_path / "openai.toml"
        config.write_text(
            f'[llm]\nprovider = "openai"\ntriage_model = "{OPENAI.triage_model}"\n'
            f'confirm_model = "{OPENAI.confirm_model}"\n'
        )
        args += ["--config", str(config)]
    return runner.invoke(cli.app, args), sdk


@pytest.mark.parametrize("provider", PROVIDERS)
def test_cli_finishes_with_a_one_line_notice_when_triage_is_retired(
    provider: str, isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = CONFIGS[provider]
    result, sdk = _run(provider, {config.triage_model}, isolated, monkeypatch)
    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    notices = [ln for ln in result.output.splitlines() if ln.startswith("Notice:")]
    assert notices == [
        f"Notice: triage model {config.triage_model} is unavailable (HTTP 404 "
        f"{'not_found_error' if provider == 'anthropic' else 'model_not_found'}); used "
        f"{config.confirm_model} instead. To set a different default, change [llm] "
        "triage_model in your config file (`tracedoc paths` shows where)."
    ]
    report = json.loads((isolated / "out" / "report.json").read_text())
    meta = report["metadata"]
    assert meta["model_fallback"]["unavailable"] == config.triage_model
    assert meta["model_fallback"]["used_instead"] == config.confirm_model
    assert meta["analysis_models"]["triage"] == config.confirm_model
    assert meta["judged"] is True and not meta["judge_errors"]
    # The retired model was tried only until the fallback took over (once per worker at most).
    tried = [c["model"] for c in sdk.calls]
    assert 1 <= tried.count(config.triage_model) <= 4
    assert tried.count(config.confirm_model) > tried.count(config.triage_model)
    assert (isolated / "out" / "report.html").is_file()


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("which", ["confirm", "both"])
def test_cli_stops_with_one_line_when_confirm_model_is_unavailable(
    provider: str, which: str, isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = CONFIGS[provider]
    gone = {config.confirm_model} | ({config.triage_model} if which == "both" else set())
    result, _ = _run(provider, gone, isolated, monkeypatch)
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output
    lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0].startswith(f"Error: confirmation model {config.confirm_model} is unavailable")
    assert "[llm] confirm_model" in lines[0] and "--model" in lines[0]
    assert not (isolated / "out" / "report.json").exists()
    assert "Error:" not in result.stdout  # the progress line was ended, not glued to the error


def test_fallback_needs_no_extra_network_calls(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only messages.parse exists on the stand-in SDK: any model listing would fail loudly."""
    result, sdk = _run("anthropic", {ANTHROPIC.triage_model}, isolated, monkeypatch)
    assert result.exit_code == 0, result.output
    assert set(vars(sdk)) == {"unavailable", "oracle", "calls", "messages"}
