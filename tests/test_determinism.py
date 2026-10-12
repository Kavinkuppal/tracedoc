"""Same traces, same report: judge cache, temperature 0, slot limits, and order-free results.

No network: stand-in SDK objects and the scriptable FakeLLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import openai
import pytest
from pydantic import BaseModel

from tests.fakes import FakeLLM, always, verdict
from tests.fixtures.agents import AGENTS
from tests.helpers import load_fixture
from tracedoc.catalog import limit
from tracedoc.checks.base import run_rule_checks
from tracedoc.checks.judges.prompts import enabled_judges
from tracedoc.checks.judges.runner import Judge, Verdict, _dedupe
from tracedoc.checks.judges.slots import fitted, slot_problems, truncate_words
from tracedoc.checks.judges.windows import run_end_window
from tracedoc.config import Config, LLMConfig
from tracedoc.llm.cache import CachingClient, cache_key
from tracedoc.llm.client import AnthropicClient, LLMResult, OpenAIClient, with_model_fallback
from tracedoc.llm.cost import Spend
from tracedoc.schema import Finding


class Answer(BaseModel):
    label: str


def _config() -> Config:
    config = Config()
    config.judge.include_unvalidated = True
    return config


def _judge(config: Config, client: Any, spend: Spend | None = None) -> Judge:
    judges, _ = enabled_judges(config)
    return Judge(config, judges, client=client, spend=spend)


def _runs() -> list:
    return [load_fixture(n) for n in AGENTS]


# ---- slots ---------------------------------------------------------------------------------


def test_truncate_cuts_at_a_word_boundary() -> None:
    text = "The agent booked a flight from Boston although the user asked for Seattle " * 5
    cut = truncate_words(text, 200)
    assert len(cut) <= 200
    assert cut.endswith("…")
    assert text.startswith(cut[:-1])  # nothing rewritten, only cut
    assert not cut[:-1].endswith(" ")
    assert truncate_words("short", 200) == "short"
    assert truncate_words("a\n\nb   c", 200) == "a b c"


def test_every_slot_limit_is_enforced() -> None:
    flights = "BOS to JFK on Jet Blue, " * 20
    window = f"USER TASK: book a flight\n[S1] TOOL CALL search_flights\n  result: {flights}"
    long = "word " * 100
    answer = verdict(True, what=long, change=long, quote=flights)  # exact, but too long
    problems = slot_problems(answer, window)
    assert len(problems) == 3
    fit = fitted(answer, window)
    for slot in ("what_happened", "what_to_change", "evidence_quote"):
        assert 0 < len(fit[slot]) <= limit(slot), slot
    assert limit("what_happened") == limit("what_to_change") == 200
    assert limit("evidence_quote") == 120


def test_quote_must_be_copied_exactly_from_the_window() -> None:
    window = "USER TASK: book a flight\n[S1] TOOL CALL search_flights\n  result: BOS   to\nJFK"
    exact = verdict(True, quote="result: BOS to JFK")  # whitespace differences are fine
    assert slot_problems(exact, window) == []
    assert fitted(exact, window)["evidence_quote"] == "result: BOS to JFK"
    made_up = verdict(True, quote="result: LAX to SEA")
    assert any("not copied exactly" in p for p in slot_problems(made_up, window))
    assert fitted(made_up, window)["evidence_quote"] == ""


def _flagging_judge(answers: list[Verdict]) -> tuple[Judge, FakeLLM]:
    """A judge whose confirmation calls return `answers` in turn (the last one repeats)."""
    replies = iter(answers)
    last: list[Verdict] = []

    def confirm(prompt: str, check_id: str) -> Verdict:
        last[:] = [next(replies, last[0] if last else answers[-1])]
        return last[0]

    fake = FakeLLM(triage=always(True), confirm=confirm)
    config = _config()
    judges, _ = enabled_judges(config)
    only = {"verification.premature_finish": judges["verification.premature_finish"]}
    return Judge(config, only, client=fake), fake


def test_slot_breaking_answer_is_retried_once_then_cut() -> None:
    too_long = verdict(True, what="x " * 150, change="change " * 40, quote="not in the window")
    judge, fake = _flagging_judge([too_long, too_long])
    run = load_fixture("researcher_repeat_search")
    out = judge.run(judge.plan([run], []), [])
    confirms = [c for c in fake.calls if c[1] == "Verdict"]
    assert len(confirms) == 2  # the answer, and one retry
    assert "YOUR PREVIOUS ANSWER" in fake.prompts[-1]
    assert "the limit is 200" in fake.prompts[-1]
    (finding,) = out.findings
    assert len(finding.explanation) <= 200
    assert len(finding.suggested_fix) <= 200
    assert finding.evidence_quote  # replaced by an excerpt of the cited step
    assert len(finding.evidence_quote) <= 120


def test_answer_within_limits_is_not_retried() -> None:
    judge, fake = _flagging_judge([verdict(True, what="Ended early.", change="Add a step.")])
    out = judge.run(judge.plan([load_fixture("researcher_repeat_search")], []), [])
    assert [c[1] for c in fake.calls].count("Verdict") == 1
    (finding,) = out.findings
    assert finding.explanation == "Ended early."
    assert finding.change_type == "prompt instruction"
    assert finding.evidence_quote.startswith("USER TASK:")


# ---- temperature ---------------------------------------------------------------------------


def _response(status: int, url: str) -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("POST", url))


def _anthropic_temperature_error() -> anthropic.BadRequestError:
    # Stand-in wording: the real message may differ; any 400 that names the setting counts.
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "temperature is not supported"},
    }
    return anthropic.BadRequestError(
        f"Error code: 400 - {body}",
        response=_response(400, "https://api.anthropic.com/v1/messages"),
        body=body,
    )


def _openai_temperature_error() -> openai.BadRequestError:
    body = {
        "message": "Unsupported value: 'temperature' does not support 0 with this model.",
        "type": "invalid_request_error",
        "param": "temperature",
        "code": "unsupported_value",
    }
    return openai.BadRequestError(
        f"Error code: 400 - {body}",
        response=_response(400, "https://api.openai.com/v1/chat/completions"),
        body=body,
    )


class _AnthropicSDK:
    def __init__(self, rejects: set[str]) -> None:
        self.rejects, self.calls = rejects, []
        self.messages = SimpleNamespace(parse=self.parse)

    def parse(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if "extra_body" in kw and kw["model"] in self.rejects:
            raise _anthropic_temperature_error()
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        return SimpleNamespace(stop_reason="end_turn", parsed_output=Answer(label="a"), usage=usage)


class _OpenAISDK:
    def __init__(self, rejects: set[str]) -> None:
        self.rejects, self.calls = rejects, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))

    def parse(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if "temperature" in kw and kw["model"] in self.rejects:
            raise _openai_temperature_error()
        message = SimpleNamespace(parsed=Answer(label="a"), refusal=None, content=None)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_anthropic_sends_temperature_and_drops_it_for_models_that_refuse() -> None:
    sdk = _AnthropicSDK(rejects={"claude-sonnet-5-5"})
    client = AnthropicClient(LLMConfig(), client=sdk)
    client.complete_json("s", "p", Answer, model="claude-haiku-4-5", temperature=0.0)
    assert sdk.calls[-1]["extra_body"] == {"temperature": 0.0}

    client.complete_json("s", "p", Answer, model="claude-sonnet-5-5", temperature=0.0)
    assert [("extra_body" in c) for c in sdk.calls[1:]] == [True, False]  # refused, then without
    client.complete_json("s", "p", Answer, model="claude-sonnet-5-5", temperature=0.0)
    assert "extra_body" not in sdk.calls[-1]  # remembered: no second refusal
    assert len(sdk.calls) == 4

    client.complete_json("s", "p", Answer, model="claude-haiku-4-5")
    assert "extra_body" not in sdk.calls[-1]  # no temperature asked for, none sent


def test_openai_drops_temperature_for_models_that_refuse() -> None:
    sdk = _OpenAISDK(rejects={"reasoning-x"})
    client = OpenAIClient(LLMConfig(provider="openai"), client=sdk)
    client.complete_json("s", "p", Answer, model="gpt-x", temperature=0.0)
    assert sdk.calls[-1]["temperature"] == 0.0
    client.complete_json("s", "p", Answer, model="reasoning-x", temperature=0.0)
    assert "temperature" not in sdk.calls[-1]
    assert len(sdk.calls) == 3


def test_judge_asks_for_temperature_zero() -> None:
    fake = FakeLLM(triage=always(True), confirm=lambda p, c: verdict(True))
    judge = _judge(_config(), fake)
    run = load_fixture("researcher_repeat_search")
    judge.run(judge.plan([run], run_rule_checks([run])), run_rule_checks([run]))
    assert fake.temperatures and set(fake.temperatures) == {0.0}


# ---- cache ---------------------------------------------------------------------------------


class _Counting:
    """Answers every call with the same label and counts the calls that reach it."""

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, system: str, prompt: str, schema: Any, **kw: Any) -> LLMResult:
        self.calls += 1
        return LLMResult(
            value=Answer(label=prompt), model=kw["model"], tokens_in=1000, tokens_out=100
        )


def test_cache_answers_repeat_calls_without_the_model(tmp_path: Path) -> None:
    inner = _Counting()
    client = CachingClient(inner, tmp_path)
    first = client.complete_json("s", "p", Answer, model="m", temperature=0.0)
    second = client.complete_json("s", "p", Answer, model="m", temperature=0.0)
    assert inner.calls == 1
    assert not first.cached and second.cached
    assert second.value == first.value
    assert (second.tokens_in, second.tokens_out) == (1000, 100)  # what it cost originally
    client.complete_json("s", "other window", Answer, model="m", temperature=0.0)
    client.complete_json("s", "p", Answer, model="other-model", temperature=0.0)
    client.complete_json("s", "p", Answer, model="m", effort="low", temperature=0.0)
    assert inner.calls == 4  # window, model and settings are all part of the key


def test_unreadable_cache_entry_is_a_miss(tmp_path: Path) -> None:
    inner = _Counting()
    client = CachingClient(inner, tmp_path)
    client.complete_json("s", "p", Answer, model="m")
    key = cache_key("s", "p", Answer, model="m", max_tokens=2000, effort=None, temperature=None)
    (tmp_path / key[:2] / f"{key}.json").write_text("{not json")
    assert client.lookup("s", "p", Answer, model="m") is None
    client.complete_json("s", "p", Answer, model="m")
    assert inner.calls == 2
    assert client.lookup("s", "p", Answer, model="m") is not None  # rewritten


def test_cache_files_hold_no_secrets_or_paths(tmp_path: Path) -> None:
    client = CachingClient(_Counting(), tmp_path)
    client.complete_json("s", "p", Answer, model="m")
    (entry,) = tmp_path.rglob("*.json")
    assert set(json.loads(entry.read_text())) == {
        "format",
        "model",
        "tokens_in",
        "tokens_out",
        "value",
    }


def _findings_text(findings: list[Finding]) -> list[str]:
    return [f.model_dump_json() for f in sorted(findings, key=lambda f: f.id)]


def test_rerun_from_cache_gives_identical_findings_and_cost(tmp_path: Path) -> None:
    runs = _runs()
    rules = run_rule_checks(runs)
    config = _config()

    def judge_once(client: Any) -> tuple[list[Finding], Spend]:
        spend = Spend(prices=config.prices)
        judge = _judge(config, client, spend)
        return judge.run(judge.plan(runs, rules), rules).findings, spend

    fake = FakeLLM(triage=always(True), confirm=lambda p, c: verdict(True, 0.85))
    fresh, spent = judge_once(with_model_fallback(CachingClient(fake, tmp_path), config.llm))
    calls = len(fake.calls)
    again, reused = judge_once(with_model_fallback(CachingClient(fake, tmp_path), config.llm))
    assert len(fake.calls) == calls  # every answer came from the cache
    assert _findings_text(again) == _findings_text(fresh)
    assert reused.cost == 0.0 and reused.calls == 0
    assert reused.reused_calls == spent.calls
    assert reused.report_cost == pytest.approx(spent.report_cost)


def test_cached_answers_skip_the_budget_check(tmp_path: Path) -> None:
    runs = [load_fixture("researcher_repeat_search")]
    rules = run_rule_checks(runs)
    config = _config()
    fake = FakeLLM(triage=always(True), confirm=lambda p, c: verdict(True))
    client = with_model_fallback(CachingClient(fake, tmp_path), config.llm)
    judge = _judge(config, client)
    judge.run(judge.plan(runs, rules), rules)
    calls = len(fake.calls)
    broke = Spend(prices=config.prices, max_cost=0.0)  # no money at all
    judge = _judge(config, client, broke)
    out = judge.run(judge.plan(runs, rules), rules)
    assert out.stopped_for_budget is None
    assert len(fake.calls) == calls


def test_dedupe_ignores_completion_order() -> None:
    def f(conf: float, step: str) -> Finding:
        return Finding(
            check_id="design.lost_context",
            group="design",
            method="judge",
            severity="medium",
            run_id="r1",
            key="agent",
            evidence_step_ids=[step],
            explanation=step,
            confidence=conf,
        )

    findings = [f(0.8, "s2"), f(0.8, "s1"), f(0.7, "s0")]
    assert _dedupe(findings) == _dedupe(list(reversed(findings)))
    assert _dedupe(findings)[0].evidence_step_ids == ["s1"]


def test_finding_goes_to_the_agent_the_judge_names() -> None:
    run = load_fixture("supervisor_no_end")
    agents = sorted({s.node for s in run.steps if s.node and s.kind == "llm"})
    assert len(agents) > 1
    other = agents[-1]

    named = verdict(True, evidence=["S1", "S2", "S3"], agent=other)
    judge, fake = _flagging_judge([named])
    (finding,) = judge.run(judge.plan([run], []), []).findings
    assert finding.node == other  # not the agent of the first cited step

    unknown = verdict(True, evidence=["S1"], agent="no_such_agent")
    judge, fake = _flagging_judge([unknown, unknown])
    (finding,) = judge.run(judge.plan([run], []), []).findings
    assert "agent must be one of" in fake.prompts[-1]  # sent back once
    s1 = run_end_window(run, Config().judge.window_steps).steps[0]
    assert finding.node == s1.node  # then the old rule: the first cited step's agent


def test_named_agent_is_shown_at_its_own_last_step_not_a_handoff() -> None:
    run = load_fixture("supervisor_no_end")
    window = run_end_window(run, Config().judge.window_steps)
    labels = {s.id: lbl for lbl, s in zip(window.labels, window.steps, strict=True)}
    handoff = next((s for s in window.steps if s.kind == "handoff"), None)
    if handoff is None:
        pytest.skip("no handoff in the window")
    receiver = handoff.handoff_to
    own_calls = [s for s in window.steps if s.node == receiver and s.kind == "llm"]
    cited = [labels[s.id] for s in own_calls] + [labels[handoff.id]]  # handoff cited last
    judge, _ = _flagging_judge([verdict(True, evidence=cited, agent=receiver)])
    (finding,) = judge.run(judge.plan([run], []), []).findings
    assert finding.node == receiver
    shown = finding.evidence_step_ids
    assert handoff.id in shown and own_calls[-1].id in shown


def test_where_shows_the_agents_own_call_for_judge_findings() -> None:
    from tracedoc.report.view import _anchor

    run = load_fixture("supervisor_no_end")
    handoff = run.steps_of("handoff")[0]
    receiver = handoff.handoff_to
    call = next(s for s in run.steps_of("llm") if s.node == receiver)
    f = Finding(
        check_id="verification.premature_finish",
        group="verification",
        method="judge",
        severity="high",
        run_id=run.id,
        node=receiver,
        explanation="x",
        evidence_step_ids=[handoff.id, call.id],
    )  # handoff cited first
    assert _anchor(f, run).id == call.id
