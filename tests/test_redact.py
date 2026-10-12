"""Configured fields and patterns never appear in normalized runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tests.fixtures.agents._fake import ScriptedChatModel, ai
from tests.helpers import SOURCES, raw_fixture
from tracedoc import TraceCapture
from tracedoc.config import Config, RedactConfig
from tracedoc.ingest.jsonl import read_capture_file
from tracedoc.normalize import normalize
from tracedoc.redact import Redactor

FIELD_SECRET = "MARKER-FIELD-7f3a9c"
CUSTOM_FIELD_SECRET = "MARKER-CUSTOMER-Zelda-Q"
PATTERN_SECRET = "ACME-SECRET-55501"
EMAIL = "jane.doe.marker@example.com"
API_KEY = "sk-markerKEY0123456789abcdef"
MARKERS = [FIELD_SECRET, CUSTOM_FIELD_SECRET, PATTERN_SECRET, EMAIL, API_KEY]


def _config() -> Config:
    redact = RedactConfig()
    redact.fields.append("customer_name")
    redact.patterns.append(r"ACME-SECRET-\d+")
    return Config(redact=redact)


def test_field_names_match_case_insensitively_and_by_suffix() -> None:
    r = Redactor(RedactConfig(fields=["api_key"], patterns=[]))
    assert r({"API_KEY": "x", "openai_api_key": "y", "x-api_key": "z", "keep": "k"}) == {
        "API_KEY": "[REDACTED]",
        "openai_api_key": "[REDACTED]",
        "x-api_key": "[REDACTED]",
        "keep": "k",
    }


def test_does_not_redact_lookalike_keys() -> None:
    r = Redactor(RedactConfig(fields=["token"], patterns=[]))
    assert r({"tokens_in": 5, "max_tokens": 10}) == {"tokens_in": 5, "max_tokens": 10}


def test_patterns_apply_inside_nested_strings() -> None:
    r = Redactor(RedactConfig(fields=[], patterns=[r"\d{3}-\d{4}"]))
    assert r({"a": ["call 555-1234 now", {"b": ("555-9999",)}]}) == {
        "a": ["call [REDACTED] now", {"b": ["[REDACTED]"]}]
    }


def test_default_patterns() -> None:
    r = Redactor(RedactConfig())
    text = r(f"mail {EMAIL} key {API_KEY} auth Bearer abc.def-123")
    assert EMAIL not in text and API_KEY not in text and "abc.def-123" not in text


def _run_secret_agent(tmp_path: Path) -> Path:
    @tool
    def crm_lookup(customer_name: str, api_key: str) -> str:
        """Look up a customer."""
        return f"Customer {customer_name} ({EMAIL}) has ticket {PATTERN_SECRET}."

    args = {"customer_name": CUSTOM_FIELD_SECRET, "api_key": FIELD_SECRET}
    model = ScriptedChatModel(
        responses=[
            ai(tool_calls=[{"name": "crm_lookup", "args": args, "id": "c1"}]),
            ai(f"Found {PATTERN_SECRET} for {EMAIL}; used key {API_KEY}."),
        ]
    )

    def agent(state: MessagesState) -> dict[str, Any]:
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([crm_lookup]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    capture = TraceCapture(tmp_path)
    graph.compile().invoke(
        {"messages": [HumanMessage(f"I'm {EMAIL}, my key is {API_KEY}. Check {PATTERN_SECRET}")]},
        config={"callbacks": [capture]},
    )
    (path,) = capture.files_written
    return path


def test_markers_never_appear_downstream(tmp_path: Path) -> None:
    path = _run_secret_agent(tmp_path)
    raw_text = path.read_text()
    assert all(m in raw_text for m in MARKERS), "precondition: the raw capture holds the secrets"

    run = normalize(read_capture_file(path), _config())
    dumped = run.model_dump_json()
    for marker in MARKERS:
        assert marker not in dumped
    assert "[REDACTED]" in run.input_preview
    (tool_step,) = run.steps_of("tool")
    assert tool_step.input == {"customer_name": "[REDACTED]", "api_key": "[REDACTED]"}


@pytest.mark.parametrize("source", SOURCES)
def test_redaction_applies_to_imported_sources(source: str) -> None:
    config = Config(redact=RedactConfig(fields=["order_id"], patterns=[r"A-1042"]))
    run = normalize(raw_fixture("tool_output_unused", source), config)
    assert "A-1042" not in run.model_dump_json()


def test_redaction_applies_to_errors() -> None:
    config = Config(redact=RedactConfig(fields=[], patterns=[r"example\.com"]))
    run = normalize(raw_fixture("tool_retry_same_args", "capture"), config)
    assert "example.com" not in run.model_dump_json()


def test_learned_values_are_scrubbed_everywhere() -> None:
    r = Redactor(RedactConfig(fields=["password"], patterns=[]))
    r.learn({"login": {"password": "hunter22"}})
    assert r({"log": "tried hunter22 twice", "password": "hunter22"}) == {
        "log": "tried [REDACTED] twice",
        "password": "[REDACTED]",
    }


def test_short_values_are_not_learned() -> None:
    r = Redactor(RedactConfig(fields=["pin"], patterns=[]))
    r.learn({"pin": "42"})
    assert r("answer is 42") == "answer is 42"


def test_only_luhn_valid_digit_runs_count_as_card_numbers() -> None:
    from tracedoc.config import RedactConfig
    from tracedoc.redact import Redactor, luhn_valid

    redact = Redactor(RedactConfig())
    # Years in a search query are not a card number: kept.
    query = "Amazon employee count 2020 2021 2022 2023 headcount history"
    assert redact.text(query) == query
    assert redact.text("order 1234567890123 shipped") == "order 1234567890123 shipped"
    # Real card numbers (they pass the Luhn checksum) are redacted, however they are grouped,
    # and the text around them keeps its spacing.
    assert redact.text("card 4111 1111 1111 1111 on file") == "card [REDACTED] on file"
    assert redact.text("card 4111-1111-1111-1111.") == "card [REDACTED]."
    assert redact.text("5500000000000004") == "[REDACTED]"
    assert luhn_valid("4111111111111111") and not luhn_valid("2020202120222023")
