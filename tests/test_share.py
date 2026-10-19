"""Data sharing, client side: the payload allow-list, the marker test, consent, and the flag."""

from __future__ import annotations

import contextlib
import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import ValidationError
from typer.testing import CliRunner

from tests.fixtures.agents import AGENTS
from tests.fixtures.agents._fake import ScriptedChatModel, ai
from tests.fixtures.generate_traces import CollectingTracer
from tests.helpers import TRACES, load_fixture
from tracedoc import TraceCapture, cli
from tracedoc.analyze import analyze
from tracedoc.config import Config, ShareConfig
from tracedoc.ingest.jsonl import read_capture_file
from tracedoc.ingest.langsmith import traces_from_runs
from tracedoc.normalize import normalize
from tracedoc.report import all_findings
from tracedoc.share import consent
from tracedoc.share.payload import (
    Contribution,
    build_contribution,
    public_model_name,
    serialize,
)
from tracedoc.share.upload import ShareError, check_endpoint, last_payload_path

runner = CliRunner()
HASH = "a" * 64  # a well-formed deletion-key hash for payloads built directly

# A unique marker in every content field a trace can carry.
M = {
    "run_name": "MKR01runname",
    "agent_node": "MKR02agentnode_researcher",
    "tool_node": "MKR03toolnode",
    "second_node": "MKR04secondnode_writer",
    "tool_name": "mkr05toolname",
    "tool_name_2": "mkr06failingtool",
    "system": "MKR07systemprompt",
    "user": "MKR08userinput",
    "arg": "MKR09toolarg",
    "path": "/srv/MKR10filepath/data.csv",
    "key": "MKR11keylike-9f8e7d6c5b4a",
    "result": "MKR12toolresult",
    "error": "MKR13errormessage",
    "ai": "MKR14modeloutput",
    "code": "def MKR15code(): pass",
    "model": "MKR16-custom-model",
    "tag": "MKR17tag",
    "metadata": "MKR18metadata",
    "framework_version": "MKR19version",
    "writer_out": "MKR20writeroutput",
}


def _marker_agent() -> Any:
    @tool(M["tool_name"])
    def search(query: str, path: str, secret: str) -> str:
        """Search."""
        return f"{M['result']} for {query} {path}"

    @tool(M["tool_name_2"])
    def failing(query: str) -> str:
        """Fails."""
        raise RuntimeError(M["error"])

    args = {"query": M["arg"], "path": M["path"], "secret": M["key"]}
    model = ScriptedChatModel(
        model_name=M["model"],
        responses=[
            ai(tool_calls=[{"name": M["tool_name"], "args": args, "id": "c1"}]),
            ai(tool_calls=[{"name": M["tool_name"], "args": args, "id": "c2"}]),
            ai(tool_calls=[{"name": M["tool_name"], "args": args, "id": "c3"}]),
            ai(tool_calls=[{"name": M["tool_name_2"], "args": {"query": M["arg"]}, "id": "c4"}]),
            ai(f"{M['ai']} {M['code']}"),
        ],
    )
    writer = ScriptedChatModel(model_name=M["model"], responses=[ai(M["writer_out"])])

    def agent(state: MessagesState) -> dict[str, Any]:
        return {"messages": [model.invoke([SystemMessage(M["system"]), *state["messages"]])]}

    def second(state: MessagesState) -> dict[str, Any]:
        return {"messages": [writer.invoke([SystemMessage(M["system"]), *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node(M["agent_node"], agent)
    graph.add_node(M["tool_node"], ToolNode([search, failing], handle_tool_errors=True))
    graph.add_node(M["second_node"], second)
    graph.add_edge(START, M["agent_node"])
    graph.add_conditional_edges(
        M["agent_node"], tools_condition, {"tools": M["tool_node"], END: M["second_node"]}
    )
    graph.add_edge(M["tool_node"], M["agent_node"])
    graph.add_edge(M["second_node"], END)
    return graph.compile()


def _invoke(callbacks: list) -> None:
    config = {
        "callbacks": callbacks,
        "run_name": M["run_name"],
        "tags": [M["tag"]],
        "metadata": {"note": M["metadata"]},
    }
    with contextlib.suppress(GraphRecursionError):
        _marker_agent().invoke({"messages": [HumanMessage(f"{M['user']} {M['path']}")]}, config)


def _marker_runs(tmp_path: Path) -> tuple[list, str]:
    """Normalized runs from both sources, plus the raw inputs tracedoc read."""
    capture = TraceCapture(tmp_path)
    tracer = CollectingTracer()
    _invoke([capture, tracer])
    (path,) = capture.files_written
    # Also plant a marker in the header's framework version.
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header["framework_version"] = M["framework_version"]
    path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    exported = tracer.export()
    (ls_trace,) = traces_from_runs(exported)
    ls_trace.framework_version = M["framework_version"]
    raw = path.read_text() + json.dumps(exported)
    return [normalize(read_capture_file(path)), normalize(ls_trace)], raw


def test_no_marker_reaches_the_serialized_payload(tmp_path: Path) -> None:
    runs, raw = _marker_runs(tmp_path)
    for name, marker in M.items():
        assert marker in raw, f"precondition: marker {name} should be in the raw trace input"

    report = analyze(runs, Config(), client=None)
    findings = all_findings(report)
    assert {f.check_id for f in findings} >= {"design.loop", "tools.errors_retries"}
    assert any(f.key == M["tool_name"] for f in findings)  # names exist in findings...
    votes = [("design.loop", "up"), (M["user"], "down")]
    payload = serialize(build_contribution(uuid.uuid4(), HASH, runs, findings, votes))

    text = payload.decode()
    for name, marker in M.items():
        assert marker.lower() not in text.lower(), f"{name} marker leaked into the payload"
    # ...but only fixed identifiers made it out.
    data = json.loads(payload)
    assert data["runs"][0]["findings"]
    assert data["runs"][0]["models_by_role"][0]["models"] == ["custom"]
    assert data["runs"][0]["framework_version"] is None
    assert data["feedback"] == [{"check_id": "design.loop", "vote": "up"}]


ALLOWED_KEYS = {
    "schema_version", "install_id", "deletion_key_hash", "tool_version", "runs", "feedback",
    "task_type", "framework", "framework_version", "models_by_role", "role", "models",
    "graph", "agent_count", "tool_count", "max_depth", "status", "step_count", "tokens_in",
    "tokens_out", "cost", "duration_s", "findings", "check_id", "severity", "vote",
}  # fmt: skip


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def test_payload_has_only_allow_listed_fields() -> None:
    runs = [load_fixture(n) for n in AGENTS]
    report = analyze(runs, Config())
    data = json.loads(serialize(build_contribution(uuid.uuid4(), HASH, runs, all_findings(report))))
    assert _keys(data) <= ALLOWED_KEYS
    assert len(data["runs"]) == 10
    shapes = {(r["graph"]["agent_count"], r["graph"]["tool_count"]) for r in data["runs"]}
    assert (3, 1) in shapes  # healthy: researcher, writer, critic; one tool


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(prompt="hello"),
        lambda d: d["runs"][0].update(input_preview="hello"),
        lambda d: d["runs"][0]["findings"].append(
            {"check_id": "x.made_up", "severity": "low", "role": None}
        ),
        lambda d: d["runs"][0]["findings"].append(
            {"check_id": "design.loop", "severity": "low", "role": None, "key": "my_tool"}
        ),
        lambda d: d["runs"][0].update(task_type="my secret project"),
        lambda d: d["runs"][0]["models_by_role"].append(
            {"role": "writer", "models": ["ft:gpt-4o:acme:x"]}
        ),
        lambda d: d.update(install_id=str(uuid.uuid1())),
        lambda d: d.update(deletion_key_hash="the-token-itself"),
        lambda d: d.update(deletion_token="abc"),
        lambda d: d["runs"][0].update(framework_version="/Users/someone/venv"),
    ],
)
def test_schema_rejects_anything_outside_the_allow_list(mutate: Any) -> None:
    runs = [load_fixture("researcher_repeat_search")]
    report = analyze(runs, Config())
    data = json.loads(serialize(build_contribution(uuid.uuid4(), HASH, runs, all_findings(report))))
    Contribution.model_validate(data)
    mutate(data)
    with pytest.raises(ValidationError):
        Contribution.model_validate(data)


def test_public_model_names() -> None:
    assert public_model_name("claude-opus-5") == "claude-opus-5"
    assert public_model_name("anthropic.claude-haiku-4-5") == "claude-haiku-4-5"
    assert public_model_name("gpt-4o-mini") == "gpt-4o-mini"
    for custom in ("ft:gpt-4o:acme:abc", "llama-acme-internal", "claude-opus-5-acmecorp", "x"):
        assert public_model_name(custom) == "custom"
    assert public_model_name(None) == "unknown"


# ---- install ID and consent --------------------------------------------------------------


def test_install_id_is_random_uuid4_and_persistent() -> None:
    first = consent.load_state()
    assert first.install_id.version == 4
    assert consent.load_state().install_id == first.install_id
    other = consent.load_state(consent.state_path().parent / "other.json")
    assert other.install_id != first.install_id
    assert first.choice is None and first.needs_asking


def test_choice_terms_version_and_timestamp_are_stored() -> None:
    state = consent.record_choice(consent.load_state(), "yes")
    saved = json.loads(consent.state_path().read_text())
    assert saved["choice"] == "yes"
    assert saved["terms_version"] == consent.TERMS_VERSION
    assert saved["decided_at"]
    assert state.sharing and not state.needs_asking


def test_new_terms_version_asks_again(monkeypatch: pytest.MonkeyPatch) -> None:
    consent.record_choice(consent.load_state(), "yes")
    monkeypatch.setattr(consent, "TERMS_VERSION", consent.TERMS_VERSION + 1)
    state = consent.load_state()
    assert state.needs_asking and not state.sharing


# ---- the feature flag, preview, and the CLI ---------------------------------------------------

CAPTURES = str(TRACES / "capture")


def _write_config(tmp_path: Path, enabled: bool, endpoint: str = "") -> Path:
    path = tmp_path / "c.toml"
    path.write_text(f'[share]\nenabled = {str(enabled).lower()}\nendpoint = "{endpoint}"\n')
    return path


def test_flag_off_by_default() -> None:
    assert ShareConfig().enabled is False
    assert ShareConfig().endpoint == ""
    assert Config().share.enabled is False


def test_flag_off_never_prompts_or_sends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    sent: list[bytes] = []
    monkeypatch.setattr(cli, "upload", lambda payload, config: sent.append(payload))
    result = runner.invoke(cli.app, ["analyze", CAPTURES, "--no-judge", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Share anonymized statistics?" not in result.output
    assert not sent
    assert consent.load_state().choice is None
    assert last_payload_path().is_file()  # preview still works for review


def test_preview_prints_the_saved_payload_bytes(tmp_path: Path) -> None:
    runner.invoke(cli.app, ["analyze", CAPTURES, "--no-judge", "--out", str(tmp_path)])
    result = runner.invoke(cli.app, ["share", "--preview"])
    assert result.exit_code == 0
    assert result.stdout_bytes == last_payload_path().read_bytes()
    Contribution.model_validate_json(result.stdout_bytes)


def test_share_on_and_delete_refused_while_flag_off() -> None:
    for flag in ("--on", "--delete"):
        result = runner.invoke(cli.app, ["share", flag])
        assert result.exit_code == 2
        assert result.stderr.strip() == (
            "Error: data sharing is not yet enabled in this version of tracedoc, so nothing is sent"
        )


def test_share_off_always_works() -> None:
    result = runner.invoke(cli.app, ["share", "--off"])
    assert result.exit_code == 0
    assert consent.load_state().choice == "no"
    assert "not yet enabled" in result.stdout


def test_share_says_it_is_not_enabled_yet(tmp_path: Path) -> None:
    help_text = " ".join(runner.invoke(cli.app, ["share", "--help"]).stdout.split())
    assert "NOT YET ENABLED in this version, so nothing is sent" in help_text
    runner.invoke(cli.app, ["analyze", CAPTURES, "--no-judge", "--out", str(tmp_path)])
    preview = runner.invoke(cli.app, ["share", "--preview"])
    assert "not yet enabled" in preview.stderr  # the payload itself stays on stdout, unchanged
    assert preview.stdout_bytes == last_payload_path().read_bytes()


def test_terms_are_marked_as_a_draft() -> None:
    assert consent.TERMS.startswith("DRAFT: these terms are not in effect.")


def test_share_needs_exactly_one_option() -> None:
    assert runner.invoke(cli.app, ["share"]).exit_code == 2
    assert runner.invoke(cli.app, ["share", "--on", "--off"]).exit_code == 2


def test_consent_prompt_enter_means_no(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    sent: list[bytes] = []
    monkeypatch.setattr(cli, "upload", lambda payload, config: sent.append(payload))
    config = _write_config(tmp_path, True, "https://stats.example.invalid")
    args = ["analyze", CAPTURES, "--no-judge", "--out", str(tmp_path), "--config", str(config)]
    result = runner.invoke(cli.app, args, input="\n")
    assert result.exit_code == 0, result.output
    assert "What is never sent" in result.output
    assert consent.load_state().choice == "no"
    assert not sent
    # Asked once: the next run does not ask again.
    again = runner.invoke(cli.app, args, input="\n")
    assert "Share anonymized statistics?" not in again.output


def test_consent_yes_uploads_exactly_the_preview_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    sent: list[bytes] = []
    monkeypatch.setattr(cli, "upload", lambda payload, config: sent.append(payload))
    config = _write_config(tmp_path, True, "https://stats.example.invalid")
    args = ["analyze", CAPTURES, "--no-judge", "--out", str(tmp_path), "--config", str(config)]
    assert runner.invoke(cli.app, args, input="y\n").exit_code == 0
    preview = runner.invoke(cli.app, ["share", "--preview"]).stdout_bytes
    assert sent == [preview]


def test_non_interactive_runs_are_never_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[bytes] = []
    monkeypatch.setattr(cli, "upload", lambda payload, config: sent.append(payload))
    config = _write_config(tmp_path, True, "https://stats.example.invalid")
    args = ["analyze", CAPTURES, "--no-judge", "--out", str(tmp_path), "--config", str(config)]
    assert runner.invoke(cli.app, args).exit_code == 0
    assert consent.load_state().choice is None
    assert not sent


@pytest.mark.parametrize(
    ("enabled", "endpoint", "ok"),
    [
        (False, "https://x.example", False),
        (True, "", False),
        (True, "http://stats.example.com", False),
        (True, "ftp://stats.example.com", False),
        (True, "https://stats.example.com/", True),
        (True, "http://127.0.0.1:8000", True),
        (True, "http://localhost:8000", True),
    ],
)
def test_endpoint_rules(enabled: bool, endpoint: str, ok: bool) -> None:
    config = ShareConfig(enabled=enabled, endpoint=endpoint)
    if ok:
        assert check_endpoint(config).startswith(("https://", "http://"))
    else:
        with pytest.raises(ShareError):
            check_endpoint(config)


# ---- the deletion token -----------------------------------------------------------------------


def test_deletion_token_created_at_install_and_stored_owner_only() -> None:
    import os
    import stat

    state = consent.load_state()
    assert len(state.deletion_token) >= 43  # 32 random bytes, url-safe base64
    assert state.deletion_key_hash == consent.deletion_key_hash(state.deletion_token)
    saved = json.loads(consent.state_path().read_text())
    assert saved["deletion_token"] == state.deletion_token
    assert saved["install_id"] == str(state.install_id)
    if os.name == "posix":
        assert stat.S_IMODE(consent.state_path().stat().st_mode) == 0o600
    assert consent.load_state().deletion_token == state.deletion_token  # persistent
    other = consent.load_state(consent.state_path().parent / "other.json")
    assert other.deletion_token != state.deletion_token


def test_state_from_before_tokens_gets_one() -> None:
    path = consent.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"install_id": str(uuid.uuid4()), "choice": "no"}))
    state = consent.load_state()
    assert state.deletion_token
    assert json.loads(path.read_text())["deletion_token"] == state.deletion_token


def test_rotation_replaces_both_id_and_token() -> None:
    state = consent.load_state()
    old_id, old_token = state.install_id, state.deletion_token
    consent.rotate_install_id(state)
    fresh = consent.load_state()
    assert fresh.install_id != old_id and fresh.deletion_token != old_token


def test_token_never_leaves_in_payload_preview_or_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracedoc.share.upload import upload_log_path

    monkeypatch.setattr(cli, "_interactive", lambda: True)
    sent: list[bytes] = []
    monkeypatch.setattr(cli, "upload", lambda payload, config: sent.append(payload))
    config = _write_config(tmp_path, True, "https://stats.example.invalid")
    args = ["analyze", CAPTURES, "--no-judge", "--out", str(tmp_path), "--config", str(config)]
    assert runner.invoke(cli.app, args, input="y\n").exit_code == 0
    state = consent.load_state()
    preview = runner.invoke(cli.app, ["share", "--preview"]).stdout_bytes
    assert state.deletion_token.encode() not in preview
    assert json.loads(preview)["deletion_key_hash"] == state.deletion_key_hash
    assert sent == [preview]
    assert (
        not upload_log_path().exists() or state.deletion_token not in upload_log_path().read_text()
    )
