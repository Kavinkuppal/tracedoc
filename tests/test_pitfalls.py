"""Known framework pitfalls: one catalog, one entry per pitfall. First entry: a
langgraph-supervisor system on non-OpenAI models with include_agent_name at its default."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pypdf import PdfReader

from tests.fixtures.agents import supervisor_agent_name
from tests.helpers import SOURCES, load_fixture
from tracedoc import TraceCapture
from tracedoc.analyze import analyze
from tracedoc.catalog import check_entry
from tracedoc.checks.base import REGISTRY, run_rule_checks
from tracedoc.checks.pitfalls import PITFALLS
from tracedoc.config import Config
from tracedoc.ingest.jsonl import read_capture_file
from tracedoc.normalize import normalize
from tracedoc.report import write_reports
from tracedoc.report.view import confidence_level, is_serious

PITFALL = "framework.langgraph_supervisor.include_agent_name"


def _pitfall_findings(run):
    return [f for f in run_rule_checks([run]) if f.check_id == PITFALL]


@pytest.mark.parametrize("source", SOURCES)
def test_supervisor_on_claude_without_include_agent_name_is_serious(source: str) -> None:
    run = load_fixture("supervisor_agent_name", source)
    (f,) = _pitfall_findings(run)
    assert f.severity == "high" and confidence_level(f) == "high" and is_serious(f)
    assert f.node == "supervisor"
    assert "claude-sonnet-5-5" in f.explanation and "research_expert" in f.explanation
    assert f.suggested_fix.startswith('Pass include_agent_name="inline" to create_supervisor')


def test_the_fix_cites_the_librarys_documentation(tmp_path: Path) -> None:
    run = load_fixture("supervisor_agent_name")
    files = write_reports(analyze([run], Config()), [run], tmp_path)
    (card,) = [c for c in files.view.cards if c.title == check_entry(PITFALL)["title"]]
    assert card.what_to_change.startswith('Pass include_agent_name="inline"')
    details = " ".join(
        " ".join((p.extract_text() or "").split()) for p in PdfReader(files.details).pages
    )
    assert (
        'Documentation langgraph-supervisor, create_supervisor(include_agent_name): "None: '
        "Relies on the LLM provider using the name attribute on the AI message. Currently, "
        "only OpenAI supports this."
    ) in details
    assert "https://github.com/langchain-ai/langgraph-supervisor-py" in details


def _captured(include_agent_name: str | None):
    with tempfile.TemporaryDirectory() as tmp:
        capture = TraceCapture(tmp)
        supervisor_agent_name.build(include_agent_name).invoke(
            supervisor_agent_name.INPUT,
            config={"callbacks": [capture], "run_name": "supervisor_agent_name"},
        )
        (written,) = capture.files_written
        return normalize(read_capture_file(written))


def test_inline_agent_names_are_not_flagged() -> None:
    assert _pitfall_findings(_captured(None))  # the same system, default setting: flagged
    assert _pitfall_findings(_captured("inline")) == []


@pytest.mark.parametrize("model", ["gpt-5", "openai/gpt-4.1-mini", "o4-mini", None])
def test_openai_or_unknown_supervisor_models_are_not_flagged(model: str | None) -> None:
    run = load_fixture("supervisor_agent_name")
    for step in run.steps_of("llm"):
        step.model = model
    assert _pitfall_findings(run) == []


def test_other_systems_are_not_flagged() -> None:
    for name in ("nested_agents", "healthy", "swarm_parallel_handoff", "supervisor_no_end"):
        assert _pitfall_findings(load_fixture(name)) == [], name


def test_each_pitfall_is_one_entry_that_becomes_a_check() -> None:
    assert PITFALLS
    for p in PITFALLS:
        assert p.id.startswith("framework.")
        assert REGISTRY[p.id].default_severity == p.severity  # registered as a rule check
        entry = check_entry(p.id)  # its reviewed text is in the catalog
        for key in ("title", "looks_for", "why", "what_happened", "what_to_change", "docs"):
            assert entry[key], (p.id, key)
        assert "https://" in p.docs
