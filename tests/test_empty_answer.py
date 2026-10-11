"""Evidence for an empty final answer: the last real answer any agent wrote in the run before
the empty reply, shown as "Answer that never reached the user"."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfReader

from tests.builders import ai, build_run, step
from tests.helpers import SOURCES, load_fixture
from tracedoc.analyze import analyze
from tracedoc.checks.rules.empty_answer import EmptyAnswer, lost_answer
from tracedoc.config import Config
from tracedoc.report import write_reports


def _lost(view) -> str | None:
    (f,) = [f for f in view.findings if f.finding.check_id == "design.empty_answer"]
    return f.lost_answer


@pytest.mark.parametrize("source", SOURCES)
def test_supervisor_drops_the_research_agents_answer(source: str, tmp_path: Path) -> None:
    """The fixture: research_agent answers, then the supervisor's final reply is empty."""
    run = load_fixture("nested_agents", source)
    (finding,) = EmptyAnswer(Config()).run([run])
    assert finding.evidence_excerpts[-1] == (
        "Answer that never reached the user (research_agent): Apple had 164,000 employees in 2024."
    )
    files = write_reports(analyze([run], Config()), [run], tmp_path)
    assert _lost(files.view) == "“Apple had 164,000 employees in 2024.” (research_agent, reply 2)"
    details = " ".join(
        " ".join((p.extract_text() or "").split()) for p in PdfReader(files.details).pages
    )
    assert (
        "Answer that never reached the user “Apple had 164,000 employees in 2024.” "
        "(research_agent, reply 2)"
    ) in details


def _run(*replies: tuple[str, dict]) -> object:
    specs = []
    for node, output in replies:
        specs += [step("node", node), step("llm", node, output=output)]
    return build_run("r1", specs, status="no_answer", input_preview="Summarize Q3 sales.")


def test_routing_words_tool_calls_and_punctuation_are_not_answers() -> None:
    run = _run(
        ("analyst", ai("Q3 sales rose 12% to $4.1M.")),
        ("supervisor", ai("writer")),  # routing: an agent's name
        (
            "writer",
            {"type": "ai", "content": "Let me check.", "tool_calls": [{"name": "lookup"}]},
        ),
        ("writer", ai("...")),
        ("supervisor", ai("FINISH")),
        ("supervisor", ai("")),
    )
    found = lost_answer(run, run.steps_of("llm")[-1])
    assert found is not None and found.node == "analyst"


def test_an_answer_that_hands_off_still_counts() -> None:
    run = _run(
        (
            "writer",
            {
                "type": "ai",
                "content": "Q3 sales rose 12%.",
                "tool_calls": [{"name": "transfer_back_to_supervisor"}],
            },
        ),
        ("supervisor", ai("")),
    )
    found = lost_answer(run, run.steps_of("llm")[-1])
    assert found is not None and found.node == "writer"


def test_no_earlier_answer_says_so(tmp_path: Path) -> None:
    run = _run(("supervisor", ai("writer")), ("writer", ai("")))
    (finding,) = EmptyAnswer(Config()).run([run])
    assert len(finding.evidence_step_ids) == 1
    files = write_reports(analyze([run], Config()), [run], tmp_path)
    assert _lost(files.view) == "No agent wrote an answer in this run before the empty reply."


def test_other_findings_have_no_lost_answer() -> None:
    from tracedoc.report.view import build_view

    run = load_fixture("tool_retry_same_args")
    view = build_view(analyze([run], Config()), [run])
    assert view.findings and all(f.lost_answer is None for f in view.findings)
