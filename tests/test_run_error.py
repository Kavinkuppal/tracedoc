"""design.run_error: a run that ends in an error is a serious problem at the failing step."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pypdf import PdfReader

from tests.helpers import SOURCES, load_fixture
from tracedoc.checks.rules.run_error import RunError, error_summary, failed_step
from tracedoc.config import Config
from tracedoc.report.view import build_view, confidence_level, is_serious
from tracedoc.schema import Report, Run, Step, Totals

T0 = datetime(2026, 10, 2, 12, tzinfo=timezone.utc)
API_400 = (
    "AnthropicInvalidRequestError: Error code: 400 - {'type': 'error', 'error': {'type': "
    "'invalid_request_error', 'message': 'messages.4: `tool_use` ids were found without "
    "`tool_result` blocks immediately after: toolu_1. Each `tool_use` block must have a "
    "corresponding `tool_result` block in the next message.'}}"
)


def _step(i: int, kind: str, node: str, **kw: object) -> Step:
    return Step(
        id=f"s{i}",
        run_id="r1",
        kind=kind,  # type: ignore[arg-type]
        name=str(kw.pop("name", kind)),
        node=node,
        started_at=T0 + timedelta(seconds=i),
        ended_at=T0 + timedelta(seconds=i + 1),
        **kw,  # type: ignore[arg-type]
    )


def _run(steps: list[Step], status: str = "error", error: str | None = None) -> Run:
    return Run(
        id="r1",
        source="capture",
        status=status,  # type: ignore[arg-type]
        error=error,
        input_preview="Book a flight and a hotel.",
        started_at=T0,
        ended_at=T0 + timedelta(seconds=len(steps) + 1),
        steps=steps,
        totals=Totals(cost=0.01, duration_s=9.0, step_count=3),
    )


def _parallel_handoff_run() -> Run:
    return _run(
        [
            _step(
                0,
                "llm",
                "flight_assistant",
                output={
                    "type": "ai",
                    "content": "Booking now and transferring you.",
                    "tool_calls": [
                        {"name": "book_flight", "args": {"flight_id": "1"}},
                        {"name": "transfer_to_hotel_assistant", "args": {}},
                    ],
                },
            ),
            _step(1, "node", "flight_assistant", name="tools", error="ParentCommand: Command(...)"),
            _step(2, "tool", "flight_assistant", name="book_flight", output="Booked."),
            _step(3, "node", "hotel_assistant", name="model", error=API_400),
            _step(4, "llm", "hotel_assistant", error=API_400),
        ],
        error=API_400,
    )


def test_error_run_is_a_serious_finding_at_the_failing_agent() -> None:
    (finding,) = RunError(Config()).run([_parallel_handoff_run()])
    assert finding.check_id == "design.run_error"
    assert finding.severity == "high"
    assert confidence_level(finding) == "high" and is_serious(finding)
    assert finding.node == "hotel_assistant"  # the agent whose step failed
    assert any(
        e.startswith("Error: AnthropicInvalidRequestError") for e in finding.evidence_excerpts
    )


def test_parallel_handoff_is_recognized_from_the_trace() -> None:
    (finding,) = RunError(Config()).run([_parallel_handoff_run()])
    assert "flight_assistant called book_flight and transfer_to_hotel_assistant" in (
        finding.explanation
    )
    assert "parallel_tool_calls=False" in finding.suggested_fix
    assert "flight_assistant's model" in finding.suggested_fix
    assert finding.evidence_step_ids == ["s0", "s4"]


def test_handoff_control_flow_is_not_the_failure() -> None:
    run = _parallel_handoff_run()
    assert failed_step(run).id == "s4"  # type: ignore[union-attr]
    only_control_flow = _run(
        [_step(0, "node", "a", error="ParentCommand: Command(graph='__parent__')")],
        error="RuntimeError: graph failed",
    )
    (finding,) = RunError(Config()).run([only_control_flow])
    assert finding.node is None
    assert finding.explanation == "the run ended with an error: RuntimeError: graph failed"


def test_failing_tool_variant() -> None:
    run = _run([_step(0, "tool", "researcher", name="web_search", error="TimeoutError: 30 s")])
    (finding,) = RunError(Config()).run([run])
    assert finding.node == "researcher"
    assert finding.explanation.startswith("web_search failed for researcher")
    assert "web_search return an error message" in finding.suggested_fix


def test_runs_without_an_error_are_not_flagged() -> None:
    for status in ("completed", "no_answer", "recursion_limit", "interrupted"):
        assert RunError(Config()).run([_run([_step(0, "llm", "a")], status=status)]) == []


def test_error_summary_keeps_class_and_message() -> None:
    assert error_summary(API_400).startswith(
        "AnthropicInvalidRequestError: messages.4: `tool_use` ids were found"
    )
    assert error_summary("ValueError: bad input\nTraceback ...") == "ValueError: bad input"


def test_error_run_counts_as_serious_in_the_report() -> None:
    run = _parallel_handoff_run()
    findings = RunError(Config()).run([run])
    from tracedoc.group import group_findings

    view = build_view(Report(groups=group_findings(findings, [run])), [run])
    assert view.tiles[0].value == "1 (100%)"
    hotel = next(a for a in view.agents if a.name == "hotel_assistant")
    assert hotel.serious_runs == 1


# ---- the swarm fixture: a real LangGraph run, driven by fake models, that ends in an error ----


@pytest.mark.parametrize("source", SOURCES)
def test_fixture_run_that_ends_in_an_error(source: str) -> None:
    run = load_fixture("swarm_parallel_handoff", source)
    assert run.status == "error"
    (finding,) = RunError(Config()).run([run])
    assert finding.severity == "high" and confidence_level(finding) == "high"
    assert finding.node == "hotel_assistant"  # its model call is the one that failed
    assert finding.explanation.startswith(
        "flight_assistant called book_flight and transfer_to_hotel_assistant in one reply."
    )
    assert "InvalidRequestError: `tool_use` ids were found without `tool_result`" in (
        finding.explanation
    )
    assert "parallel_tool_calls=False" in finding.suggested_fix
    (error,) = [e for e in finding.evidence_excerpts if e.startswith("Error: ")]
    assert "toolu_1" in error and "Traceback" not in error


def test_fixture_error_evidence_in_the_pdfs(tmp_path: Path) -> None:
    from tracedoc.analyze import analyze
    from tracedoc.report import write_reports

    run = load_fixture("swarm_parallel_handoff")
    files = write_reports(analyze([run], Config()), [run], tmp_path)
    (card,) = files.view.cards
    assert card.title == "Run ended in an error"
    assert card.where == "hotel_assistant, model call 1, which failed"
    f = next(f for f in files.view.findings if f.finding.check_id == "design.run_error")
    assert f.error.startswith("InvalidRequestError: Error code: 400")
    assert [(t.name, t.result) for t in f.tools] == [
        ("book_flight", "Booked flight BA117, confirmation K7Q2LM.")
    ]
    details = " ".join(
        " ".join((p.extract_text() or "").split()) for p in PdfReader(files.details).pages
    )
    assert "Error message InvalidRequestError: Error code: 400" in details
    assert "returned: Booked flight BA117, confirmation K7Q2LM." in details
