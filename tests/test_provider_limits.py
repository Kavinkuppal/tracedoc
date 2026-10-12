"""A rate, spending or billing limit stops the analysis at once: no call after call."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.fakes import FakeLLM, always
from tests.fixtures.agents import AGENTS
from tests.helpers import TRACES, load_fixture
from tracedoc import cli
from tracedoc.checks.base import run_rule_checks
from tracedoc.checks.judges.prompts import enabled_judges
from tracedoc.checks.judges.runner import Judge
from tracedoc.config import Config
from tracedoc.llm.client import ProviderLimitError


class LimitedLLM(FakeLLM):
    """Answers like FakeLLM until `limit_after` calls, then the provider refuses."""

    def __init__(self, limit_after: int) -> None:
        super().__init__(triage=always(True))
        self.limit_after = limit_after

    def complete_json(self, *args: Any, **kwargs: Any):
        if len(self.calls) >= self.limit_after:
            self.calls.append((kwargs["model"], "refused"))
            raise ProviderLimitError("HTTP 429 rate_limit_error")
        return super().complete_json(*args, **kwargs)


def test_judging_stops_at_the_first_refusal() -> None:
    config = Config()
    config.judge.include_unvalidated = True
    config.judge.concurrency = 1
    runs = [load_fixture(n) for n in AGENTS]
    findings = run_rule_checks(runs)
    client = LimitedLLM(limit_after=3)
    judges, _ = enabled_judges(config)
    judge = Judge(config, judges, client=client)
    with pytest.raises(ProviderLimitError):
        judge.run(judge.plan(runs, findings), findings)
    assert [c[1] for c in client.calls].count("refused") == 1  # no retry after a refusal


def test_cli_stops_with_one_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(cli, "make_client", lambda config: LimitedLLM(limit_after=0))
    args = ["analyze", str(TRACES / "capture"), "--out", str(tmp_path), "--yes"]
    result = CliRunner().invoke(cli.app, [*args, "--include-unvalidated"])
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    errors = [line for line in result.output.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1 and "rate, spending or billing limit" in errors[0]
