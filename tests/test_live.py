"""Acceptance against a real model. Spends money. Run with:

ANTHROPIC_API_KEY=... pytest -m live --live -s
"""

from __future__ import annotations

import pytest

from tests.fixtures.agents import AGENTS
from tests.helpers import load_fixture
from tracedoc.analyze import analyze
from tracedoc.config import load_config
from tracedoc.llm.client import has_credentials, make_client
from tracedoc.llm.cost import Spend

pytestmark = pytest.mark.live


def test_real_judges_find_intended_issues_on_fixtures() -> None:
    config = load_config()
    if not has_credentials(config.llm):
        pytest.skip("no model key configured")
    config.judge.include_unvalidated = True
    runs = [load_fixture(n) for n in AGENTS]
    names = {r.id: n for r, n in zip(runs, AGENTS, strict=True)}
    spend = Spend(prices=config.prices, max_cost=2.0)
    report = analyze(runs, config, client=make_client(config.llm), spend=spend)

    found: dict[str, set[str]] = {n: set() for n in AGENTS}
    for group in report.groups:
        for f in group.findings:
            if not f.needs_confirmation:
                found[names[f.run_id]].add(f.check_id)
    print("\nFindings per fixture:", found, f"\nCost: ${spend.cost:.3f} ({spend.calls} calls)")
    for name, module in AGENTS.items():
        if module.BUG:
            assert module.BUG in found[name], name
    healthy_high = [
        f
        for g in report.groups
        for f in g.findings
        if names[f.run_id] == "healthy" and f.severity == "high"
    ]
    assert not healthy_high
