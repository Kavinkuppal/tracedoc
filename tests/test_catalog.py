"""The reviewed catalog: every check has its text, templates fill, and the words stay plain."""

from __future__ import annotations

import re

import pytest

from tests.fixtures.agents import AGENTS
from tests.helpers import load_fixture
from tracedoc.catalog import CatalogError, catalog, check_templates, fill, placeholders
from tracedoc.checks.base import REGISTRY, run_rule_checks
from tracedoc.checks.judges.prompts import confirmation_prompts, judge_prompts
from tracedoc.checks.signals import find_signals

INTERNAL_WORDS = re.compile(r"\b(node|nodes|span|spans|check_id|langgraph_node)\b", re.IGNORECASE)
CHECK_ID = re.compile(r"\b[a-z]+\.[a-z_]+\b")


def test_every_rule_check_has_title_explanation_and_templates() -> None:
    checks = catalog()["checks"]
    for check_id in REGISTRY:
        entry = checks[check_id]
        assert entry["how"] == "rule"
        assert entry["title"] and entry["looks_for"] and entry["why"]
        variants = [k for k, v in entry.items() if isinstance(v, dict)]
        for variant in variants or [None]:
            what, change = check_templates(check_id, variant)
            assert what and change


def test_every_judge_check_has_title_and_explanation() -> None:
    checks = catalog()["checks"]
    for check_id in judge_prompts():
        entry = checks[check_id]
        assert entry["how"] == "judge"
        assert entry["title"] and entry["looks_for"] and entry["why"]
    assert set(confirmation_prompts()) <= set(REGISTRY)


def test_no_orphan_catalog_checks() -> None:
    known = set(REGISTRY) | set(judge_prompts())
    assert set(catalog()["checks"]) == known


def test_signals_have_lines_and_sentences() -> None:
    for entry in catalog()["signals"].values():
        assert entry["title"] and entry["line"] and entry["looks_for"] and entry["what_happened"]


def test_titles_are_unique_short_and_plain() -> None:
    entries = {**catalog()["checks"], **catalog()["signals"]}
    titles = [e["title"] for e in entries.values()]
    assert len(set(titles)) == len(titles)
    for title in titles:
        assert len(title) <= 45, title
        assert not INTERNAL_WORDS.search(title), title


def test_report_text_has_no_internal_terms_or_check_ids() -> None:
    def strings(value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [s for v in value.values() for s in strings(v)]
        if isinstance(value, list):
            return [s for v in value for s in strings(v)]
        return []

    for name, table in catalog().items():
        for s in strings(table):
            assert not INTERNAL_WORDS.search(s), (name, s)
            ids = [m for m in CHECK_ID.findall(s) if not m.endswith(("pdf", "json", "md"))]
            assert not ids, (name, s)


def test_fill_requires_every_placeholder() -> None:
    assert fill("{agent} called {tool}", agent="a", tool="t", extra=1) == "a called t"
    with pytest.raises(CatalogError):
        fill("{agent} called {tool}", agent="a")
    assert placeholders("In {runs} of {total} runs") == {"runs", "total"}


def test_rule_and_signal_sentences_are_filled_from_facts() -> None:
    runs = [load_fixture(n) for n in AGENTS]
    findings = run_rule_checks(runs) + find_signals(runs)
    assert findings
    for f in findings:
        assert f.facts, f.check_id
        for sentence in (f.explanation, f.suggested_fix):
            assert "{" not in sentence and "}" not in sentence, sentence
