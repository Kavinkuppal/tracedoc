"""Render the report offline in Chromium, Firefox, and WebKit (Safari's engine).

pip install playwright && python -m playwright install chromium firefox webkit
pytest -m browser --browser
"""

from __future__ import annotations

import pytest

from tests.fixtures.agents import AGENTS
from tests.helpers import load_fixture
from tracedoc.analyze import analyze
from tracedoc.config import Config
from tracedoc.report import write_reports

pytestmark = pytest.mark.browser
playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def report_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    runs = [load_fixture(n) for n in AGENTS]
    out = tmp_path_factory.mktemp("report")
    return (write_reports(analyze(runs, Config()), runs, out)).as_uri()


@pytest.mark.parametrize("engine", ["chromium", "firefox", "webkit"])
@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_renders_offline_without_errors(report_url: str, engine: str, scheme: str) -> None:
    with playwright.sync_playwright() as p:
        browser = getattr(p, engine).launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900}, color_scheme=scheme)
        errors: list[str] = []
        external: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on(
            "request",
            lambda r: (
                external.append(r.url)
                if not r.url.startswith(("file:", "data:", "blob:"))
                else None
            ),
        )
        page.goto(report_url)
        page.wait_for_timeout(500)
        assert page.locator(".chart svg").count() == 9
        assert page.locator(".chart[data-empty]").count() == 0
        assert page.locator("#top-fixes .fix").count() == 3
        assert page.locator("#findings-body tr").count() > 0
        page.select_option("#f-task", "research")
        assert "3 of 6 runs" in page.locator("#filter-note").inner_text()
        page.select_option("#f-task", "")
        page.locator("#top-fixes a.evidence").first.click()
        page.wait_for_timeout(200)
        assert page.locator(".step.highlight").count() > 0
        page.locator("#findings-body button[data-vote=up]").first.click()
        assert page.locator("#vote-cmd").inner_text().startswith("tracedoc feedback ")
        assert not errors, errors
        assert not external, external
        browser.close()
