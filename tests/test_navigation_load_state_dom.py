"""Browser-DOM test for a navigation whose load state lags its commit.

The unit suite mocks ``page.goto`` and ``page.wait_for_load_state``, so it
can only assert the order of the calls; whether a committed page with a
rendered ``<main>`` can genuinely sit below ``domcontentloaded`` is a claim
about the browser. Chromium holds ``DOMContentLoaded`` until every parser
blocking script has finished, so a ``<script src>`` whose request is never
answered is exactly the shape LinkedIn's SDUI pages take (#42): the DOM is
on screen and usable, and the load state is not coming.

The navigator's load-state budget, ``PageNavigator._LOAD_STATE_TIMEOUT``,
is lowered for the test so the real wait runs against the real page with a
budget the test can afford.

Skipped automatically when chromium is not installed; run locally after
``uv run patchright install chromium --no-shell``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

import logging
import os

import pytest
from patchright.async_api import (
    Page,
    Route,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]

URL = "https://www.linkedin.com/in/testuser/"
SLOW_SCRIPT = "https://www.linkedin.com/slow.js"
LOAD_STATE_BUDGET_SECONDS = 1.5

HTML = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<title>Test User | LinkedIn</title></head>"
    "<body><main>Ready</main><script src='/slow.js'></script></body></html>"
)


async def _dom_page() -> AsyncIterator[Page]:
    """A real page serving ``URL`` with a script request that never answers.

    The held route is aborted on teardown so the browser closes with no
    request still pending.
    """
    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(
                channel="chromium", headless=True
            )
            page = await browser.new_page()
        except Exception as exc:
            if os.environ.get("CI"):
                raise
            pytest.skip(f"chromium unavailable: {exc}")
        held: list[Route] = []

        async def hold(route: Route) -> None:
            held.append(route)

        await page.route(SLOW_SCRIPT, hold)
        await page.route(
            URL,
            lambda route: route.fulfill(content_type="text/html", body=HTML),
        )
        try:
            yield page
        finally:
            for route in held:
                await route.abort()
            await browser.close()


@pytest.fixture
async def dom_page():
    async for page in _dom_page():
        yield page


async def test_the_premise_a_rendered_page_can_sit_below_domcontentloaded(dom_page):
    """What the fix relies on, measured: commit lands, DOM renders, DCL waits."""
    await dom_page.goto(URL, wait_until="commit")

    assert await dom_page.locator("main").inner_text() == "Ready"
    with pytest.raises(PlaywrightTimeoutError):
        await dom_page.wait_for_load_state(
            "domcontentloaded", timeout=LOAD_STATE_BUDGET_SECONDS * 1000
        )
    assert await dom_page.evaluate("document.readyState") == "loading"


async def test_the_navigator_proceeds_on_the_committed_page(
    dom_page, caplog, monkeypatch
):
    """The real navigator, over the real page: a warning, not a failure."""
    monkeypatch.setattr(PageNavigator, "_LOAD_STATE_TIMEOUT", LOAD_STATE_BUDGET_SECONDS)
    navigator = PageNavigator(ScrapingSession(cast(Page, dom_page)))

    with caplog.at_level(logging.WARNING, logger="linkedin_mcp_server"):
        await navigator._goto_with_auth_checks(URL)

    assert any(
        "lagged behind a committed navigation" in r.getMessage() for r in caplog.records
    )
    assert dom_page.url == URL
    assert await dom_page.locator("main").inner_text() == "Ready"
    # Still below the load state it waited for: the navigator did not get
    # there by outlasting the script, it got there by not requiring it.
    assert await dom_page.evaluate("document.readyState") == "loading"
