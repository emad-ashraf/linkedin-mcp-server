"""Regression witnesses for the four upstream reliability backports."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from linkedin_mcp_server.core.auth import raise_if_auth_route
from linkedin_mcp_server.core.exceptions import (
    AuthenticationError,
    AccountRestrictedError,
)
from linkedin_mcp_server.drivers import browser as driver
from linkedin_mcp_server.scraping.contracts import ExtractedSection
from linkedin_mcp_server.scraping.job_pages import JobPageCapture
from linkedin_mcp_server.scraping.jobs import JobScraper
from linkedin_mcp_server.session_state import portable_cookie_path


@pytest.mark.parametrize(
    "module_name,tool_name", [("person", "search_people"), ("post", "search_posts")]
)
async def test_shaped_search_errors_are_redacted(module_name, tool_name):
    from importlib import import_module

    module = import_module(f"linkedin_mcp_server.tools.{module_name}")
    mcp = FastMCP("regression")
    getattr(module, f"register_{module_name}_tools")(mcp)
    config = SimpleNamespace(
        proxy_username="private-user", proxy_password="private-pass"
    )
    with (
        patch.object(
            module,
            "get_ready_extractor",
            AsyncMock(
                side_effect=ToolError("http://private-user:private-pass@proxy.example")
            ),
        ),
        patch(
            "linkedin_mcp_server.core.proxy_errors._browser_config", return_value=config
        ),
    ):
        with pytest.raises(ToolError) as raised:
            await mcp.call_tool(tool_name, {"keywords": "python"})
    assert "private-user" not in str(raised.value)
    assert "private-pass" not in str(raised.value)
    assert "proxy.example" in str(raised.value)


@pytest.mark.parametrize(
    "path,error",
    [
        ("/checkpoint/challenge/123", AuthenticationError),
        ("/authwall", AuthenticationError),
        ("/login", AuthenticationError),
        ("/checkpoint/lg/login/login-restriction", AccountRestrictedError),
    ],
)
def test_auth_routes_are_not_throttles(path, error):
    with pytest.raises(error):
        raise_if_auth_route(f"https://www.linkedin.com{path}")


@pytest.mark.parametrize(
    "url",
    [
        "https://www.linkedin.com/in/authwall-consultant",
        "https://linkedin.com.example/checkpoint",
        "https://example.com/authwall",
    ],
)
def test_auth_routes_do_not_match_slugs_or_foreign_hosts(url):
    raise_if_auth_route(url)


@pytest.mark.parametrize(
    "cookies,restore",
    [
        ([], True),
        ([{"name": "li_at", "domain": ".linkedin.com"}], False),
        ([{"name": "li_at", "domain": "notlinkedin.com"}], True),
    ],
)
async def test_cookie_store_recovery(tmp_path, cookies, restore):
    profile = tmp_path / "profile"
    portable_cookie_path(profile).write_text("[]")
    browser = MagicMock()
    browser.context.cookies = AsyncMock(return_value=cookies)
    browser.import_cookies = AsyncMock(return_value=True)
    await driver._restore_portable_session_if_the_store_lost_it(browser, profile)
    assert browser.import_cookies.await_count == int(restore)


async def test_restore_precedes_trusted_probe(tmp_path):
    browser = MagicMock()
    browser.start = AsyncMock()
    restored = False

    async def restore(*args):
        nonlocal restored
        restored = True

    async def cookies(page):
        assert restored, "auth probe ran before portable-session recovery"
        return {}

    with (
        patch.object(driver, "_make_browser", return_value=browser),
        patch.object(
            driver,
            "_restore_portable_session_if_the_store_lost_it",
            side_effect=restore,
        ),
        patch.object(driver, "auth_cookies", side_effect=cookies),
        patch.object(driver, "_probe_still_trusted", return_value=True),
    ):
        assert (
            await driver._authenticate_existing_profile(
                tmp_path, launch_options={}, viewport={}
            )
            is browser
        )


async def test_job_page_timeout_preserves_previous_results():
    session = MagicMock()
    session.pace = AsyncMock()
    session.jittered = lambda value: value
    pages = MagicMock()
    first = True

    async def page(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            return JobPageCapture(
                section=ExtractedSection(text="first page", references=[]),
                landed_url="https://www.linkedin.com/jobs/search/?keywords=python",
                scroll_seconds=0,
            )
        await asyncio.sleep(0.2)
        raise AssertionError("page escaped the search deadline")

    pages._extract_search_page = AsyncMock(side_effect=page)
    pages._extract_job_ids = AsyncMock(return_value=["123"])
    pages._extract_promoted_job_ids = AsyncMock(return_value=[])
    pages._get_total_search_pages = AsyncMock(return_value=3)
    scraper = JobScraper(session, MagicMock(), MagicMock(), pages)
    with patch("linkedin_mcp_server.scraping.jobs.nav_delay", return_value=0):
        result = await scraper.search_jobs("python", max_pages=3, tool_timeout=0.1)
    assert result["job_ids"] == ["123"]
    assert "first page" in result["sections"]["search_results"]
    assert result["section_errors"]["search_results"]["error_type"] == "TimeoutError"
    assert pages._extract_search_page.await_count == 2


@pytest.mark.parametrize("confirmed", [True, False])
async def test_late_auth_route_requires_confirmation(confirmed):
    from linkedin_mcp_server.core.exceptions import TransientBarrierError
    from linkedin_mcp_server.core.utils import detect_rate_limit

    page = MagicMock()
    page.url = "https://www.linkedin.com/checkpoint/challenge/123"
    confirm = AsyncMock(return_value=confirmed)
    with patch("linkedin_mcp_server.core.utils.barrier_confirmed", confirm):
        with pytest.raises(AuthenticationError if confirmed else TransientBarrierError):
            await detect_rate_limit(page)
    confirm.assert_awaited_once()


async def test_unaffordable_job_retry_keeps_throttle_capture():
    import time
    from linkedin_mcp_server.scraping.contracts import RATE_LIMITED_SECTION_TEXT
    from linkedin_mcp_server.scraping.job_pages import JobPageReader
    from linkedin_mcp_server.scraping.session import ScrapingSession

    session = ScrapingSession(MagicMock())
    reader = JobPageReader(session, MagicMock(), MagicMock())
    with (
        patch.object(
            reader,
            "_extract_search_page_once",
            AsyncMock(
                return_value=ExtractedSection(
                    text=RATE_LIMITED_SECTION_TEXT, references=[]
                )
            ),
        ),
        patch.object(ScrapingSession, "claim_soft_retry", AsyncMock()) as retry,
    ):
        result = await reader._extract_search_page(
            "https://www.linkedin.com/jobs/search/",
            "search_results",
            page_deadline=time.monotonic() + 1,
        )
    assert result.section.text == RATE_LIMITED_SECTION_TEXT
    retry.assert_not_awaited()
