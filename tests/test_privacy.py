"""Privacy boundaries for messaging URLs and retained diagnostics."""

from unittest.mock import AsyncMock, MagicMock, patch

from linkedin_mcp_server.privacy import (
    is_linkedin_messaging_url,
    is_private_linkedin_navigation_url,
    redact_private_navigation_value,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession


def test_messaging_url_redaction_removes_thread_ids_and_search_terms():
    value = {
        "target_url": "https://www.linkedin.com/messaging/thread/private-id/",
        "hops": ["https://www.linkedin.com/messaging/?searchTerm=private+medical+term"],
        "ordinary": "https://www.linkedin.com/feed/",
    }

    redacted = redact_private_navigation_value(value)

    assert "private-id" not in repr(redacted)
    assert "medical" not in repr(redacted)
    assert redacted["ordinary"] == "https://www.linkedin.com/feed/"


def test_only_linkedin_messaging_routes_trigger_private_mode():
    assert is_linkedin_messaging_url("https://www.linkedin.com/messaging/")
    assert is_linkedin_messaging_url(
        "https://www.linkedin.com/messaging/thread/private/"
    )
    assert not is_linkedin_messaging_url("https://www.linkedin.com/feed/")
    assert not is_linkedin_messaging_url("https://example.com/messaging/")
    assert is_private_linkedin_navigation_url(
        "https://www.linkedin.com/in/private-person/"
    )


def test_profile_url_redaction_removes_participant_identity():
    redacted = redact_private_navigation_value(
        "navigation failed at https://www.linkedin.com/in/private-person/"
    )

    assert "private-person" not in redacted


async def test_navigation_failure_log_omits_messaging_page_content(caplog):
    page = MagicMock()
    page.url = "https://www.linkedin.com/messaging/thread/private-current/"
    page.title = AsyncMock(return_value="Private title")
    page.evaluate = AsyncMock(return_value="Private body")
    locator = MagicMock()
    locator.count = AsyncMock(return_value=0)
    page.locator.return_value = locator
    navigator = PageNavigator(ScrapingSession(page))

    with patch(
        "linkedin_mcp_server.scraping.navigation.detect_auth_barrier",
        new_callable=AsyncMock,
        return_value=None,
    ):
        await navigator._log_navigation_failure(
            "https://www.linkedin.com/messaging/?searchTerm=private+query",
            "domcontentloaded",
            RuntimeError(
                "failed at https://www.linkedin.com/messaging/thread/private-error/"
            ),
            ["https://www.linkedin.com/messaging/thread/private-hop/"],
        )

    rendered = caplog.text
    assert "private-current" not in rendered
    assert "private query" not in rendered
    assert "private-error" not in rendered
    assert "private-hop" not in rendered
    assert "Private title" not in rendered
    assert "Private body" not in rendered
    page.title.assert_not_awaited()
    page.evaluate.assert_not_awaited()
