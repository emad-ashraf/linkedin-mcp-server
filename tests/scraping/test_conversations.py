"""Tests for passive messaging conversation workflows."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from linkedin_mcp_server.core.exceptions import (
    InvalidReferenceError,
    LinkedInScraperException,
)
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.conversations import ConversationReader
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.profile_page import ProfilePageReader
from linkedin_mcp_server.scraping.session import ScrapingSession


async def _no_message_target() -> SimpleNamespace:
    raise AssertionError("the conversation reader never reads a message target")


def _reader(page: Any) -> ConversationReader:
    session = ScrapingSession(page)
    return ConversationReader(
        session,
        PageNavigator(session),
        PageContentReader(session),
        ProfilePageReader(session, _no_message_target),
    )


def _member(username: str, first: str, last: str) -> dict[str, Any]:
    return {
        "participantType": {
            "member": {
                "firstName": {"text": first},
                "lastName": {"text": last},
                "profileUrl": f"https://www.linkedin.com/in/{username}/",
            }
        }
    }


def _conversation(
    thread_id: str,
    *,
    title: str = "Ada Lovelace",
    username: str = "ada",
    read: bool = True,
) -> dict[str, Any]:
    return {
        "backendUrn": f"urn:li:messagingThread:{thread_id}",
        "entityUrn": f"urn:li:msg_conversation:({thread_id})",
        "conversationUrl": f"https://www.linkedin.com/messaging/thread/{thread_id}/",
        "conversationParticipants": [
            _member(username, title.split()[0], title.split()[-1]),
            _member("viewer", "Current", "User"),
        ],
        "title": {"text": title},
        "read": read,
        "unreadCount": 0 if read else 1,
    }


def _inbox(*items: dict[str, Any]) -> dict[str, Any]:
    return {
        "data": {
            "messengerConversationsByCategoryQuery": {
                "elements": list(items),
                "metadata": {"nextCursor": "private"},
            }
        }
    }


def _messages(*bodies: str) -> dict[str, Any]:
    target_urn = "urn:li:msg_conversation:(2-target)"
    return {
        "data": {
            "messengerMessagesBySyncToken": {
                "elements": [
                    {
                        "backendConversationUrn": target_urn,
                        "deliveredAt": index,
                        "sender": _member("ada", "Ada", "Lovelace"),
                        "body": {"text": body},
                    }
                    for index, body in enumerate(bodies)
                ]
            }
        }
    }


class FakeCapture:
    def __init__(
        self,
        payloads: list[dict[str, Any]],
        *,
        messages: dict[str, Any] | None = None,
        has_message_template: bool = True,
    ) -> None:
        self.payloads: list[dict[str, Any]] = []
        self._queued = list(payloads)
        self.messages = messages or _messages("Hello")
        self.has_message_template = has_message_template
        self.fetch_calls: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    async def __aenter__(self) -> FakeCapture:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def wait_for_payload(self, *, after: int, timeout: float = 12.0) -> None:
        del timeout
        if len(self.payloads) <= after and self._queued:
            self.payloads.append(self._queued.pop(0))

    async def settle(self, timeout: float = 1.0) -> None:
        del timeout

    def conversations_since(self, offset: int = 0) -> list[dict[str, Any]]:
        return [
            item
            for payload in self.payloads[offset:]
            for item in payload["data"]["messengerConversationsByCategoryQuery"][
                "elements"
            ]
        ]

    async def fetch_message_history(
        self,
        target: dict[str, Any],
        *,
        conversations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        self.fetch_calls.append((target, conversations))
        return self.messages["data"]["messengerMessagesBySyncToken"]["elements"]


@pytest.fixture(autouse=True)
def session_boundaries():
    with (
        patch.object(ScrapingSession, "check_rate_limit", new_callable=AsyncMock),
        patch.object(ScrapingSession, "dismiss_modal", new_callable=AsyncMock),
        patch.object(ScrapingSession, "delay", new_callable=AsyncMock),
    ):
        yield


def _root(text: str, references: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"source": "root", "text": text, "references": references or []}


class TestGetInbox:
    async def test_mixed_read_states_produce_refs_without_row_clicks(self, mock_page):
        unread = _conversation("2-unread", read=False)
        read = _conversation("2-read", title="Grace Hopper", username="grace")
        capture = FakeCapture([_inbox(unread, read)])
        reader = _reader(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.conversations.MessagingApiCapture",
                return_value=capture,
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(reader, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                reader, "_scroll_main_scrollable_region", new_callable=AsyncMock
            ),
            patch.object(
                PageContentReader,
                "_extract_root_content",
                new_callable=AsyncMock,
                return_value=_root("Ada Lovelace\nGrace Hopper"),
            ),
        ):
            result = await reader.get_inbox(limit=20)

        assert [ref["url"] for ref in result["references"]["inbox"]] == [
            "/messaging/thread/2-unread/",
            "/messaging/thread/2-read/",
        ]
        assert unread["read"] is False
        assert read["read"] is True
        mock_page.locator.return_value.click.assert_not_called()

    async def test_empty_page_omits_optional_sections_and_references(self, mock_page):
        capture = FakeCapture([_inbox()])
        reader = _reader(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.conversations.MessagingApiCapture",
                return_value=capture,
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(reader, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                reader, "_scroll_main_scrollable_region", new_callable=AsyncMock
            ),
            patch.object(
                PageContentReader,
                "_extract_root_content",
                new_callable=AsyncMock,
                return_value=_root(""),
            ),
        ):
            result = await reader.get_inbox(limit=5)

        assert result == {
            "url": "https://www.linkedin.com/messaging/",
            "sections": {},
        }


class TestGetConversation:
    async def test_direct_id_uses_one_inbox_navigation_and_no_thread_navigation(
        self, mock_page
    ):
        source = _conversation("2-source")
        target = _conversation("2-target", read=False)
        capture = FakeCapture(
            [_inbox(source, target)], messages=_messages("Hi", "Hello")
        )
        reader = _reader(mock_page)
        nav = AsyncMock()
        scroll = AsyncMock()
        with (
            patch(
                "linkedin_mcp_server.scraping.conversations.MessagingApiCapture",
                return_value=capture,
            ),
            patch.object(PageNavigator, "_navigate_to_page", nav),
            patch.object(reader, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(reader, "_scroll_main_scrollable_region", scroll),
        ):
            result = await reader.get_conversation(thread_id="2-target")

        nav.assert_awaited_once_with("https://www.linkedin.com/messaging/")
        scroll.assert_awaited_once_with(position="bottom", attempts=1, pause_time=0.5)
        assert result["url"] == "https://www.linkedin.com/messaging/thread/2-target/"
        conversation = result["sections"]["conversation"]
        assert conversation.count("sent the following message") == 2
        assert conversation.index("Hi") < conversation.index("Hello")
        assert capture.fetch_calls[0][0] is target
        assert target["read"] is False

    async def test_direct_id_preserves_message_references(self, mock_page):
        target = _conversation("2-target", read=False)
        messages = _messages("Attachment follows")
        messages["data"]["messengerMessagesBySyncToken"]["elements"][0][
            "attachments"
        ] = [{"url": "https://example.com/private-file"}]
        capture = FakeCapture([_inbox(target)], messages=messages)
        reader = _reader(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.conversations.MessagingApiCapture",
                return_value=capture,
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(reader, "_wait_for_main_text", new_callable=AsyncMock),
        ):
            result = await reader.get_conversation(thread_id="2-target")

        refs = result["references"]["conversation"]
        assert any(ref["url"] == "/in/ada/" for ref in refs)
        assert any(ref["url"] == "https://example.com/private-file" for ref in refs)

    async def test_direct_id_outside_passive_results_fails_instead_of_guessing(
        self, mock_page
    ):
        capture = FakeCapture([_inbox(_conversation("2-source"))])
        reader = _reader(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.conversations.MessagingApiCapture",
                return_value=capture,
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(reader, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                reader, "_scroll_main_scrollable_region", new_callable=AsyncMock
            ) as scroll,
        ):
            with pytest.raises(LinkedInScraperException, match="Could not map"):
                await reader.get_conversation(thread_id="2-older")

        assert capture.fetch_calls == []
        assert [call.kwargs for call in scroll.await_args_list] == [
            {"position": "bottom", "attempts": 1, "pause_time": 0.5},
            {"position": "bottom", "attempts": 5, "pause_time": 0.5},
        ]

    async def test_username_uses_profile_identity_not_duplicate_display_name(
        self, mock_page
    ):
        namesake = _conversation("2-wrong", title="Alex Smith", username="other-alex")
        target = _conversation(
            "2-target", title="Alex Smith", username="target-alex", read=False
        )
        capture = FakeCapture([_inbox(namesake, target)])
        reader = _reader(mock_page)
        nav = AsyncMock()
        with (
            patch(
                "linkedin_mcp_server.scraping.conversations.MessagingApiCapture",
                return_value=capture,
            ),
            patch.object(PageNavigator, "_navigate_to_page", nav),
            patch.object(reader, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                ProfilePageReader,
                "_read_profile_display_name",
                new_callable=AsyncMock,
                return_value="Alex Smith",
            ),
        ):
            result = await reader.get_conversation(linkedin_username="target-alex")

        assert [call.args[0] for call in nav.await_args_list] == [
            "https://www.linkedin.com/in/target-alex/",
            "https://www.linkedin.com/messaging/?searchTerm=Alex+Smith",
        ]
        assert capture.fetch_calls[0][0] is target
        assert result["url"].endswith("/messaging/thread/2-target/")

    async def test_username_loads_inbox_when_search_emits_no_message_template(
        self, mock_page
    ):
        target = _conversation("2-target", username="ada", read=False)
        source = _conversation("2-source", username="grace")
        capture = FakeCapture(
            [_inbox(target), _inbox(source)], has_message_template=False
        )
        reader = _reader(mock_page)
        nav = AsyncMock()
        with (
            patch(
                "linkedin_mcp_server.scraping.conversations.MessagingApiCapture",
                return_value=capture,
            ),
            patch.object(PageNavigator, "_navigate_to_page", nav),
            patch.object(reader, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                ProfilePageReader,
                "_read_profile_display_name",
                new_callable=AsyncMock,
                return_value="Ada Lovelace",
            ),
        ):
            await reader.get_conversation(linkedin_username="ada")

        assert [call.args[0] for call in nav.await_args_list] == [
            "https://www.linkedin.com/in/ada/",
            "https://www.linkedin.com/messaging/?searchTerm=Ada+Lovelace",
            "https://www.linkedin.com/messaging/",
        ]
        assert capture.fetch_calls[0][0] is target
        assert source in capture.fetch_calls[0][1]

    async def test_username_index_keeps_provider_order(self, mock_page):
        first = _conversation("2-first", username="ada")
        second = _conversation("2-second", username="ada")
        capture = FakeCapture([_inbox(first, second)])
        reader = _reader(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.conversations.MessagingApiCapture",
                return_value=capture,
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(reader, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                ProfilePageReader,
                "_read_profile_display_name",
                new_callable=AsyncMock,
                return_value="Ada Lovelace",
            ),
        ):
            await reader.get_conversation(linkedin_username="ada", index=1)

        assert capture.fetch_calls[0][0] is second

    async def test_no_identifier_is_refused_before_page_work(self, mock_page):
        reader = _reader(mock_page)
        with patch.object(
            PageNavigator, "_navigate_to_page", new_callable=AsyncMock
        ) as nav:
            with pytest.raises(LinkedInScraperException, match="at least one"):
                await reader.get_conversation()
        nav.assert_not_awaited()

    async def test_invalid_thread_id_is_refused_before_navigation(self, mock_page):
        capture = FakeCapture([])
        reader = _reader(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.conversations.MessagingApiCapture",
                return_value=capture,
            ),
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as nav,
        ):
            with pytest.raises(InvalidReferenceError):
                await reader.get_conversation(thread_id="../../feed")
        nav.assert_not_awaited()


class TestSearchConversations:
    async def test_search_returns_passive_refs_without_selecting_results(
        self, mock_page
    ):
        unread = _conversation("2-unread", read=False)
        capture = FakeCapture([_inbox(unread)])
        reader = _reader(mock_page)
        nav = AsyncMock()
        with (
            patch(
                "linkedin_mcp_server.scraping.conversations.MessagingApiCapture",
                return_value=capture,
            ),
            patch.object(PageNavigator, "_navigate_to_page", nav),
            patch.object(reader, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                PageContentReader,
                "_extract_root_content",
                new_callable=AsyncMock,
                return_value=_root("Ada Lovelace"),
            ),
        ):
            result = await reader.search_conversations("hello world", limit=10)

        nav.assert_awaited_once_with(
            "https://www.linkedin.com/messaging/?searchTerm=hello+world"
        )
        assert result["references"]["search_results"][0]["url"] == (
            "/messaging/thread/2-unread/"
        )
        assert unread["read"] is False
        mock_page.locator.return_value.click.assert_not_called()
