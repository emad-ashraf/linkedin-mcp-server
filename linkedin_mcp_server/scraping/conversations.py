"""Messaging inbox, thread and conversation-search workflows."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Literal
from urllib.parse import quote_plus

import logging

from patchright.async_api import Route
from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.core.exceptions import LinkedInScraperException
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.identifiers import (
    messaging_thread_url,
    normalize_person_identifier,
    normalize_thread_id,
    person_profile_url,
)
from linkedin_mcp_server.scraping.link_metadata import (
    Reference,
    build_references,
    dedupe_references,
)
from linkedin_mcp_server.scraping.messaging_api import MessagingApiCapture
from linkedin_mcp_server.scraping.messaging_payload import (
    build_conversation_references,
    build_message_references,
    conversation_matches_username,
    conversation_thread_path,
    find_conversation_by_thread_id,
    format_message_elements,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.profile_page import ProfilePageReader
from linkedin_mcp_server.scraping.session import ScrapingSession
from linkedin_mcp_server.scraping.text import strip_linkedin_noise

logger = logging.getLogger(__name__)

# ponytail: LinkedIn's /messaging/ page redirects to the newest thread and
# POSTs its read flag here (measured 2026-09-20, same call as the "Mark as
# unread" menu); aborting the POST keeps the user's unread state, GETs still
# flow so the passive capture works
_READ_STATE_WRITE_ROUTES = (
    "**/voyager/api/voyagerMessagingDashMessengerConversations*",
    "**/voyager/api/voyagerMessagingDashMessagingBadge*",
)


class ConversationReader:
    """Own every workflow whose subject is a LinkedIn messaging thread.

    LinkedIn's messaging UI emits stable conversation URLs, participant profile
    URLs, read flags and message entities in its authenticated data responses.
    Reading those responses avoids selecting sidebar rows or opening thread
    routes, which preserves unread state and eliminates layout-dependent clicks.
    """

    def __init__(
        self,
        session: ScrapingSession,
        navigator: PageNavigator,
        content: PageContentReader,
        profile_page: ProfilePageReader,
    ):
        self._session = session
        self._navigator = navigator
        self._content = content
        self._profile_page = profile_page

    @staticmethod
    def _single_section_result(
        url: str,
        section_name: str,
        text: str,
        references: list[Reference] | None = None,
    ) -> dict[str, Any]:
        """Build a standard single-section scraping response."""
        result: dict[str, Any] = {"url": url, "sections": {}}
        if text:
            result["sections"][section_name] = text
            if references:
                result["references"] = {section_name: references}
        return result

    async def _wait_for_main_text(
        self,
        *,
        minimum_length: int = 100,
        timeout: int = 10000,
        log_context: str,
    ) -> None:
        """Wait for main content to populate enough text to scrape."""
        try:
            await self._session.page.wait_for_function(
                """({ minimumLength }) => {
                    const main = document.querySelector('main');
                    if (!main) return false;
                    return main.innerText.length > minimumLength;
                }""",
                arg={"minimumLength": minimum_length},
                timeout=timeout,
            )
        except PlaywrightTimeoutError:
            logger.debug("%s content did not appear", log_context)

    async def _scroll_main_scrollable_region(
        self,
        *,
        position: Literal["top", "bottom"],
        attempts: int,
        pause_time: float = 0.5,
    ) -> None:
        """Scroll the largest scrollable region inside main when one exists."""
        for _ in range(attempts):
            await self._session.page.evaluate(
                """({ position }) => {
                    const main = document.querySelector('main');
                    if (!main) return false;

                    const isScrollable = element => {
                        const style = window.getComputedStyle(element);
                        return (
                            (style.overflowY === 'auto' || style.overflowY === 'scroll') &&
                            element.scrollHeight > element.clientHeight + 20
                        );
                    };

                    const candidates = [main, ...main.querySelectorAll('*')].filter(isScrollable);
                    const target = candidates.sort(
                        (left, right) => right.scrollHeight - left.scrollHeight
                    )[0] || main;
                    target.scrollTop = position === 'top' ? 0 : target.scrollHeight;
                    return true;
                }""",
                {"position": position},
            )
            await self._session.delay(pause_time)

    async def _load_messaging_page(
        self,
        capture: MessagingApiCapture,
        url: str,
        *,
        log_context: str,
        scroll_attempts: int = 0,
    ) -> int:
        """Navigate once and wait for the matching passive conversation payload."""
        offset = len(capture.payloads)
        await self._navigator._navigate_to_page(url)
        await self._session.check_rate_limit()
        await self._wait_for_main_text(log_context=log_context)
        await self._session.dismiss_modal()
        if scroll_attempts:
            await self._scroll_main_scrollable_region(
                position="bottom", attempts=scroll_attempts, pause_time=0.5
            )
        await capture.wait_for_payload(after=offset)
        await capture.settle()
        return offset

    @staticmethod
    async def _abort_read_state_write(route: Route) -> None:
        if route.request.method == "POST":
            await route.abort()
        else:
            await route.continue_()

    @contextlib.asynccontextmanager
    async def _without_read_state_writes(self) -> AsyncIterator[None]:
        """Run a messaging workflow with LinkedIn's read-flag POSTs aborted."""
        page = self._session.page
        handler = self._abort_read_state_write
        for pattern in _READ_STATE_WRITE_ROUTES:
            await page.route(pattern, handler)
        try:
            yield
        finally:
            for pattern in _READ_STATE_WRITE_ROUTES:
                await page.unroute(pattern, handler)

    @staticmethod
    def _unique_conversations(
        conversations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in conversations:
            path = conversation_thread_path(item)
            if path is None or path in seen:
                continue
            seen.add(path)
            unique.append(item)
        return unique

    async def _conversation_for_username(
        self,
        capture: MessagingApiCapture,
        linkedin_username: str,
        index: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Resolve a username through exact participant profile URLs."""
        if index < 0:
            raise LinkedInScraperException(f"index must be non-negative (got {index}).")
        username = normalize_person_identifier(linkedin_username)
        await self._navigator._navigate_to_page(person_profile_url(username, "/"))
        await self._session.check_rate_limit()
        try:
            await self._session.page.wait_for_selector("main")
        except PlaywrightTimeoutError:
            logger.debug("Profile page did not load while resolving a conversation")
        await self._session.dismiss_modal()
        display_name = await self._profile_page._read_profile_display_name()
        if not display_name:
            raise LinkedInScraperException(
                "Could not resolve a display name for the requested profile."
            )

        offset = await self._load_messaging_page(
            capture,
            f"https://www.linkedin.com/messaging/?searchTerm={quote_plus(display_name)}",
            log_context="Messaging search results",
        )
        candidates = self._unique_conversations(capture.conversations_since(offset))
        matches = [
            item for item in candidates if conversation_matches_username(item, username)
        ]
        if not matches or not capture.has_message_template:
            offset = await self._load_messaging_page(
                capture,
                "https://www.linkedin.com/messaging/",
                log_context="Messaging inbox",
                scroll_attempts=2 if not matches else 0,
            )
            inbox_candidates = self._unique_conversations(
                capture.conversations_since(offset)
            )
            if not matches:
                matches = [
                    item
                    for item in inbox_candidates
                    if conversation_matches_username(item, username)
                ]
            candidates = self._unique_conversations(candidates + inbox_candidates)
        if not matches:
            raise LinkedInScraperException(
                "Could not find a conversation for the requested profile."
            )
        if index >= len(matches):
            raise LinkedInScraperException(
                f"index {index} out of range: only {len(matches)} thread(s) exist."
            )
        return matches[index], candidates

    async def get_inbox(self, limit: int = 20) -> dict[str, Any]:
        """List recent conversations without selecting any sidebar row."""
        async with self._without_read_state_writes():
            url = "https://www.linkedin.com/messaging/"
            async with MessagingApiCapture(self._session.page) as capture:
                offset = await self._load_messaging_page(
                    capture,
                    url,
                    log_context="Messaging inbox",
                    scroll_attempts=max(1, limit // 10),
                )
                raw_result = await self._content._extract_root_content(["main"])
                payloads = capture.payloads[offset:]

            raw = raw_result["text"]
            cleaned = strip_linkedin_noise(raw) if raw else ""
            references: list[Reference] = (
                build_references(raw_result["references"], "inbox") if cleaned else []
            )
            conversation_refs = build_conversation_references(
                payloads, limit=limit, context="inbox"
            )
            if conversation_refs:
                references = dedupe_references(conversation_refs + references)
            return self._single_section_result(
                url, "inbox", cleaned, references=references
            )

    async def get_conversation(
        self,
        linkedin_username: str | None = None,
        thread_id: str | None = None,
        index: int = 0,
    ) -> dict[str, Any]:
        """Read a conversation through LinkedIn's passive message response."""
        if not linkedin_username and not thread_id:
            raise LinkedInScraperException(
                "Provide at least one of linkedin_username or thread_id"
            )

        async with self._without_read_state_writes():
            async with MessagingApiCapture(self._session.page) as capture:
                if thread_id:
                    normalized_thread_id = normalize_thread_id(thread_id)
                    offset = await self._load_messaging_page(
                        capture,
                        "https://www.linkedin.com/messaging/",
                        log_context="Messaging inbox",
                        scroll_attempts=1,
                    )
                    conversations = self._unique_conversations(
                        capture.conversations_since(offset)
                    )
                    target = find_conversation_by_thread_id(
                        conversations, normalized_thread_id
                    )
                    if target is None:
                        await self._scroll_main_scrollable_region(
                            position="bottom", attempts=5, pause_time=0.5
                        )
                        await capture.settle()
                        conversations = self._unique_conversations(
                            capture.conversations_since(offset)
                        )
                        target = find_conversation_by_thread_id(
                            conversations, normalized_thread_id
                        )
                    if target is None:
                        raise LinkedInScraperException(
                            "Could not map that thread ID to a passive conversation response."
                        )
                else:
                    target, conversations = await self._conversation_for_username(
                        capture, linkedin_username or "", index
                    )
                    path = conversation_thread_path(target)
                    if path is None:
                        raise LinkedInScraperException(
                            "The matched conversation has no stable thread URL."
                        )
                    normalized_thread_id = normalize_thread_id(
                        path.removeprefix("/messaging/thread/").removesuffix("/")
                    )

                messages = await capture.fetch_message_history(
                    target,
                    conversations=conversations,
                )

            cleaned = format_message_elements(
                messages, timezone=datetime.now().astimezone().tzinfo
            )
            references = (
                build_message_references(messages, conversation=target)
                if cleaned
                else []
            )
            return self._single_section_result(
                messaging_thread_url(normalized_thread_id, "/"),
                "conversation",
                cleaned,
                references=references,
            )

    async def search_conversations(
        self, keywords: str, limit: int = 20
    ) -> dict[str, Any]:
        """Search messages without selecting any result row."""
        async with self._without_read_state_writes():
            search_url = (
                f"https://www.linkedin.com/messaging/?searchTerm={quote_plus(keywords)}"
            )
            async with MessagingApiCapture(self._session.page) as capture:
                offset = await self._load_messaging_page(
                    capture, search_url, log_context="Messaging search"
                )
                raw_result = await self._content._extract_root_content(["main"])
                payloads = capture.payloads[offset:]

            raw = raw_result["text"]
            cleaned = strip_linkedin_noise(raw) if raw else ""
            references: list[Reference] = (
                build_references(raw_result["references"], "search_results")
                if cleaned
                else []
            )
            conversation_refs = build_conversation_references(
                payloads, limit=limit, context="search_results"
            )
            if conversation_refs:
                references = dedupe_references(conversation_refs + references)
            return self._single_section_result(
                self._session.page.url,
                "search_results",
                cleaned,
                references=references,
            )
