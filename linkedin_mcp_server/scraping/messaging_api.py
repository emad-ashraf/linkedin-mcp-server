"""Passive capture and replay of LinkedIn messaging data requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import asyncio

import anyio
import anyio.lowlevel

from linkedin_mcp_server.core.exceptions import LinkedInScraperException
from linkedin_mcp_server.scraping.messaging_payload import (
    build_message_page_url,
    conversation_elements,
    merge_message_elements,
    message_elements,
    replace_conversation_urn,
)


_FORWARDED_HEADERS = frozenset(
    {
        "accept",
        "csrf-token",
        "x-li-lang",
        "x-li-page-instance",
        "x-li-track",
        "x-restli-protocol-version",
    }
)
_MESSAGE_PAGE_SIZE = 20
_MAX_OLDER_PAGES = 3


def _operation(url: str) -> str | None:
    query_id = parse_qs(urlparse(url).query).get("queryId", [""])[0]
    operation = query_id.split(".", 1)[0]
    return operation if operation.isidentifier() else None


def _is_history_request(url: str) -> bool:
    variables = parse_qs(urlparse(url).query).get("variables", [""])[0]
    return all(
        marker in variables
        for marker in ("deliveredAt:", "countBefore:", "countAfter:")
    )


def _has_conversation_container(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    return isinstance(data, dict) and isinstance(
        data.get("messengerConversationsByCategoryQuery"), dict
    )


def _verified_message_batch(
    payload: Any,
    expected_urns: frozenset[str],
    *,
    container_key: str,
    allow_empty: bool = False,
) -> bool:
    """Require every returned message to name the requested conversation."""
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    if not isinstance(data, dict):
        return False
    container = data.get(container_key)
    if not isinstance(container, dict):
        return False
    elements = container.get("elements")
    return (
        isinstance(elements, list)
        and (allow_empty or bool(elements))
        and all(
            isinstance(item, dict)
            and item.get("backendConversationUrn") in expected_urns
            for item in elements
        )
    )


@dataclass(frozen=True, slots=True)
class _RequestTemplate:
    url: str
    headers: dict[str, str]


class MessagingApiCapture:
    """Collect messaging payloads emitted by one already-authenticated page."""

    def __init__(self, page: Any):
        self._page = page
        self.payloads: list[dict[str, Any]] = []
        self._request_template: _RequestTemplate | None = None
        self._history_template: _RequestTemplate | None = None
        self._payload_event = asyncio.Event()
        self._template_event = asyncio.Event()
        self._history_template_event = asyncio.Event()
        self._pending: list[asyncio.Task[None]] = []
        self._template_capture_started = False
        self._history_template_capture_started = False
        # Event emitters remove listeners by the registered callable identity.
        self._response_listener = self._on_response
        self._request_listener = self._on_request

    async def __aenter__(self) -> MessagingApiCapture:
        self._page.on("response", self._response_listener)
        self._page.on("request", self._request_listener)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        try:
            self._page.remove_listener("response", self._response_listener)
        except Exception:
            pass
        try:
            self._page.remove_listener("request", self._request_listener)
        except Exception:
            pass
        await self._drain_pending()

    def _on_response(self, response: Any) -> None:
        if _operation(response.url) != "messengerConversations":
            return

        async def read() -> None:
            try:
                payload = await response.json()
            except Exception:
                return
            if _has_conversation_container(payload):
                self.payloads.append(payload)
                self._payload_event.set()

        self._pending.append(asyncio.create_task(read()))

    def _on_request(self, request: Any) -> None:
        if (
            request.method.upper() != "GET"
            or _operation(request.url) != "messengerMessages"
        ):
            return
        history = _is_history_request(request.url)
        if history:
            if self._history_template_capture_started:
                return
            self._history_template_capture_started = True
        else:
            if self._template_capture_started:
                return
            self._template_capture_started = True

        async def read() -> None:
            try:
                raw_headers = await request.all_headers()
            except Exception:
                if history:
                    self._history_template_capture_started = False
                else:
                    self._template_capture_started = False
                return
            headers = {
                key: value
                for key, value in raw_headers.items()
                if key.casefold() in _FORWARDED_HEADERS
            }
            template = _RequestTemplate(request.url, headers)
            if history:
                self._history_template = template
                self._history_template_event.set()
            else:
                self._request_template = template
                self._template_event.set()

        self._pending.append(asyncio.create_task(read()))

    async def wait_for_payload(self, *, after: int = 0, timeout: float = 12.0) -> None:
        """Wait until a new conversation response has been parsed."""
        if len(self.payloads) > after:
            return
        self._payload_event.clear()
        try:
            async with asyncio.timeout(timeout):
                while len(self.payloads) <= after:
                    await self._payload_event.wait()
                    self._payload_event.clear()
        except TimeoutError as exc:
            raise LinkedInScraperException(
                "LinkedIn did not return conversation data in time."
            ) from exc

    async def _wait_for_template(self, timeout: float = 12.0) -> _RequestTemplate:
        if self._request_template is None:
            try:
                await asyncio.wait_for(self._template_event.wait(), timeout=timeout)
            except TimeoutError as exc:
                raise LinkedInScraperException(
                    "LinkedIn did not expose its passive message request in time."
                ) from exc
        if self._request_template is None:
            raise LinkedInScraperException(
                "LinkedIn did not expose its passive message request."
            )
        return self._request_template

    async def _wait_for_history_template(
        self, timeout: float = 5.0
    ) -> _RequestTemplate:
        if self._history_template is None:
            try:
                await asyncio.wait_for(
                    self._history_template_event.wait(), timeout=timeout
                )
            except TimeoutError as exc:
                raise LinkedInScraperException(
                    "LinkedIn did not expose its passive history request in time."
                ) from exc
        if self._history_template is None:
            raise LinkedInScraperException(
                "LinkedIn did not expose its passive history request."
            )
        return self._history_template

    def conversations_since(self, offset: int = 0) -> list[dict[str, Any]]:
        """Flatten captured conversation payloads while preserving provider order."""
        return [
            item
            for payload in self.payloads[offset:]
            for item in conversation_elements(payload)
        ]

    @property
    def has_message_template(self) -> bool:
        """Whether the page emitted a replayable passive message request."""
        return self._request_template is not None

    @property
    def has_history_template(self) -> bool:
        """Whether the page emitted the anchor-timestamp history query."""
        return self._history_template is not None

    async def settle(self, timeout: float = 1.0) -> None:
        """Let response reads already emitted by the browser finish parsing."""
        active = [task for task in self._pending if not task.done()]
        if active:
            await asyncio.wait(active, timeout=timeout)

    @staticmethod
    def _identity_for_template(
        conversations: list[dict[str, Any]], request_url: str
    ) -> tuple[str, str] | None:
        for item in conversations:
            for key in ("entityUrn", "backendUrn"):
                value = item.get(key)
                if not isinstance(value, str):
                    continue
                if value in request_url or quote(value, safe="") in request_url:
                    return key, value
        return None

    async def fetch_messages(
        self,
        target: dict[str, Any],
        *,
        conversations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Replay the captured read-only message query for one exact conversation."""
        target_url, _, target_urns, headers = await self._target_request(
            target, conversations=conversations
        )
        result = await self._request_json(target_url, headers)
        status = result.get("status") if isinstance(result, dict) else None
        payload = result.get("payload") if isinstance(result, dict) else None
        if not isinstance(status, int) or not 200 <= status < 300:
            raise LinkedInScraperException(
                "LinkedIn refused the passive conversation read."
            )
        if not isinstance(payload, dict) or not _verified_message_batch(
            payload,
            target_urns,
            container_key="messengerMessagesBySyncToken",
        ):
            raise LinkedInScraperException(
                "LinkedIn returned unverified conversation data."
            )
        return payload

    async def _target_request(
        self,
        target: dict[str, Any],
        *,
        conversations: list[dict[str, Any]],
    ) -> tuple[str, str, frozenset[str], dict[str, str]]:
        template = await self._wait_for_template()
        source = self._identity_for_template(conversations, template.url)
        if source is None:
            raise LinkedInScraperException(
                "Could not correlate LinkedIn's passive message request."
            )
        identity_key, source_urn = source
        target_urn = target.get(identity_key)
        if not isinstance(target_urn, str):
            raise LinkedInScraperException(
                "The requested conversation has no compatible provider identity."
            )
        target_url = replace_conversation_urn(template.url, source_urn, target_urn)
        if target_url is None:
            raise LinkedInScraperException(
                "Could not target LinkedIn's passive message request safely."
            )
        target_urns = frozenset(
            value
            for key in ("entityUrn", "backendUrn")
            if isinstance((value := target.get(key)), str)
        )
        return target_url, target_urn, target_urns, template.headers

    async def _request_json(self, url: str, headers: dict[str, str]) -> dict[str, Any]:
        result = await self._page.evaluate(
            """async ({ url, headers }) => {
                try {
                    const response = await fetch(url, {
                        method: 'GET',
                        credentials: 'same-origin',
                        headers,
                    });
                    let payload = null;
                    try { payload = await response.json(); } catch (_) {}
                    return { status: response.status, payload };
                } catch (_) {
                    return { status: 0, payload: null };
                }
            }""",
            {"url": url, "headers": headers},
        )
        return result if isinstance(result, dict) else {}

    async def fetch_message_history(
        self,
        target: dict[str, Any],
        *,
        conversations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Read the current batch plus up to three bounded older batches."""
        initial = await self.fetch_messages(target, conversations=conversations)
        pages = [message_elements(initial)]
        if len(pages[0]) < _MESSAGE_PAGE_SIZE:
            return pages[0]

        if self._history_template is None:
            return pages[0]
        history_template = await self._wait_for_history_template()
        source = self._identity_for_template(conversations, history_template.url)
        if source is None:
            raise LinkedInScraperException(
                "Could not correlate LinkedIn's passive history request."
            )
        identity_key, source_urn = source
        target_urn = target.get(identity_key)
        if not isinstance(target_urn, str):
            raise LinkedInScraperException(
                "The requested conversation has no compatible history identity."
            )
        target_url = replace_conversation_urn(
            history_template.url, source_urn, target_urn
        )
        if target_url is None:
            raise LinkedInScraperException(
                "Could not target LinkedIn's passive history request safely."
            )
        target_urns = frozenset(
            value
            for key in ("entityUrn", "backendUrn")
            if isinstance((value := target.get(key)), str)
        )
        headers = history_template.headers
        oldest_seen: int | None = None
        for _ in range(_MAX_OLDER_PAGES):
            timestamps = [
                value
                for item in pages[-1]
                if isinstance((value := item.get("deliveredAt")), int)
            ]
            if not timestamps:
                break
            oldest = min(timestamps)
            if oldest_seen is not None and oldest >= oldest_seen:
                break
            oldest_seen = oldest
            page_url = build_message_page_url(
                target_url,
                target_urn,
                delivered_at=oldest,
                count_before=_MESSAGE_PAGE_SIZE,
            )
            if page_url is None:
                raise LinkedInScraperException(
                    "Could not build LinkedIn's passive history request safely."
                )
            result = await self._request_json(page_url, headers)
            status = result.get("status")
            payload = result.get("payload")
            if not isinstance(status, int) or not 200 <= status < 300:
                raise LinkedInScraperException(
                    "LinkedIn refused the passive conversation history read"
                    f" (HTTP {status if isinstance(status, int) else 'unknown'})."
                )
            if not _verified_message_batch(
                payload,
                target_urns,
                container_key="messengerMessagesByAnchorTimestamp",
                allow_empty=True,
            ):
                raise LinkedInScraperException(
                    "LinkedIn returned unverified conversation history data."
                )
            older = message_elements(payload)
            if not older:
                break
            pages.append(older)
            if len(older) < _MESSAGE_PAGE_SIZE:
                break

        return merge_message_elements(pages)

    async def _drain_pending(self) -> None:
        if not self._pending:
            return
        try:
            await asyncio.wait(self._pending, timeout=2.0)
        finally:
            for task in self._pending:
                if not task.done():
                    task.cancel()
            with anyio.CancelScope(shield=True):
                try:
                    await asyncio.wait(self._pending, timeout=1.0)
                finally:
                    for task in self._pending:
                        if task.done() and not task.cancelled():
                            task.exception()
        await anyio.lowlevel.checkpoint()
