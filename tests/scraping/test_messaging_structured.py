"""Offline proof of additive message facts and passive acquisition boundaries."""

from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from linkedin_mcp_server.scraping import messaging_payload as payload
from linkedin_mcp_server.scraping.messaging_api import MessagingApiCapture
from scraping.test_conversations import (
    FakeCapture,
    _conversation,
    _inbox,
    _member,
    _messages,
    _reader,
)
from scraping.test_messaging_api import _history_request, _request


def _history(reason="short_batch", pages=1, count=1):
    return {
        "scope": "bounded_thread",
        "exhaustion": "unknown",
        "stopReason": reason,
        "pagesFetched": pages,
        "messageCount": count,
        "pageSize": 20,
        "maxOlderPages": 3,
    }


def test_provider_facts_are_additive_exact_and_not_attachment_participants():
    target = _conversation("2-target")
    target["conversationParticipants"][0]["entityUrn"] = "urn:li:msg_participant:ada"
    messages = [
        {
            "entityUrn": "urn:li:msg_message:one",
            "deliveredAt": 1792418400123,
            "body": {"text": "  Exact text\n"},
            "sender": _member("ada", "Ada", "Lovelace"),
            "attachments": [{"profileUrl": "https://www.linkedin.com/in/stranger/"}],
        }
    ]
    before = deepcopy((target, messages))
    result = payload.build_structured_conversation(target, messages, history=_history())
    assert result["id"] == "2-target"
    assert result["participants"] == [
        {
            "name": "Ada Lovelace",
            "profileUrl": "/in/ada/",
            "providerIdentity": "urn:li:msg_participant:ada",
        },
        {"name": "Current User", "profileUrl": "/in/viewer/"},
    ]
    assert result["messages"] == [
        {
            "id": "urn:li:msg_message:one",
            "sentAt": "2026-10-19T14:00:00.123+00:00",
            "body": "  Exact text\n",
            "sender": {"name": "Ada Lovelace", "profileUrl": "/in/ada/"},
        }
    ]
    assert result["timestampSource"] == "deliveredAt_unix_ms"
    assert result["history"] == _history()
    assert (target, messages) == before


@pytest.mark.parametrize(
    "value", [None, True, False, "1792418400123", 1.5, -1, 10**100]
)
def test_bad_timestamps_are_omitted_from_messages_and_inbox(value):
    target = _conversation("2-target")
    target["lastActivityAt"] = value
    result = payload.build_structured_conversation(
        target, [{"deliveredAt": value}], history=_history()
    )
    assert result["messages"] == [{}]
    assert (
        "last_activity_at"
        not in payload.build_conversation_references(
            [_inbox(target)], limit=1, context="inbox"
        )[0]
    )


def test_unknown_facts_and_event_fallback_are_never_promoted_to_message_identity():
    target = _conversation("2-target", title="Group title")
    target["conversationParticipants"] = [
        None,
        {"participantType": {"organization": {"name": "Org"}}},
    ]
    result = payload.build_structured_conversation(
        target,
        [{"entityUrn": "", "renderContentFallbackText": {"text": "Attachment"}}],
        history=_history(),
    )
    assert result["participants"] == [{}, {}]
    assert result["messages"] == [{}]
    assert (
        payload.format_message_elements(
            [{"renderContentFallbackText": {"text": "Attachment"}}]
        )
        == "Attachment"
    )


def test_provider_id_alternatives_are_used_without_content_hashes():
    result = payload.build_structured_conversation(
        _conversation("2-target"),
        [
            {"backendUrn": "urn:message:b"},
            {"messageUrn": "urn:message:c"},
            {"body": {"text": "same"}},
            {"body": {"text": "same"}},
        ],
        history=_history(count=4),
    )
    assert result["messages"] == [
        {"id": "urn:message:b"},
        {"id": "urn:message:c"},
        {"body": "same"},
        {"body": "same"},
    ]


def test_inbox_exact_activity_timestamp_preserves_reference_contract():
    target = _conversation("2-target")
    target["lastActivityAt"] = 1792418400123
    reference = payload.build_conversation_references(
        [_inbox(target)], limit=1, context="inbox"
    )[0]
    assert reference == {
        "kind": "conversation",
        "url": "/messaging/thread/2-target/",
        "text": "Ada Lovelace",
        "context": "inbox",
        "last_activity_at": "2026-10-19T14:00:00.123+00:00",
    }


async def test_reader_preserves_rendered_text_and_exposes_envelope(mock_page):

    target = _conversation("2-target")
    capture = FakeCapture([_inbox(target)], messages=_messages("Hello"))
    capture.history = _history()
    reader = _reader(mock_page)

    async def load(*args, **kwargs):
        await capture.wait_for_payload(after=0)
        return 0

    with (
        patch(
            "linkedin_mcp_server.scraping.conversations.MessagingApiCapture",
            return_value=capture,
        ),
        patch.object(reader, "_load_messaging_page", side_effect=load),
        patch.object(reader._navigator, "_navigate_to_page", new_callable=AsyncMock),
    ):
        result = await reader.get_conversation(thread_id="2-target")
    assert result["conversation"]["id"] == "2-target"
    assert result["conversation"]["messages"][0]["body"] == "Hello"
    assert "Hello" in result["sections"]["conversation"]
    assert result["references"]["conversation"][0]["url"] == "/in/ada/"
    mock_page.locator.return_value.click.assert_not_called()


@pytest.mark.parametrize(
    "case,reason,page_count",
    [
        ("short", "short_batch", 1),
        ("no_template", "history_template_unavailable", 1),
        ("no_timestamp", "no_timestamp", 1),
        ("empty", "empty_batch", 2),
        ("short_older", "short_batch", 2),
        ("overlap", "no_progress", 2),
        ("cap", "page_limit", 4),
    ],
)
async def test_history_reports_observed_stop_without_claiming_exhaustion(
    mock_page, case, reason, page_count
):
    target = _conversation("2-target")

    def message(n):
        return {
            "entityUrn": f"urn:message:{n}",
            "deliveredAt": n,
            "backendConversationUrn": target["entityUrn"],
        }

    initial = [message(n) for n in range(100, 120)]
    if case == "short":
        initial = initial[:1]
    if case == "no_timestamp":
        for item in initial:
            item.pop("deliveredAt")
    batches = [initial]
    if case == "empty":
        batches.append([])
    if case == "short_older":
        batches.append([message(99)])
    if case == "overlap":
        batches.append(initial)
    if case == "cap":
        batches.extend(
            [[message(n) for n in range(start, start + 20)] for start in (80, 60, 40)]
        )
    mock_page.evaluate = AsyncMock(
        side_effect=[
            {
                "status": 200,
                "payload": {
                    "data": {
                        (
                            "messengerMessagesBySyncToken"
                            if i == 0
                            else "messengerMessagesByAnchorTimestamp"
                        ): {"elements": items}
                    }
                },
            }
            for i, items in enumerate(batches)
        ]
    )
    async with MessagingApiCapture(mock_page) as capture:
        capture._on_request(_request(target))
        if case != "no_template":
            capture._on_request(_history_request(target))
        await capture.settle()
        messages = await capture.fetch_message_history(target, conversations=[target])
    assert capture.history == _history(reason, page_count, len(messages))
    assert mock_page.evaluate.await_count == page_count


@pytest.mark.browser_dom
async def test_real_browser_passive_fetch_and_read_state_guard():
    """Execute production fetch JS with all traffic fulfilled locally, no account."""
    from patchright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel="chromium", headless=True)
        try:
            page = await browser.new_page(service_workers="block")
            received = []

            async def serve(route):
                received.append((route.request.method, route.request.url))
                if route.request.resource_type == "document":
                    await route.fulfill(
                        content_type="text/html", body="<main>Offline</main>"
                    )
                else:
                    await route.fulfill(json={"data": "synthetic"})

            await page.route("**/*", serve)
            await page.goto("https://www.linkedin.com/messaging/")
            reader = _reader(page)
            async with reader._without_read_state_writes():
                for endpoint in (
                    "voyagerMessagingDashMessengerConversations",
                    "voyagerMessagingDashMessagingBadge",
                ):
                    url = "https://www.linkedin.com/voyager/api/" + endpoint
                    assert (
                        await page.evaluate(
                            "async url => {try {await fetch(url,{method:'POST'}); return 'wrote'} catch (_) {return 'blocked'}}",
                            url,
                        )
                        == "blocked"
                    )
                    assert (await MessagingApiCapture(page)._request_json(url, {}))[
                        "payload"
                    ] == {"data": "synthetic"}
                # Production replay JavaScript, not a mocked page.evaluate.
                assert (
                    await MessagingApiCapture(page)._request_json(
                        "https://www.linkedin.com/voyager/api/graphql?queryId=messengerMessages.test",
                        {},
                    )
                )["status"] == 200
            assert not any(method == "POST" for method, _ in received)
            # Only verified write intent releases the page's read latch.
            from linkedin_mcp_server.scraping.messaging_read_state import (
                allow_explicit_message_write,
            )

            allow_explicit_message_write(page)
            await page.evaluate(
                "fetch('/voyager/api/voyagerMessagingDashMessagingBadge', {method:'POST'})"
            )
            assert received[-1][0] == "POST"
        finally:
            await browser.close()


def test_unknown_identity_does_not_collapse_distinct_identical_messages():
    first = {"deliveredAt": 10, "body": {"text": "Same"}}
    second = deepcopy(first)
    assert payload.merge_message_elements([[first, second]]) == [first, second]


def test_activity_fact_survives_richer_rendered_reference():
    from linkedin_mcp_server.scraping.link_metadata import dedupe_references

    item = _conversation("2-target")
    item["lastActivityAt"] = 1792418400123
    refs = payload.build_conversation_references(
        [_inbox(item)], limit=1, context="inbox"
    )
    merged = dedupe_references(
        refs
        + [
            {
                "kind": "conversation",
                "url": refs[0]["url"],
                "text": "A longer rendered title",
                "context": "inbox",
            }
        ]
    )
    assert merged[0]["last_activity_at"] == "2026-10-19T14:00:00.123+00:00"


def test_unsupported_or_name_only_participants_preserve_membership_uncertainty():
    target = _conversation("2-target")
    target["conversationParticipants"] += [
        {"participantType": {"organization": {"name": "Org"}}},
        {"participantType": {"member": {"firstName": {"text": "Unknown"}}}},
    ]
    result = payload.build_structured_conversation(
        target, [], history=_history(count=0)
    )
    assert len(result["participants"]) == 4
    assert result["participantCoverage"] == {
        "status": "partial",
        "sourceCount": 4,
        "unresolvedCount": 2,
    }
    target.pop("conversationParticipants")
    result = payload.build_structured_conversation(
        target, [], history=_history(count=0)
    )
    assert result["participantCoverage"] == {"status": "unavailable"}


@pytest.mark.parametrize(
    "workflow", ["get_inbox", "get_conversation", "search_conversations"]
)
async def test_guard_precedes_navigation_and_cleans_up_on_failure(mock_page, workflow):
    reader = _reader(mock_page)
    events = []

    async def route(pattern, handler):
        events.append(("add", pattern, handler))

    async def unroute(pattern, handler):
        events.append(("remove", pattern, handler))

    async def fail(*args, **kwargs):
        assert [event[0] for event in events] == ["add", "add"]
        raise RuntimeError("offline navigation failed")

    mock_page.route = AsyncMock(side_effect=route)
    mock_page.unroute = AsyncMock(side_effect=unroute)
    with patch.object(reader, "_load_messaging_page", side_effect=fail):
        with pytest.raises(RuntimeError, match="offline navigation failed"):
            kwargs = (
                {"thread_id": "2-target"}
                if workflow == "get_conversation"
                else {"keywords": "example"}
                if workflow == "search_conversations"
                else {}
            )
            await getattr(reader, workflow)(**kwargs)
    assert [event[0] for event in events] == ["add", "add"]
    mock_page.unroute.assert_not_awaited()


async def test_partial_guard_registration_fails_closed_without_duplicates(mock_page):
    reader = _reader(mock_page)
    mock_page.route = AsyncMock(side_effect=[None, RuntimeError("registration failed")])
    with pytest.raises(RuntimeError, match="registration failed"):
        async with reader._without_read_state_writes():
            pytest.fail("must not enter partially guarded workflow")
    mock_page.unroute.assert_not_awaited()
    from linkedin_mcp_server.core.exceptions import LinkedInScraperException

    with pytest.raises(
        LinkedInScraperException, match="protection registration failed"
    ):
        async with reader._without_read_state_writes():
            pytest.fail("must not guess whether registration succeeded")
    assert mock_page.route.await_count == 2


async def test_guard_exit_does_not_wait_for_interception_reconfiguration(mock_page):
    import asyncio

    reader = _reader(mock_page)

    async def stalled_unroute(*args, **kwargs):
        await asyncio.Event().wait()

    mock_page.unroute = AsyncMock(side_effect=stalled_unroute)

    async def read():
        async with reader._without_read_state_writes():
            pass

    await asyncio.wait_for(read(), timeout=0.1)
    mock_page.unroute.assert_not_awaited()


async def test_page_scoped_guard_is_reused_and_preserves_unrelated_write_scope(
    mock_page,
):
    from types import SimpleNamespace

    first = _reader(mock_page)
    for _ in range(3):
        # Tool calls create new readers, sharing only their actual page.
        async with _reader(mock_page)._without_read_state_writes():
            pass
    assert mock_page.route.await_count == 2
    mock_page.unroute.assert_not_awaited()
    handler = mock_page.route.await_args_list[0].args[1]

    def request(method="POST"):
        return SimpleNamespace(
            method=method,
            url="https://www.linkedin.com/voyager/api/voyagerMessagingDashMessengerConversations",
        )

    def route(req):
        return SimpleNamespace(request=req, abort=AsyncMock(), fallback=AsyncMock())

    from linkedin_mcp_server.scraping.messaging_read_state import (
        allow_explicit_message_write,
    )

    allow_explicit_message_write(mock_page)
    outside = route(request())
    await handler(outside)
    outside.fallback.assert_awaited_once()
    outside.abort.assert_not_awaited()
    async with first._without_read_state_writes():
        post = route(request())
        await handler(post)
        post.abort.assert_awaited_once()
        post.fallback.assert_not_awaited()
        get = route(request("GET"))
        await handler(get)
        get.fallback.assert_awaited_once()
        get.abort.assert_not_awaited()
    allow_explicit_message_write(mock_page)
    other = type(mock_page)()
    other.route = AsyncMock()
    other.on = lambda *args: None
    async with _reader(other)._without_read_state_writes():
        outside = route(request())
        await handler(outside)
        outside.fallback.assert_awaited_once()
    assert other.route.await_count == 2


async def test_request_observed_during_read_remains_protected_after_scope(mock_page):
    from types import SimpleNamespace

    class Request:
        method = "POST"
        url = "https://www.linkedin.com/voyager/api/voyagerMessagingDashMessagingBadge"

    req = Request()
    async with _reader(mock_page)._without_read_state_writes():
        for callback in mock_page.listeners["request"]:
            callback(req)
    from linkedin_mcp_server.scraping.messaging_read_state import (
        allow_explicit_message_write,
    )

    allow_explicit_message_write(mock_page)
    handler = mock_page.route.await_args_list[0].args[1]
    late = SimpleNamespace(request=req, abort=AsyncMock(), fallback=AsyncMock())
    await handler(late)
    late.abort.assert_awaited_once()
    late.fallback.assert_not_awaited()


async def test_guard_nesting_and_cancellation_do_not_disable_an_outer_read(mock_page):
    import asyncio
    from types import SimpleNamespace

    class Request:
        method = "POST"
        url = "https://www.linkedin.com/voyager/api/voyagerMessagingDashMessagingBadge"

    async with _reader(mock_page)._without_read_state_writes():
        handler = mock_page.route.await_args_list[0].args[1]
        entered = asyncio.Event()

        async def inner():
            async with _reader(mock_page)._without_read_state_writes():
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(inner())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        post = SimpleNamespace(
            request=Request(), abort=AsyncMock(), fallback=AsyncMock()
        )
        await handler(post)
        post.abort.assert_awaited_once()
    assert mock_page.route.await_count == 2
    mock_page.unroute.assert_not_awaited()


async def test_cancelled_registration_is_reused_without_duplicate_handlers(mock_page):
    import asyncio

    entered = asyncio.Event()
    release = asyncio.Event()

    async def register(*args):
        entered.set()
        await release.wait()

    mock_page.route.side_effect = register

    async def read():
        async with _reader(mock_page)._without_read_state_writes():
            pass

    first = asyncio.create_task(read())
    await entered.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    await read()
    assert mock_page.route.await_count == 2


@pytest.mark.browser_dom
async def test_guard_exits_with_busy_iframe_and_context_routes_without_accumulation():
    import asyncio
    from patchright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel="chromium", headless=True)
        try:
            context = await browser.new_context(service_workers="block")
            await context.set_offline(True)
            writes = []

            async def serve(route):
                if route.request.method == "POST":
                    writes.append(route.request.url)
                if "frame.example.org" in route.request.url:
                    html = "<script>onmessage=()=>{parent.postMessage('ready','*');setTimeout(()=>{const end=Date.now()+3500;while(Date.now()<end){}},300)}</script>"
                else:
                    html = "<main>Offline</main><iframe src='https://frame.example.org/'></iframe><script>onmessage=e=>{if(e.data==='ready')document.body.dataset.ready='yes'}</script>"
                await route.fulfill(content_type="text/html", body=html)

            await context.route("**/*", serve)
            page: Any = await context.new_page()
            await page.goto("https://www.linkedin.com/messaging/")
            additions = []
            original = page.route

            async def register(pattern, handler):
                additions.append((pattern, handler))
                return await original(pattern, handler)

            page.route = register
            reader = _reader(page)
            for _ in range(3):
                async with reader._without_read_state_writes():
                    for endpoint in (
                        "voyagerMessagingDashMessengerConversations",
                        "voyagerMessagingDashMessagingBadge",
                    ):
                        path = "/voyager/api/" + endpoint
                        assert await page.evaluate(
                            "async path => {try {await fetch(path,{method:'POST'});return false;}catch(_){return true;}}",
                            path,
                        )
                        assert (
                            await page.evaluate(
                                "async path => (await fetch(path)).status", path
                            )
                            == 200
                        )
            guard = reader._without_read_state_writes()
            await guard.__aenter__()
            await page.evaluate(
                "document.querySelector('iframe').contentWindow.postMessage('go','*')"
            )
            await page.wait_for_function("document.body.dataset.ready==='yes'")
            await asyncio.sleep(0.5)
            # Last-page-route removal used to wait for this busy child frame.
            await asyncio.wait_for(guard.__aexit__(None, None, None), timeout=1)
            assert len(additions) == 2
            assert not writes
            # Existing verified compose intent can release the latch.
            from linkedin_mcp_server.scraping.messaging_read_state import (
                allow_explicit_message_write,
            )

            allow_explicit_message_write(page)
            await page.evaluate(
                "fetch('/voyager/api/voyagerMessagingDashMessagingBadge',{method:'POST'})"
            )
            assert len(writes) == 1
            other_context = await browser.new_context(service_workers="block")
            await other_context.set_offline(True)
            await other_context.route("**/*", serve)
            other_page = await other_context.new_page()
            await other_page.goto("https://www.linkedin.com/messaging/")
            async with _reader(other_page)._without_read_state_writes():
                assert await other_page.evaluate(
                    "async()=>{try{await fetch('/voyager/api/voyagerMessagingDashMessagingBadge',{method:'POST'});return false;}catch(_){return true;}}"
                )
                await page.evaluate(
                    "fetch('/voyager/api/voyagerMessagingDashMessagingBadge',{method:'POST'})"
                )
            assert len(writes) == 2
            assert len(additions) == 2
        finally:
            await browser.close()


async def test_unobserved_late_post_after_read_is_still_blocked(mock_page):
    from types import SimpleNamespace

    class Request:
        method = "POST"
        url = "https://www.linkedin.com/voyager/api/voyagerMessagingDashMessagingBadge"

    async with _reader(mock_page)._without_read_state_writes():
        pass
    handler = mock_page.route.await_args_list[0].args[1]
    late = SimpleNamespace(request=Request(), abort=AsyncMock(), fallback=AsyncMock())
    await handler(late)
    late.abort.assert_awaited_once()
    late.fallback.assert_not_awaited()


async def test_read_guard_active_while_route_registration_awaits(mock_page):
    import asyncio
    from types import SimpleNamespace

    entered = asyncio.Event()
    release = asyncio.Event()

    async def register(*args):
        entered.set()
        await release.wait()

    mock_page.route.side_effect = register

    async def read():
        async with _reader(mock_page)._without_read_state_writes():
            pass

    task = asyncio.create_task(read())
    await entered.wait()
    handler = mock_page.route.await_args_list[0].args[1]
    post = SimpleNamespace(
        request=SimpleNamespace(method="POST"), abort=AsyncMock(), fallback=AsyncMock()
    )
    try:
        await handler(post)
        post.abort.assert_awaited_once()
    finally:
        release.set()
        await task


@pytest.mark.parametrize("verified", [False, True])
async def test_only_verified_compose_navigation_releases_read_latch(
    mock_page, verified
):
    from types import SimpleNamespace
    from linkedin_mcp_server.scraping.session import ScrapingSession

    class Request:
        method = "POST"
        url = "https://www.linkedin.com/voyager/api/voyagerMessagingDashMessengerConversations"

    async with _reader(mock_page)._without_read_state_writes():
        pass
    handler = mock_page.route.await_args_list[0].args[1]
    from linkedin_mcp_server.scraping.message_sender import MessageSender
    from linkedin_mcp_server.scraping.navigation import PageNavigator

    session = ScrapingSession(mock_page)
    sender = MessageSender(session, PageNavigator(session))
    compose = "https://www.linkedin.com/messaging/compose/?profileUrn=ACo" + "A" * 24
    target = SimpleNamespace(profile_urn="ACo" + "A" * 24, compose_url=compose)

    async def navigate(url):
        if url == compose:
            post = SimpleNamespace(
                request=Request(), abort=AsyncMock(), fallback=AsyncMock()
            )
            await handler(post)
            post.fallback.assert_awaited_once()
            post.abort.assert_not_awaited()
            raise RuntimeError("stop before composing")

    with (
        patch.object(sender._navigator, "_navigate_to_page", side_effect=navigate),
        patch.object(ScrapingSession, "check_rate_limit", new_callable=AsyncMock),
        patch.object(
            sender,
            "_read_profile_message_target",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(status="resolved", target=target),
        ),
    ):
        if verified:
            with pytest.raises(RuntimeError, match="stop before composing"):
                await sender.send_message(
                    "synthetic-peer",
                    "Synthetic body",
                    confirm_send=False,
                    profile_urn=target.profile_urn,
                )
        else:
            result = await sender.send_message(
                "synthetic-peer",
                "Synthetic body",
                confirm_send=False,
                profile_urn="ACo" + "B" * 24,
            )
            assert result["status"] == "recipient_resolution_failed"
            post = SimpleNamespace(
                request=Request(), abort=AsyncMock(), fallback=AsyncMock()
            )
            await handler(post)
            post.abort.assert_awaited_once()


async def test_explicit_write_cannot_release_an_active_passive_read(mock_page):
    from linkedin_mcp_server.core.exceptions import LinkedInScraperException
    from linkedin_mcp_server.scraping.messaging_read_state import (
        allow_explicit_message_write,
    )

    async with _reader(mock_page)._without_read_state_writes():
        with pytest.raises(LinkedInScraperException, match="passive messaging read"):
            allow_explicit_message_write(mock_page)


async def test_page_guard_does_not_retain_closed_page_objects():
    import gc
    import weakref
    from linkedin_mcp_server.scraping.messaging_read_state import (
        without_read_state_writes,
    )

    class Page:
        def __init__(self):
            self.routes = []
            self.listeners = []

        async def route(self, pattern, handler):
            self.routes.append(handler)

        def on(self, event, callback):
            self.listeners.append(callback)

    page = Page()
    async with without_read_state_writes(page):
        pass
    reference = weakref.ref(page)
    del page
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("failed_route", [1, 2])
async def test_failed_registration_does_not_retain_the_page(failed_route):
    import asyncio
    import gc
    import weakref
    from linkedin_mcp_server.core.exceptions import LinkedInScraperException
    from linkedin_mcp_server.scraping.messaging_read_state import (
        without_read_state_writes,
    )

    class Page:
        def __init__(self):
            self.routes = []
            self.listeners = []

        async def route(self, pattern, handler):
            self.routes.append(handler)
            if len(self.routes) == failed_route:
                raise RuntimeError("registration failed")

        def on(self, event, callback):
            self.listeners.append(callback)

    page = Page()
    with pytest.raises(RuntimeError, match="registration failed"):
        async with without_read_state_writes(page):
            pytest.fail("must not enter a partially guarded read")
    with pytest.raises(
        LinkedInScraperException, match="protection registration failed"
    ):
        async with without_read_state_writes(page):
            pytest.fail("a failed registration must remain fail-closed")
    assert len(page.routes) == failed_route
    reference = weakref.ref(page)
    del page
    await asyncio.sleep(0)
    gc.collect()
    assert reference() is None
