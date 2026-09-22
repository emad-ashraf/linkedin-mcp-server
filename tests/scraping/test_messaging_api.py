"""Network-boundary tests for passive LinkedIn messaging capture."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import quote

import pytest

from linkedin_mcp_server.core.exceptions import LinkedInScraperException
from linkedin_mcp_server.pacing import JobStore, read_account_cooldown
from linkedin_mcp_server.scraping.messaging_api import MessagingApiCapture


def _conversation(thread_id: str) -> dict:
    return {
        "entityUrn": f"urn:li:msg_conversation:({thread_id})",
        "backendUrn": f"urn:li:messagingThread:{thread_id}",
        "conversationUrl": f"https://www.linkedin.com/messaging/thread/{thread_id}/",
    }


def _inbox(*items: dict) -> dict:
    return {
        "data": {"messengerConversationsByCategoryQuery": {"elements": list(items)}}
    }


def _request(source: dict) -> SimpleNamespace:
    urn = source["entityUrn"]
    return SimpleNamespace(
        method="GET",
        url=(
            "https://www.linkedin.com/voyager/api/graphql?"
            f"variables=(conversationUrn%3A{quote(urn, safe='')})&"
            "queryId=messengerMessages.hash"
        ),
        all_headers=AsyncMock(
            return_value={
                "accept": "application/vnd.linkedin.normalized+json+2.1",
                "csrf-token": "private-token",
                "cookie": "must-not-be-forwarded",
                "x-restli-protocol-version": "2.0.0",
            }
        ),
    )


def _history_request(source: dict) -> SimpleNamespace:
    urn = source["entityUrn"]
    return SimpleNamespace(
        method="GET",
        url=(
            "https://www.linkedin.com/voyager/api/graphql?"
            "variables=(deliveredAt%3A123%2C"
            f"conversationUrn%3A{quote(urn, safe='')}%2C"
            "countBefore%3A20%2CcountAfter%3A0)&"
            "queryId=messengerMessages.history-hash"
        ),
        all_headers=AsyncMock(
            return_value={
                "accept": "application/vnd.linkedin.normalized+json+2.1",
                "csrf-token": "private-token",
                "cookie": "must-not-be-forwarded",
            }
        ),
    )


async def test_capture_collects_payload_and_only_safe_replay_headers(mock_page):
    source = _conversation("2-source")
    response = SimpleNamespace(
        url=(
            "https://www.linkedin.com/voyager/api/graphql?"
            "queryId=messengerConversations.hash"
        ),
        json=AsyncMock(return_value=_inbox(source)),
    )
    request = _request(source)

    async with MessagingApiCapture(mock_page) as capture:
        for callback in mock_page.listeners["response"]:
            callback(response)
        for callback in mock_page.listeners["request"]:
            callback(request)
        await capture.wait_for_payload()
        await capture.settle()

    assert capture.conversations_since() == [source]
    assert capture._request_template is not None
    assert capture._request_template.headers == {
        "accept": "application/vnd.linkedin.normalized+json+2.1",
        "csrf-token": "private-token",
        "x-restli-protocol-version": "2.0.0",
    }
    assert mock_page.listeners["response"] == []
    assert mock_page.listeners["request"] == []


async def test_fetch_rewrites_exact_target_and_verifies_response_identity(mock_page):
    source = _conversation("2-source")
    target = _conversation("2-target")
    request = _request(source)
    mock_page.evaluate = AsyncMock(
        return_value={
            "status": 200,
            "payload": {
                "data": {
                    "messengerMessagesBySyncToken": {
                        "elements": [
                            {
                                "backendConversationUrn": target["entityUrn"],
                                "body": {"text": "private"},
                            }
                        ]
                    }
                }
            },
        }
    )

    async with MessagingApiCapture(mock_page) as capture:
        for callback in mock_page.listeners["request"]:
            callback(request)
        await capture.settle()
        result = await capture.fetch_messages(target, conversations=[source, target])

    assert result["data"]["messengerMessagesBySyncToken"]["elements"]
    await_args = mock_page.evaluate.await_args
    assert await_args is not None
    argument = await_args.args[1]
    assert quote(source["entityUrn"], safe="") not in argument["url"]
    assert quote(target["entityUrn"], safe="") in argument["url"]
    assert "cookie" not in argument["headers"]


@pytest.mark.parametrize(
    "result",
    [
        {"status": 429, "payload": None},
        {"status": 200, "payload": {"data": {}}},
        {"status": 200, "payload": "not-json"},
    ],
)
async def test_fetch_fails_closed_on_refusal_or_unverified_data(mock_page, result):
    source = _conversation("2-source")
    target = _conversation("2-target")
    request = _request(source)
    mock_page.evaluate = AsyncMock(return_value=result)

    async with MessagingApiCapture(mock_page) as capture:
        for callback in mock_page.listeners["request"]:
            callback(request)
        await capture.settle()
        with pytest.raises(LinkedInScraperException):
            await capture.fetch_messages(target, conversations=[source, target])


@pytest.mark.parametrize(
    "payload",
    [
        {
            "data": {
                "messengerMessagesBySyncToken": {"elements": []},
                "unrelatedMetadata": {
                    "backendConversationUrn": "urn:li:msg_conversation:(2-target)"
                },
            }
        },
        {
            "data": {
                "messengerMessagesBySyncToken": {
                    "elements": [
                        {
                            "backendConversationUrn": (
                                "urn:li:msg_conversation:(2-target)"
                            )
                        },
                        {
                            "backendConversationUrn": (
                                "urn:li:msg_conversation:(2-other)"
                            )
                        },
                    ]
                }
            }
        },
        {
            "data": {
                "messengerMessagesBySyncToken": {
                    "elements": [{"body": {"text": "missing identity"}}]
                }
            }
        },
    ],
)
async def test_fetch_rejects_empty_mixed_or_unidentified_message_batches(
    mock_page, payload
):
    source = _conversation("2-source")
    target = _conversation("2-target")
    mock_page.evaluate = AsyncMock(return_value={"status": 200, "payload": payload})

    async with MessagingApiCapture(mock_page) as capture:
        for callback in mock_page.listeners["request"]:
            callback(_request(source))
        await capture.settle()
        with pytest.raises(LinkedInScraperException, match="unverified"):
            await capture.fetch_messages(target, conversations=[source, target])


async def test_history_fetches_older_page_only_when_initial_batch_is_full(mock_page):
    source = _conversation("2-source")
    target = _conversation("2-target")

    def message(number: int) -> dict:
        return {
            "entityUrn": f"urn:message:{number}",
            "backendConversationUrn": target["backendUrn"],
            "deliveredAt": number,
            "body": {"text": str(number)},
        }

    initial = {
        "data": {
            "messengerMessagesBySyncToken": {
                "elements": [message(number) for number in range(20, 40)]
            }
        }
    }
    older = {
        "data": {
            "messengerMessagesByAnchorTimestamp": {
                "elements": [message(19), message(20)]
            }
        }
    }
    mock_page.evaluate = AsyncMock(
        side_effect=[
            {"status": 200, "payload": initial},
            {"status": 200, "payload": older},
        ]
    )

    async with MessagingApiCapture(mock_page) as capture:
        for callback in mock_page.listeners["request"]:
            callback(_request(source))
            callback(_history_request(source))
        await capture.settle()
        messages = await capture.fetch_message_history(
            target, conversations=[source, target]
        )

    assert [item["deliveredAt"] for item in messages] == list(range(19, 40))
    assert mock_page.evaluate.await_count == 2
    older_url = mock_page.evaluate.await_args_list[1].args[1]["url"]
    assert "queryId=messengerMessages.history-hash" in older_url
    assert "countBefore%3A20" in older_url
    assert "deliveredAt%3A20" in older_url


async def test_history_stops_after_short_initial_batch(mock_page):
    source = _conversation("2-source")
    target = _conversation("2-target")
    payload = {
        "data": {
            "messengerMessagesBySyncToken": {
                "elements": [
                    {
                        "backendConversationUrn": target["backendUrn"],
                        "deliveredAt": 1,
                        "body": {"text": "only"},
                    }
                ]
            }
        }
    }
    mock_page.evaluate = AsyncMock(return_value={"status": 200, "payload": payload})

    async with MessagingApiCapture(mock_page) as capture:
        for callback in mock_page.listeners["request"]:
            callback(_request(source))
        await capture.settle()
        messages = await capture.fetch_message_history(
            target, conversations=[source, target]
        )

    assert len(messages) == 1
    assert mock_page.evaluate.await_count == 1


async def test_full_batch_does_not_induce_history_discovery(mock_page):
    source = _conversation("2-source")
    target = _conversation("2-target")
    elements = [
        {
            "backendConversationUrn": target["backendUrn"],
            "deliveredAt": number,
            "body": {"text": str(number)},
        }
        for number in range(20)
    ]
    payload = {"data": {"messengerMessagesBySyncToken": {"elements": elements}}}
    mock_page.evaluate = AsyncMock(return_value={"status": 200, "payload": payload})

    async with MessagingApiCapture(mock_page) as capture:
        for callback in mock_page.listeners["request"]:
            callback(_request(source))
        await capture.settle()
        messages = await capture.fetch_message_history(
            target, conversations=[source, target]
        )

    assert messages == elements
    assert mock_page.evaluate.await_count == 1


async def test_a_payload_that_never_arrives_is_half_a_throttle_signal(
    mock_page, tmp_path
):
    """The retry storm in issue #57 was this timeout, retried at once.

    One is ambiguous with a slow proxy, so it arms rather than pauses; the
    second inside ten minutes is the strike.
    """
    store = JobStore(tmp_path / "jobs")

    async with MessagingApiCapture(mock_page) as capture:
        with pytest.raises(LinkedInScraperException, match="did not return"):
            await capture.wait_for_payload(timeout=0.01)
        assert read_account_cooldown(store).half_at is not None
        assert read_account_cooldown(store).until is None

        with pytest.raises(LinkedInScraperException, match="did not return"):
            await capture.wait_for_payload(timeout=0.01)

    cooldown = read_account_cooldown(store)
    assert cooldown.strikes == 1
    assert cooldown.until is not None
    assert cooldown.last_signal is not None
    assert cooldown.last_signal["signal"] == "payload_timeout"
