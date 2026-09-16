"""Pure parser coverage for passive LinkedIn messaging payloads."""

from __future__ import annotations

from datetime import UTC, datetime

from linkedin_mcp_server.scraping.messaging_payload import (
    build_message_page_url,
    build_message_references,
    build_conversation_references,
    conversation_elements,
    conversation_matches_username,
    find_conversation_by_thread_id,
    format_message_elements,
    message_elements,
    merge_message_elements,
    replace_conversation_urn,
)


def _member(username: str, first: str, last: str) -> dict:
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
    title: str,
    username: str,
    read: bool,
) -> dict:
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
        "unreadCount": 0 if read else 2,
    }


def _inbox(*items: dict) -> dict:
    return {
        "data": {
            "messengerConversationsByCategoryQuery": {
                "elements": list(items),
                "metadata": {"nextCursor": "private-cursor"},
            }
        }
    }


def test_conversation_parser_keeps_mixed_read_states_and_order():
    unread = _conversation("2-unread", title="Ada Lovelace", username="ada", read=False)
    read = _conversation("2-read", title="Grace Hopper", username="grace", read=True)

    assert conversation_elements(_inbox(unread, read)) == [unread, read]


def test_references_use_payload_urls_without_clicking_rows():
    first = _conversation("2-a", title="Ada Lovelace", username="ada", read=False)
    duplicate = _conversation("2-a", title="Stale title", username="ada", read=True)
    second = _conversation("2-b", title="Grace Hopper", username="grace", read=True)

    refs = build_conversation_references(
        [_inbox(first, duplicate), _inbox(second)], limit=2, context="inbox"
    )

    assert refs == [
        {
            "kind": "conversation",
            "url": "/messaging/thread/2-a/",
            "text": "Ada Lovelace",
            "context": "inbox",
        },
        {
            "kind": "conversation",
            "url": "/messaging/thread/2-b/",
            "text": "Grace Hopper",
            "context": "inbox",
        },
    ]


def test_reference_title_accepts_the_live_plain_string_shape():
    item = _conversation("2-a", title="Ada Lovelace", username="ada", read=False)
    item["title"] = "Ada Lovelace"

    refs = build_conversation_references([_inbox(item)], limit=1, context="inbox")

    assert refs[0]["text"] == "Ada Lovelace"


def test_reference_title_accepts_a_nested_text_union():
    item = _conversation("2-a", title="Ada Lovelace", username="ada", read=False)
    item["title"] = {"memberTitle": {"text": "Ada Lovelace"}}

    refs = build_conversation_references([_inbox(item)], limit=1, context="inbox")

    assert refs[0]["text"] == "Ada Lovelace"


def test_username_matching_uses_profile_url_not_duplicate_display_name():
    target = _conversation(
        "2-a", title="Alex Smith", username="target-alex", read=False
    )
    namesake = _conversation(
        "2-b", title="Alex Smith", username="other-alex", read=True
    )

    assert conversation_matches_username(target, "target-alex")
    assert not conversation_matches_username(namesake, "target-alex")


def test_thread_lookup_requires_the_exact_conversation_url_identity():
    target = _conversation("2-a", title="Ada Lovelace", username="ada", read=False)
    other = _conversation("2-aa", title="Grace Hopper", username="grace", read=True)

    assert find_conversation_by_thread_id([other, target], "2-a") is target
    assert find_conversation_by_thread_id([other], "2-a") is None


def test_message_parser_formats_sender_and_body_without_provider_chrome():
    payload = {
        "data": {
            "messengerMessagesBySyncToken": {
                "elements": [
                    {
                        "deliveredAt": 2,
                        "sender": _member("viewer", "Current", "User"),
                        "body": {"text": "Second message"},
                    },
                    {
                        "deliveredAt": 1,
                        "sender": _member("ada", "Ada", "Lovelace"),
                        "body": {"text": "First message"},
                    },
                ]
            }
        }
    }

    elements = message_elements(payload)

    assert format_message_elements(elements) == (
        "Ada Lovelace\nFirst message\n\nCurrent User\nSecond message"
    )


def test_message_parser_can_preserve_the_rendered_conversation_contract():
    first_at = int(datetime(2026, 9, 16, 8, 30, tzinfo=UTC).timestamp() * 1000)
    second_at = int(datetime(2026, 9, 16, 9, 15, tzinfo=UTC).timestamp() * 1000)
    elements = [
        {
            "deliveredAt": second_at,
            "sender": _member("viewer", "Current", "User"),
            "body": {"text": "Second message"},
        },
        {
            "deliveredAt": first_at,
            "sender": _member("ada", "Ada", "Lovelace"),
            "body": {"text": "First message"},
        },
    ]

    assert format_message_elements(elements, timezone=UTC) == (
        "Sep 16, 2026\n"
        "Ada Lovelace sent the following message at 8:30 AM\n"
        "View Ada Lovelace's profile\n"
        "Ada Lovelace 8:30 AM\n"
        "First message\n\n"
        "Current User sent the following message at 9:15 AM\n"
        "View Current User's profile\n"
        "Current User 9:15 AM\n"
        "Second message"
    )


def test_message_parser_keeps_provider_event_fallback_text():
    event = {
        "deliveredAt": 1,
        "sender": _member("ada", "Ada", "Lovelace"),
        "body": None,
        "renderContentFallbackText": {"text": "Shared an attachment"},
    }

    assert format_message_elements([event]) == "Ada Lovelace\nShared an attachment"


def test_request_rewrite_changes_only_the_exact_percent_encoded_urn():
    source = "urn:li:msg_conversation:(2-source==)"
    target = "urn:li:msg_conversation:(2-target==)"
    url = (
        "https://www.linkedin.com/voyager/api/graphql?"
        "variables=(conversationUrn%3Aurn%3Ali%3Amsg_conversation%3A%282-source%3D%3D%29)"
        "&queryId=messengerMessages.hash"
    )

    rewritten = replace_conversation_urn(url, source, target)

    assert rewritten is not None
    assert "2-source" not in rewritten
    assert "2-target%3D%3D" in rewritten
    assert "queryId=messengerMessages.hash" in rewritten


def test_request_rewrite_fails_closed_when_source_identity_is_absent():
    assert (
        replace_conversation_urn(
            "https://www.linkedin.com/voyager/api/graphql?queryId=messengerMessages.hash",
            "urn:li:msg_conversation:(2-source)",
            "urn:li:msg_conversation:(2-target)",
        )
        is None
    )


def test_older_message_url_preserves_operation_and_uses_exact_anchor():
    source = "urn:li:msg_conversation:(2-target==)"
    url = (
        "https://www.linkedin.com/voyager/api/graphql?"
        "variables=(deliveredAt%3A999%2CconversationUrn%3A"
        "urn%3Ali%3Amsg_conversation%3A%282-target%3D%3D%29%2C"
        "countBefore%3A20%2CcountAfter%3A0)&"
        "queryId=messengerMessages.history-hash"
    )

    older = build_message_page_url(url, source, delivered_at=123456, count_before=20)

    assert older is not None
    assert "queryId=messengerMessages.history-hash" in older
    assert "conversationUrn%3Aurn%3Ali%3Amsg_conversation" in older
    assert "countBefore%3A20" in older
    assert "deliveredAt%3A123456" in older
    assert "countAfter%3A0" in older


def test_older_message_url_rejects_missing_or_duplicate_variables():
    base = "https://www.linkedin.com/voyager/api/graphql?queryId=messengerMessages.hash"
    assert build_message_page_url(base, "urn:target", delivered_at=1) is None
    assert (
        build_message_page_url(
            f"{base}&variables=(deliveredAt:1,conversationUrn:urn:target,"
            "countBefore:20,countAfter:0)&deliveredAt:2",
            "urn:target",
            delivered_at=1,
        )
        is None
    )


def test_merge_message_pages_dedupes_anchor_overlap_and_orders_chronologically():
    first = {
        "entityUrn": "urn:message:1",
        "deliveredAt": 1,
        "body": {"text": "First"},
    }
    second = {
        "entityUrn": "urn:message:2",
        "deliveredAt": 2,
        "body": {"text": "Second"},
    }

    merged = merge_message_elements([[second], [first, second]])

    assert merged == [first, second]


def test_message_references_keep_participants_and_attachments():
    elements = [
        {
            "sender": _member("ada", "Ada", "Lovelace"),
            "attachments": [
                {"url": "https://example.com/private-file", "name": "Document"}
            ],
        }
    ]

    refs = build_message_references(elements)

    assert refs == [
        {
            "kind": "person",
            "url": "/in/ada/",
            "text": "Ada Lovelace",
            "context": "conversation",
        },
        {
            "kind": "external",
            "url": "https://example.com/private-file",
            "context": "conversation",
        },
    ]


def test_conversation_participants_supply_both_profiles_for_one_sided_messages():
    target = _conversation("2-a", title="Ada Lovelace", username="ada", read=False)

    assert build_message_references([], conversation=target) == [
        {
            "kind": "person",
            "url": "/in/ada/",
            "text": "Ada Lovelace",
            "context": "conversation",
        },
        {
            "kind": "person",
            "url": "/in/viewer/",
            "text": "Current User",
            "context": "conversation",
        },
    ]


def test_message_person_links_cannot_become_conversation_participants():
    target = _conversation("2-a", title="Ada Lovelace", username="ada", read=False)
    target["conversationParticipants"] = [_member("viewer", "Current", "User")]
    elements = [
        {
            "sender": _member("viewer", "Current", "User"),
            "body": {
                "text": "Linked profile",
                "url": "https://www.linkedin.com/in/grace/",
            },
        }
    ]

    assert build_message_references(elements, conversation=target) == [
        {
            "kind": "person",
            "url": "/in/viewer/",
            "text": "Current User",
            "context": "conversation",
        }
    ]
