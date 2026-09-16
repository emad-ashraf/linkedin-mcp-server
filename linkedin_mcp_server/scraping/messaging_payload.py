"""Locale-independent parsing for LinkedIn's passive messaging payloads."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, tzinfo
from typing import Any
from urllib.parse import quote, urlparse

import re

from linkedin_mcp_server.scraping.identifiers import (
    normalize_person_identifier,
    normalize_thread_id,
)
from linkedin_mcp_server.scraping.link_metadata import (
    Reference,
    classify_link,
    dedupe_references,
)

_MESSAGE_CONTAINER_KEYS = (
    "messengerMessagesBySyncToken",
    "messengerMessagesByAnchorTimestamp",
)
_DELIVERED_AT_RE = re.compile(r"(?P<prefix>deliveredAt(?:%3A|:))(?P<value>\d+)")
_COUNT_BEFORE_RE = re.compile(r"countBefore(?:%3A|:)(?P<value>\d+)")


def _mapping(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _nested_mapping(root: Any, *keys: str) -> dict[str, Any] | None:
    current = _mapping(root)
    for key in keys:
        if current is None:
            return None
        current = _mapping(current.get(key))
    return current


def _elements(container: dict[str, Any] | None) -> list[dict[str, Any]]:
    values = container.get("elements") if container is not None else None
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, dict)]


def conversation_elements(payload: Any) -> list[dict[str, Any]]:
    """Return conversation entities from a messengerConversations response."""
    return _elements(
        _nested_mapping(payload, "data", "messengerConversationsByCategoryQuery")
    )


def message_elements(payload: Any) -> list[dict[str, Any]]:
    """Return message entities from a passive messengerMessages response."""
    data = _nested_mapping(payload, "data")
    if data is None:
        return []
    for key in _MESSAGE_CONTAINER_KEYS:
        container = _mapping(data.get(key))
        if container is not None:
            return _elements(container)
    return []


def _rich_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    mapping = _mapping(value)
    if mapping is None:
        return ""
    text = mapping.get("text")
    if isinstance(text, str):
        return text.strip()
    for child in mapping.values():
        nested = _rich_text(child)
        if nested:
            return nested
    return ""


def conversation_thread_path(item: dict[str, Any]) -> str | None:
    raw_url = item.get("conversationUrl")
    if not isinstance(raw_url, str):
        return None
    parsed = urlparse(raw_url)
    path = parsed.path
    prefix = "/messaging/thread/"
    if not path.startswith(prefix):
        return None
    remainder = path[len(prefix) :].strip("/")
    if not remainder or "/" in remainder:
        return None
    try:
        thread_id = normalize_thread_id(remainder)
    except Exception:
        return None
    return f"/messaging/thread/{thread_id}/"


def _conversation_title(item: dict[str, Any]) -> str:
    return _rich_text(item.get("title"))


def build_conversation_references(
    payloads: Iterable[Any], *, limit: int | None, context: str
) -> list[Reference]:
    """Build stable thread references without selecting conversation rows."""
    references: list[Reference] = []
    seen: set[str] = set()
    for payload in payloads:
        for item in conversation_elements(payload):
            path = conversation_thread_path(item)
            if path is None or path in seen:
                continue
            seen.add(path)
            reference: Reference = {
                "kind": "conversation",
                "url": path,
                "context": context,
            }
            title = _conversation_title(item)
            if title:
                reference["text"] = title
            references.append(reference)
            if limit is not None and len(references) >= limit:
                return references
    return references


def _member_profile_urls(item: dict[str, Any]) -> Iterable[str]:
    participants = item.get("conversationParticipants")
    if not isinstance(participants, list):
        return
    for participant in participants:
        member = _nested_mapping(participant, "participantType", "member")
        profile_url = member.get("profileUrl") if member is not None else None
        if isinstance(profile_url, str):
            yield profile_url


def conversation_matches_username(item: dict[str, Any], username: str) -> bool:
    """Match a conversation through a participant's stable profile URL."""
    try:
        expected = normalize_person_identifier(username).casefold()
    except Exception:
        return False
    for profile_url in _member_profile_urls(item):
        try:
            actual = normalize_person_identifier(profile_url).casefold()
        except Exception:
            continue
        if actual == expected:
            return True
    return False


def find_conversation_by_thread_id(
    conversations: Iterable[dict[str, Any]], thread_id: str
) -> dict[str, Any] | None:
    """Return only the conversation whose URL carries the exact thread ID."""
    expected = f"/messaging/thread/{normalize_thread_id(thread_id)}/"
    return next(
        (item for item in conversations if conversation_thread_path(item) == expected),
        None,
    )


def replace_conversation_urn(
    request_url: str, source_urn: str, target_urn: str
) -> str | None:
    """Replace exactly one encoded or raw URN while preserving query syntax."""
    encoded_source = quote(source_urn, safe="")
    encoded_target = quote(target_urn, safe="")
    if request_url.count(encoded_source) == 1:
        return request_url.replace(encoded_source, encoded_target, 1)
    if request_url.count(source_urn) == 1:
        return request_url.replace(source_urn, target_urn, 1)
    return None


def build_message_page_url(
    request_url: str,
    conversation_urn: str,
    *,
    delivered_at: int,
    count_before: int = 20,
) -> str | None:
    """Retarget a captured message query to one bounded older-history page."""
    if not conversation_urn or delivered_at < 0 or count_before < 1:
        return None
    encoded_urn = quote(conversation_urn, safe="")
    identity_count = request_url.count(encoded_urn) + request_url.count(
        conversation_urn
    )
    anchors = list(_DELIVERED_AT_RE.finditer(request_url))
    counts = list(_COUNT_BEFORE_RE.finditer(request_url))
    if (
        identity_count != 1
        or len(anchors) != 1
        or len(counts) != 1
        or int(counts[0].group("value")) != count_before
    ):
        return None
    return _DELIVERED_AT_RE.sub(
        lambda match: f"{match.group('prefix')}{delivered_at}", request_url, count=1
    )


def _message_identity(item: dict[str, Any]) -> tuple[Any, ...]:
    for key in ("entityUrn", "backendUrn", "messageUrn"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return (key, value)
    return (
        "content",
        item.get("deliveredAt"),
        _participant_name(item.get("sender")),
        _rich_text(item.get("body")),
    )


def merge_message_elements(
    pages: Iterable[Iterable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge overlapping history pages and retain chronological order."""
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for page in pages:
        for item in page:
            unique.setdefault(_message_identity(item), item)
    return sorted(
        unique.values(),
        key=lambda item: (
            item.get("deliveredAt")
            if isinstance(item.get("deliveredAt"), int)
            else 2**63 - 1
        ),
    )


def _walk_mappings(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def build_message_references(
    elements: Iterable[dict[str, Any]],
    *,
    conversation: dict[str, Any] | None = None,
) -> list[Reference]:
    """Extract participant and attachment links from verified message data."""
    references: list[Reference] = []
    if conversation is not None:
        participants = conversation.get("conversationParticipants")
        if isinstance(participants, list):
            for participant in participants:
                member = _nested_mapping(participant, "participantType", "member")
                if member is None:
                    continue
                profile_url = member.get("profileUrl")
                if not isinstance(profile_url, str):
                    continue
                classified = classify_link(profile_url)
                if classified is None or classified[0] != "person":
                    continue
                name = " ".join(
                    part
                    for key in ("firstName", "lastName")
                    if (part := _rich_text(member.get(key)))
                )
                reference: Reference = {
                    "kind": "person",
                    "url": classified[1],
                    "context": "conversation",
                }
                if name:
                    reference["text"] = name
                references.append(reference)
    for item in elements:
        mappings = list(_walk_mappings(item))
        for mapping in mappings:
            profile_url = mapping.get("profileUrl")
            if not isinstance(profile_url, str):
                continue
            classified = classify_link(profile_url)
            if classified is None:
                continue
            kind, url = classified
            if conversation is not None and kind == "person":
                continue
            reference: Reference = {
                "kind": kind,
                "url": url,
                "context": "conversation",
            }
            name = " ".join(
                part
                for part in (
                    _rich_text(mapping.get("firstName")),
                    _rich_text(mapping.get("lastName")),
                )
                if part
            )
            if name:
                reference["text"] = name
            references.append(reference)

        for mapping in mappings:
            for value in mapping.values():
                if not isinstance(value, str) or not value.startswith(
                    ("http://", "https://")
                ):
                    continue
                classified = classify_link(value)
                if classified is None:
                    continue
                kind, url = classified
                if conversation is not None and kind == "person":
                    continue
                references.append({"kind": kind, "url": url, "context": "conversation"})
    return dedupe_references(references, cap=12)


def _participant_name(participant: Any) -> str:
    member = _nested_mapping(participant, "participantType", "member")
    if member is None:
        return ""
    parts = (_rich_text(member.get("firstName")), _rich_text(member.get("lastName")))
    return " ".join(part for part in parts if part)


def format_message_elements(
    elements: Iterable[dict[str, Any]], *, timezone: tzinfo | None = None
) -> str:
    """Render provider messages chronologically, optionally in the legacy shape."""
    indexed = list(enumerate(elements))

    def order(entry: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, item = entry
        delivered_at = item.get("deliveredAt")
        return (
            delivered_at if isinstance(delivered_at, int) else 2**63 - 1,
            index,
        )

    blocks: list[str] = []
    rendered_date = None
    for _, item in sorted(indexed, key=order):
        body = next(
            (
                text
                for key in ("body", "renderContentFallbackText", "subject", "footer")
                if (text := _rich_text(item.get(key)))
            ),
            "",
        )
        if not body:
            continue
        sender = _participant_name(item.get("sender"))
        delivered_at = item.get("deliveredAt")
        when = None
        if timezone is not None and isinstance(delivered_at, int):
            try:
                when = datetime.fromtimestamp(delivered_at / 1000, tz=timezone)
            except (OSError, OverflowError, ValueError):
                pass
        if when is None or not sender:
            blocks.append(f"{sender}\n{body}" if sender else body)
            continue

        hour = when.hour % 12 or 12
        clock = f"{hour}:{when.minute:02d} {'AM' if when.hour < 12 else 'PM'}"
        lines: list[str] = []
        if when.date() != rendered_date:
            lines.append(f"{when:%b} {when.day}, {when.year}")
            rendered_date = when.date()
        lines.extend(
            (
                f"{sender} sent the following message at {clock}",
                f"View {sender}'s profile",
                f"{sender} {clock}",
                body,
            )
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
