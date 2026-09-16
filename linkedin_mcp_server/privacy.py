"""Small privacy guards for values that may enter diagnostics."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import re


_PRIVATE_LINKEDIN_URL_IN_TEXT = re.compile(
    r"https://(?:[A-Za-z0-9-]+\.)*linkedin\.com/(?:messaging|in)(?:/[^\s\"'<>]*)?"
)
_REDACTED_LINKEDIN_URL = "https://www.linkedin.com/<private-route>"


def is_linkedin_messaging_url(value: Any) -> bool:
    """Return whether a value is an absolute LinkedIn messaging URL."""
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    host = (parsed.hostname or "").casefold()
    return (host == "linkedin.com" or host.endswith(".linkedin.com")) and (
        parsed.path == "/messaging" or parsed.path.startswith("/messaging/")
    )


def is_private_linkedin_navigation_url(value: Any) -> bool:
    """Return whether a URL can identify a message thread or person."""
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    host = (parsed.hostname or "").casefold()
    if host != "linkedin.com" and not host.endswith(".linkedin.com"):
        return False
    return is_linkedin_messaging_url(value) or parsed.path.startswith("/in/")


def redact_private_navigation_value(value: Any) -> Any:
    """Remove messaging route IDs and queries from nested diagnostic values."""
    if isinstance(value, str):
        if is_private_linkedin_navigation_url(value):
            return _REDACTED_LINKEDIN_URL
        return _PRIVATE_LINKEDIN_URL_IN_TEXT.sub(_REDACTED_LINKEDIN_URL, value)
    if isinstance(value, dict):
        return {
            key: redact_private_navigation_value(child) for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_private_navigation_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(redact_private_navigation_value(child) for child in value)
    return value
