"""Identity a loaded person page states about itself."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Protocol

from linkedin_mcp_server.scraping.session import ScrapingSession


class MessageTarget(Protocol):
    """The recipient identity a top-card compose action carries."""

    @property
    def profile_urn(self) -> str: ...


class MessageTargetResolution(Protocol):
    """One attempt at reading that identity off the current page."""

    @property
    def target(self) -> MessageTarget | None: ...


# The Message action can identify a profile when it is available. The profile
# top-card component provides the read-only fallback for pages without that
# action. Keep the Message action read borrowed from the sender so this module
# does not need to know how composing or sending works.
ReadMessageTarget = Callable[[], Awaitable[MessageTargetResolution]]

_TOPCARD_ID_RE = re.compile(
    r"com\.linkedin\.sdui\.profile\.card\.ref(ACo[A-Za-z0-9_-]{20,})Topcard"
)
_TOPCARD_SNAPSHOT_JS = """() => ({
    pageUrl: window.location.href,
    topcardIds: [...document.querySelectorAll(
        'main [id^="com.linkedin.sdui.profile.card.ref"][id$="Topcard"]'
    )].map(element => element.id),
})"""


class ProfilePageReader:
    """Read the profile URN and display name off the bound person page."""

    def __init__(
        self,
        session: ScrapingSession,
        read_message_target: ReadMessageTarget,
    ):
        self._session = session
        self._read_message_target = read_message_target

    async def _extract_profile_urn(
        self, *, expected_loaded_url: str | None = None
    ) -> str | None:
        """Read one unambiguous identity from the loaded profile's top card."""
        if expected_loaded_url is None:
            resolution = await self._read_message_target()
            if resolution.target:
                return resolution.target.profile_urn
        try:
            snapshot = await self._session.page.evaluate(_TOPCARD_SNAPSHOT_JS)
        except Exception:
            return None
        if not isinstance(snapshot, dict):
            return None
        if (
            expected_loaded_url is not None
            and snapshot.get("pageUrl") != expected_loaded_url
        ):
            return None
        topcard_ids = snapshot.get("topcardIds")
        if not isinstance(topcard_ids, list) or len(topcard_ids) != 1:
            return None
        topcard_id = topcard_ids[0]
        match = (
            _TOPCARD_ID_RE.fullmatch(topcard_id)
            if isinstance(topcard_id, str)
            else None
        )
        return match.group(1) if match else None

    async def _read_profile_display_name(self) -> str | None:
        """Read the visible profile name from the current person page."""
        display_name = await self._session.page.evaluate(
            """() => {
                const heading = document.querySelector('main h1');
                const normalize = value => (value || '').replace(/\\s+/g, ' ').trim();
                if (heading) {
                    const headingText = normalize(
                        heading.innerText || heading.textContent || ''
                    );
                    if (headingText) return headingText;
                }

                const main = document.querySelector('main');
                if (!main) return '';
                const lines = (main.innerText || '')
                    .split('\\n')
                    .map(normalize)
                    .filter(Boolean);
                return lines[0] || '';
            }"""
        )
        if not isinstance(display_name, str):
            return None
        display_name = display_name.strip()
        return display_name or None
