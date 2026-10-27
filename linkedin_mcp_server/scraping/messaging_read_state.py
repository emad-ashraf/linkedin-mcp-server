"""Page-scoped routing for passive messaging read-state protection."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse
from weakref import WeakKeyDictionary, WeakSet

from linkedin_mcp_server.core.exceptions import LinkedInScraperException

_READ_STATE_WRITE_ROUTES = (
    "**/voyager/api/voyagerMessagingDashMessengerConversations*",
    "**/voyager/api/voyagerMessagingDashMessagingBadge*",
)
_READ_STATE_PATHS = (
    "/voyager/api/voyagerMessagingDashMessengerConversations",
    "/voyager/api/voyagerMessagingDashMessagingBadge",
)


class _PageGuard:
    def __init__(self) -> None:
        self.depth = 0
        self.read_mode = True
        self.registered: set[str] = set()
        self.listening = False
        self.registration: asyncio.Task[None] | None = None
        self.registration_failed = False
        self.protected: WeakSet[Any] = WeakSet()
        # Store stable callback identities; readers are recreated for each tool.
        self.request_listener = self.observe
        self.route_handler = self.handle

    def observe(self, request: Any) -> None:
        if (
            self.read_mode
            and request.method.upper() == "POST"
            and urlparse(request.url).path.startswith(_READ_STATE_PATHS)
        ):
            self.protected.add(request)

    async def handle(self, route: Any) -> None:
        request = route.request
        # A request notification can precede its asynchronous route callback.
        # Remember the notification's read scope even if that scope has exited.
        if request.method.upper() == "POST" and (
            self.read_mode or request in self.protected
        ):
            await route.abort()
        else:
            await route.fallback()

    async def install(self, page: Any) -> None:
        if self.registration_failed:
            raise LinkedInScraperException("Read-state protection registration failed.")
        if self.registration is None:
            self.registration = asyncio.create_task(self._install(page))
            self.registration.add_done_callback(self._registration_finished)
        # A cancelled caller must not leave a half-registered route that a
        # later caller registers again. The registration itself makes no reads.
        await asyncio.shield(self.registration)

    def _registration_finished(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() or task.exception() is not None:
            self.registration_failed = True
            # A failed task's traceback retains the page through this weak-map value.
            self.registration = None

    async def _install(self, page: Any) -> None:
        if not self.listening:
            page.on("request", self.request_listener)
            self.listening = True
        for pattern in _READ_STATE_WRITE_ROUTES:
            if pattern not in self.registered:
                await page.route(pattern, self.route_handler)
                self.registered.add(pattern)


# Values do not retain the page; the page owns its callbacks until it closes.
_GUARDS: WeakKeyDictionary[Any, _PageGuard] = WeakKeyDictionary()


@asynccontextmanager
async def without_read_state_writes(page: Any) -> AsyncIterator[None]:
    """Activate read protection without reconfiguring interception at exit."""
    state = _GUARDS.get(page)
    if state is None:
        state = _PageGuard()
        _GUARDS[page] = state
    state.depth += 1
    state.read_mode = True
    try:
        await state.install(page)
        yield
    finally:
        state.depth -= 1
        # Keep protection latched for late request events. Only a verified
        # compose operation can release it; no interception update is needed.


def allow_explicit_message_write(page: Any) -> None:
    """Release the read latch only when the existing writer has verified its target."""
    state = _GUARDS.get(page)
    if state is None:
        return
    if state.depth:
        raise LinkedInScraperException(
            "Cannot compose while a passive messaging read is active."
        )
    state.read_mode = False
