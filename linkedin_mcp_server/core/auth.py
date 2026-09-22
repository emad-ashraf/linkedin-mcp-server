"""Authentication functions for LinkedIn."""

import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

from patchright.async_api import (
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)

from linkedin_mcp_server.debug_trace import record_page_trace
from linkedin_mcp_server.debug_utils import stabilize_navigation
from linkedin_mcp_server.privacy import redact_private_navigation_value

from .exceptions import AuthenticationError, NetworkError, RateLimitError
from .proxy_errors import (
    goto_reporting_proxy_errors,
    raise_if_proxy_error,
    redact_proxy_credentials,
)
from .rate_limit_markers import HTTP_TOO_MANY_REQUESTS, REDIRECT_LOOP_NAV_FAILURE

logger = logging.getLogger(__name__)

_AUTH_BLOCKER_URL_PATTERNS = (
    "/login",
    "/authwall",
    "/checkpoint",
    "/challenge",
    "/uas/login",
    "/uas/consumer-email-challenge",
)
_LOGIN_TITLE_PATTERNS = (
    "linkedin login",
    "sign in | linkedin",
)
# English only, and knowingly so: these are the words the account picker uses,
# and the words change with the interface language while nothing about the page
# announces which one is in play. The structural check below carries the
# locales this table does not, which is why it runs first.
_AUTH_BARRIER_TEXT_MARKERS = (
    ("welcome back", "sign in using another account"),
    ("welcome back", "join now"),
    ("choose an account", "sign in using another account"),
    ("continue as", "sign in using another account"),
)
_REMEMBER_ME_CONTAINER_SELECTOR = "#rememberme-div"
_REMEMBER_ME_BUTTON_SELECTOR = "#rememberme-div button"
_MANUAL_LOGIN_STATUS_INTERVAL_SECONDS = 30
_AUTH_COOKIE_URL = "https://www.linkedin.com/feed/"
# Seconds to wait before loading /feed/ a second time after one barrier
# sighting. Long enough for an interstitial to clear, short enough that a dead
# session is still reported inside one tool timeout.
_BARRIER_REPROBE_DELAY_RANGE = (5.0, 10.0)
# Auth routes that only a signed-out browser is sent to. ``/checkpoint`` and
# ``/challenge`` are not here: LinkedIn shows those to signed-in sessions too,
# and clears them on the next load.
_LOGIN_REDIRECT_PATHS = ("/login", "/uas/login", "/authwall")
AUTH_COOKIE_NAMES = ("li_at", "JSESSIONID")


async def is_logged_in(page: Page) -> bool:
    """Check if currently logged in to LinkedIn.

    Uses a three-tier strategy:
    1. Fail-fast on auth blocker URLs
    2. Check for navigation elements (primary)
    3. URL-based fallback for authenticated-only pages
    """
    try:
        current_url = page.url

        # Step 1: Fail-fast on auth blockers
        if _is_auth_blocker_url(current_url):
            return False

        # Step 2: Selector check (PRIMARY)
        old_selectors = '.global-nav__primary-link, [data-control-name="nav.settings"]'
        old_count = await page.locator(old_selectors).count()

        new_selectors = 'nav a[href*="/feed"], nav button:has-text("Home"), nav a[href*="/mynetwork"]'
        new_count = await page.locator(new_selectors).count()

        has_nav_elements = old_count > 0 or new_count > 0

        # Step 3: URL fallback
        authenticated_only_pages = [
            "/feed",
            "/mynetwork",
            "/messaging",
            "/notifications",
        ]
        is_authenticated_page = any(
            pattern in current_url for pattern in authenticated_only_pages
        )

        if not is_authenticated_page:
            return has_nav_elements

        if has_nav_elements:
            return True

        # Empty authenticated-only pages are a false positive during cookie
        # bridge recovery. Require some real page content before trusting URL.
        body_text = await page.evaluate("() => document.body?.innerText || ''")
        if not isinstance(body_text, str):
            return False

        return bool(body_text.strip())
    except PlaywrightTimeoutError:
        logger.warning(
            "Timeout checking login status on %s — treating as not logged in",
            redact_private_navigation_value(page.url),
        )
        return False
    except Exception as exc:
        logger.error(
            "Unexpected error checking login status: %s",
            redact_private_navigation_value(str(exc)),
        )
        raise


async def detect_auth_barrier(page: Page) -> str | None:
    """Detect LinkedIn auth/account-picker barriers on the current page."""
    return await _detect_auth_barrier(page, include_body_text=True)


async def _detect_auth_barrier(
    page: Page,
    *,
    include_body_text: bool,
) -> str | None:
    """Detect LinkedIn auth/account-picker barriers on the current page."""
    try:
        current_url = page.url
        if _is_auth_blocker_url(current_url):
            return f"auth blocker URL: {current_url}"

        try:
            title = (await page.title()).strip().lower()
        except Exception:
            title = ""
        if any(pattern in title for pattern in _LOGIN_TITLE_PATTERNS):
            return f"login title: {title}"

        # An id, so it says the same thing in every interface language, which
        # the picker's own words do not. The rest of the codebase already reads
        # this container as the picker; here it is the only signal that
        # survives a locale change, because the URL of an in-place picker is
        # the page that was asked for and its title is that page's title.
        #
        # Ahead of the quick check's exit, and not behind it, because the two
        # signals it does read are exactly the two this page defeats. The
        # quick check runs after every navigation, so a picker served in a
        # locale the table below does not cover reached every scraping tool
        # as page text. It costs one selector count, where the body read
        # below is what the quick check exists to skip.
        try:
            if await page.locator(_REMEMBER_ME_CONTAINER_SELECTOR).count() > 0:
                return f"account picker: {_REMEMBER_ME_CONTAINER_SELECTOR}"
        except Exception:
            logger.debug("Could not count remember-me containers", exc_info=True)

        if not include_body_text:
            return None

        try:
            body_text = await page.evaluate("() => document.body?.innerText || ''")
        except Exception:
            body_text = ""
        if not isinstance(body_text, str):
            body_text = ""

        normalized = re.sub(r"\s+", " ", body_text).strip().lower()
        for marker_group in _AUTH_BARRIER_TEXT_MARKERS:
            if all(marker in normalized for marker in marker_group):
                return f"auth barrier text: {' + '.join(marker_group)}"

        return None
    except PlaywrightTimeoutError:
        logger.warning(
            "Timeout checking auth barrier on %s — continuing without barrier detection",
            redact_private_navigation_value(page.url),
        )
        return None
    except Exception as exc:
        logger.error(
            "Unexpected error checking auth barrier: %s",
            redact_private_navigation_value(str(exc)),
        )
        return None


async def detect_auth_barrier_quick(page: Page) -> str | None:
    """Cheap auth-barrier check for normal navigations.

    Uses URL and title only, avoiding a full body-text fetch on healthy pages.
    """
    return await _detect_auth_barrier(page, include_body_text=False)


async def resolve_remember_me_prompt(page: Page, *, timeout: int | None = None) -> bool:
    """Click through LinkedIn's saved-account chooser when it appears.

    ``timeout`` bounds the whole attempt in milliseconds. ``None`` retains the
    normal per-operation limits.
    """
    deadline = None
    loop = None
    if timeout is not None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout / 1000

    def _operation_timeout(default: int) -> int | None:
        if deadline is None or loop is None:
            return default
        remaining = int((deadline - loop.time()) * 1000)
        if remaining <= 0:
            return None
        return min(default, remaining)

    try:
        logger.debug(
            "Checking remember-me prompt on %s",
            redact_private_navigation_value(page.url),
        )
        try:
            operation_timeout = _operation_timeout(3000)
            if operation_timeout is None:
                return False
            await page.wait_for_selector(
                _REMEMBER_ME_CONTAINER_SELECTOR, timeout=operation_timeout
            )
            logger.debug("Remember-me container appeared")
        except PlaywrightTimeoutError:
            logger.debug("Remember-me container did not appear in time")
            return False

        target = page.locator(_REMEMBER_ME_BUTTON_SELECTOR).first
        try:
            operation_timeout = _operation_timeout(3000)
            if operation_timeout is None:
                return False
            await target.wait_for(state="visible", timeout=operation_timeout)
            logger.debug("Remember-me button became visible")
        except PlaywrightTimeoutError:
            logger.debug(
                "Remember-me prompt container appeared without a visible login button"
            )
            return False

        logger.info("Clicking LinkedIn saved-account chooser to resume session")
        try:
            operation_timeout = _operation_timeout(3000)
            if operation_timeout is None:
                return False
            await target.scroll_into_view_if_needed(timeout=operation_timeout)
        except PlaywrightTimeoutError:
            logger.debug("Remember-me button did not scroll into view in time")

        try:
            operation_timeout = _operation_timeout(5000)
            if operation_timeout is None:
                return False
            await target.click(timeout=operation_timeout)
            logger.debug("Remember-me button click succeeded")
        except PlaywrightTimeoutError:
            logger.debug("Retrying remember-me prompt click with force=True")
            operation_timeout = _operation_timeout(5000)
            if operation_timeout is None:
                return False
            await target.click(timeout=operation_timeout, force=True)
            logger.debug("Remember-me button force-click succeeded")
        try:
            operation_timeout = _operation_timeout(10000)
            if operation_timeout is None:
                return False
            await page.wait_for_load_state(
                "domcontentloaded", timeout=operation_timeout
            )
        except PlaywrightTimeoutError:
            logger.debug("Remember-me prompt click did not finish loading in time")

        if deadline is None or loop is None:
            await asyncio.sleep(1)
        else:
            remaining = deadline - loop.time()
            if remaining > 0:
                await asyncio.sleep(min(1, remaining))
        return True
    except PlaywrightTimeoutError:
        logger.debug("Remember-me prompt was present but not clickable in time")
        return False
    except Exception:
        logger.debug("Failed to resolve remember-me prompt", exc_info=True)
        return False


def _is_auth_blocker_url(url: str) -> bool:
    """Return True only for real auth routes, not arbitrary slug substrings."""
    path = urlparse(url).path or "/"

    if path in _AUTH_BLOCKER_URL_PATTERNS:
        return True

    return any(
        path == f"{pattern}/" or path.startswith(f"{pattern}/")
        for pattern in _AUTH_BLOCKER_URL_PATTERNS
    )


async def _has_auth_cookie(page: Page) -> bool:
    """Return whether this context can send ``li_at`` to LinkedIn's feed."""
    return "li_at" in await auth_cookies(page)


async def auth_cookies(page: Page) -> dict[str, str]:
    """The auth cookies this context would send to /feed/, by name."""
    cookies = await page.context.cookies(_AUTH_COOKIE_URL)
    return {
        cookie["name"]: cookie["value"]
        for cookie in cookies
        if cookie.get("name") in AUTH_COOKIE_NAMES and cookie.get("value")
    }


async def barrier_confirmed(
    page: Page,
    barrier: str,
    *,
    detect: Callable[[Page], Awaitable[str | None]] | None = None,
) -> bool:
    """Load /feed/ once more before believing *barrier*.

    One sighting used to be the verdict, and every caller turns it into an
    ``AuthenticationError`` whose recovery rotates the session into
    ``invalid-state-*``. A ``/checkpoint`` interstitial that LinkedIn clears on
    the next load therefore cost the whole session: three rotations in 25
    minutes, measured under lease contention. Only a login redirect with no
    ``li_at`` in the context is believed at once; everything else has to be
    seen twice, which costs one extra page load in the rare case and none in
    the common one.

    Leaves the page on /feed/ either way. A caller that was reading another
    page has lost it and must navigate again rather than read on.

    *detect* is the check to repeat on the re-load, the quick one unless the
    caller sighted the barrier with another: a picker only the body text
    reveals would otherwise always read as cleared, and the caller would keep
    reporting a transient where a re-login is due.
    """
    path = urlparse(page.url).path
    if "li_at" not in await auth_cookies(page) and any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in _LOGIN_REDIRECT_PATHS
    ):
        return True
    logger.warning(
        "Auth barrier on %s (%s); re-probing /feed/ once before treating the "
        "session as expired",
        redact_private_navigation_value(page.url),
        redact_private_navigation_value(barrier),
    )
    await asyncio.sleep(random.uniform(*_BARRIER_REPROBE_DELAY_RANGE))
    load_error: str | None = None
    try:
        response = await goto_reporting_proxy_errors(
            page, _AUTH_COOKIE_URL, wait_until="domcontentloaded"
        )
        if response is not None and response.status == HTTP_TOO_MANY_REQUESTS:
            # Throttling on the second look is still throttling, and it is
            # read before the detector: whatever the page shows now is not
            # the session's doing.
            raise RateLimitError(
                "LinkedIn rate-limited /feed/ (HTTP 429) while re-checking an "
                "auth barrier. The saved LinkedIn session was not changed; "
                "wait before retrying."
            )
        await stabilize_navigation("feed re-probe", logger)
    except (NetworkError, RateLimitError):
        raise
    except Exception as exc:
        raise_if_proxy_error(exc)
        load_error = redact_proxy_credentials(f"{type(exc).__name__}: {exc}")
        if REDIRECT_LOOP_NAV_FAILURE in str(exc):
            # The loop can stop on an auth route, and the detector would
            # confirm it.
            raise NetworkError(
                f"/feed/ re-probe ended in a redirect loop ({load_error}), "
                f"which LinkedIn uses to throttle. The saved LinkedIn session "
                f"was not changed."
            ) from exc
        # A failed load still leaves a URL behind, and a barrier on it is
        # evidence; a load that fails without one is not.
    again = await (detect or detect_auth_barrier_quick)(page)
    await record_page_trace(
        page, "feed-reprobe", extra={"barrier": again, "error": load_error}
    )
    if again is not None:
        return True
    if load_error is not None:
        raise NetworkError(
            f"/feed/ re-probe did not finish loading and no auth barrier was "
            f"found ({load_error}). The saved LinkedIn session was not changed."
        )
    logger.info("Auth barrier cleared on the /feed/ re-probe; keeping the session")
    return False


async def wait_for_manual_login(page: Page, timeout: int = 300000) -> None:
    """Wait for user to manually complete login.

    Args:
        page: Patchright page object
        timeout: Timeout in milliseconds. ``0`` waits with no time limit.

    Raises:
        AuthenticationError: If the timeout elapses before login completes.
    """
    minutes = timeout / 60000
    if timeout:
        logger.info(
            "Please complete the login process manually in the browser. "
            "Waiting up to %.0f minutes...",
            minutes,
        )
    else:
        logger.info(
            "Please complete the login process manually in the browser. "
            "Waiting with no time limit (LOGIN_TIMEOUT=0)..."
        )

    def _timeout_error() -> AuthenticationError:
        return AuthenticationError(
            f"Manual login timeout: login was not completed within {minutes:.0f} "
            "minutes. Increase the limit with LOGIN_TIMEOUT (seconds, 0 = no "
            "limit) and run --login again."
        )

    loop = asyncio.get_running_loop()
    start_time = loop.time()
    deadline = start_time + timeout / 1000 if timeout else None
    next_status_time = start_time + _MANUAL_LOGIN_STATUS_INTERVAL_SECONDS

    def _check_wait_budget(*, log_status: bool = True) -> None:
        nonlocal next_status_time
        now = loop.time()
        if deadline is not None and now > deadline:
            raise _timeout_error()
        if log_status and now >= next_status_time:
            logger.info(
                "Still waiting for manual login. Complete sign-in in any "
                "LinkedIn tab in this browser."
            )
            next_status_time = now + _MANUAL_LOGIN_STATUS_INTERVAL_SECONDS

    def _remaining_wait_seconds() -> float | None:
        if deadline is None:
            return None
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise _timeout_error()
        return remaining

    while True:
        _check_wait_budget(log_status=False)
        # interactive_login rotates the old profile before opening this context,
        # so any li_at here was issued after the current login's challenges
        # cleared. The cookie belongs to the context rather than a tab and survives
        # the tab
        # that completed sign-in being closed.
        try:
            remaining = _remaining_wait_seconds()
            if remaining is None:
                has_auth_cookie = await _has_auth_cookie(page)
            else:
                has_auth_cookie = await asyncio.wait_for(
                    _has_auth_cookie(page), timeout=remaining
                )
        except TimeoutError:
            raise _timeout_error() from None
        except PlaywrightError as exc:
            raise AuthenticationError(
                "Manual login cancelled because the browser was closed."
            ) from exc

        if has_auth_cookie:
            _check_wait_budget(log_status=False)
            logger.info("Manual login completed successfully")
            return

        _check_wait_budget()

        if page.is_closed() and not page.context.pages:
            raise AuthenticationError(
                "Manual login cancelled because the browser was closed."
            )

        resolved_prompt = False
        if not page.is_closed():
            remaining = _remaining_wait_seconds()
            if remaining is None:
                resolved_prompt = await resolve_remember_me_prompt(page)
            else:
                resolved_prompt = await resolve_remember_me_prompt(
                    page, timeout=max(1, int(remaining * 1000))
                )

        if resolved_prompt:
            logger.info("Resolved saved-account chooser during manual login flow")
            continue

        remaining = _remaining_wait_seconds()
        await asyncio.sleep(1 if remaining is None else min(1, remaining))
