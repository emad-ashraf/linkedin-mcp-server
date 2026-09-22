"""Core browser management, authentication, and scraping utilities."""

from typing import TYPE_CHECKING

from .exceptions import (
    AuthenticationError,
    ElementNotFoundError,
    LinkedInScraperException,
    NetworkError,
    ProfileNotFoundError,
    ProxyConnectionError,
    RateLimitError,
    ScrapingError,
    TransientBarrierError,
)

if TYPE_CHECKING:
    from .auth import (
        AUTH_COOKIE_NAMES,
        auth_cookies,
        barrier_confirmed,
        detect_auth_barrier,
        detect_auth_barrier_quick,
        is_logged_in,
        resolve_remember_me_prompt,
        wait_for_manual_login,
    )
    from .browser import BrowserManager, await_deferring_cancels
    from .proxy_errors import (
        as_proxy_error,
        goto_reporting_proxy_errors,
        is_proxy_error,
        proxy_hint,
        raise_if_proxy_configured,
        raise_if_proxy_error,
        redact_proxy_credentials,
        redacted_copy,
    )
    from .utils import detect_rate_limit, handle_modal_close, scroll_to_bottom

# ``linkedin_mcp_server.exceptions`` imports ``core.exceptions``, which runs
# this module, so anything eager here is reached before the greenlet guard in
# the package ``__init__`` has spoken and inside the installer supervisor,
# which runs under ``-S -I`` with no third-party package on the path. Every
# module below imports patchright at the top, or ``config`` and with it
# ``dotenv``, so only ``.exceptions`` may be eager.
_LAZY_EXPORTS = {
    "AUTH_COOKIE_NAMES": ".auth",
    "auth_cookies": ".auth",
    "barrier_confirmed": ".auth",
    "detect_auth_barrier": ".auth",
    "detect_auth_barrier_quick": ".auth",
    "is_logged_in": ".auth",
    "resolve_remember_me_prompt": ".auth",
    "wait_for_manual_login": ".auth",
    "BrowserManager": ".browser",
    "await_deferring_cancels": ".browser",
    "as_proxy_error": ".proxy_errors",
    "goto_reporting_proxy_errors": ".proxy_errors",
    "is_proxy_error": ".proxy_errors",
    "proxy_hint": ".proxy_errors",
    "raise_if_proxy_configured": ".proxy_errors",
    "raise_if_proxy_error": ".proxy_errors",
    "redact_proxy_credentials": ".proxy_errors",
    "redacted_copy": ".proxy_errors",
    "detect_rate_limit": ".utils",
    "handle_modal_close": ".utils",
    "scroll_to_bottom": ".utils",
}


def __getattr__(name: str) -> object:
    """Resolve patchright-backed exports without loading them for leaf imports."""
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "AUTH_COOKIE_NAMES",
    "AuthenticationError",
    "BrowserManager",
    "auth_cookies",
    "await_deferring_cancels",
    "barrier_confirmed",
    "detect_auth_barrier",
    "detect_auth_barrier_quick",
    "ElementNotFoundError",
    "LinkedInScraperException",
    "NetworkError",
    "ProfileNotFoundError",
    "ProxyConnectionError",
    "RateLimitError",
    "ScrapingError",
    "TransientBarrierError",
    "as_proxy_error",
    "goto_reporting_proxy_errors",
    "is_proxy_error",
    "proxy_hint",
    "raise_if_proxy_configured",
    "raise_if_proxy_error",
    "redact_proxy_credentials",
    "redacted_copy",
    "detect_rate_limit",
    "handle_modal_close",
    "is_logged_in",
    "resolve_remember_me_prompt",
    "scroll_to_bottom",
    "wait_for_manual_login",
]
