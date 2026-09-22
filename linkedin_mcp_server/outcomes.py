"""Stable structured outcomes for provider failures."""

from __future__ import annotations

import sys
from typing import Literal

import mcp.types as mt
from mcp.shared.exceptions import McpError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.core.exceptions import (
    AuthenticationError,
    LinkedInScraperException,
    ProfileNotFoundError,
    RateLimitError,
    SecurityChallengeError,
)
from linkedin_mcp_server.exceptions import (
    AccountCooldownError,
    ActionLimitError,
    CredentialsNotFoundError,
    LinkedInMCPError,
    SessionExpiredError,
)
from linkedin_mcp_server.core.proxy_errors import redact_proxy_credentials

OUTCOME_KEY = "linkedin.dev/outcome"
OUTCOME_VERSION = 1
OutcomeCode = Literal[
    "challenge_required",
    "reauth_required",
    "rate_limited",
    "not_found",
    "timeout",
    "provider_error",
]


def _exception_chain(exception: BaseException):
    current: BaseException | None = exception
    while current is not None:
        yield current
        current = current.__cause__


def _is_fastmcp_tool_timeout(exception: BaseException) -> bool:
    """Recognize the deadline FastMCP applies from ``@tool(timeout=...)``.

    FastMCP 3.4 exposes this as the generic MCP code ``-32000`` rather than a
    timeout subclass or dedicated code. Require both its code and complete
    framework-owned message shape so another ``-32000`` provider error is not
    mislabeled. This adapter is private; clients only consume our stable code.
    """
    if not isinstance(exception, McpError) or exception.error.code != -32000:
        return False
    message = exception.error.message
    return (
        isinstance(message, str)
        and message.startswith("Tool '")
        and "' execution timed out after " in message
        and message.endswith("s")
    )


def _is_owner_unreachable(exception: BaseException) -> bool:
    """Recognize the proxy's transport boundary without importing it at startup."""
    # ``daemon_proxy`` deliberately stays out of direct-server startup. By the
    # time this exception can exist that module is already loaded, so consulting
    # the module registry preserves the role boundary without classifying prose.
    module = sys.modules.get("linkedin_mcp_server.daemon_proxy")
    owner_error = getattr(module, "OwnerUnreachableError", None)
    return isinstance(owner_error, type) and isinstance(exception, owner_error)


def outcome_code(exception: BaseException) -> OutcomeCode | None:
    """Classify a recognized provider failure from its preserved cause chain."""
    chain = tuple(_exception_chain(exception))
    if any(isinstance(exc, SecurityChallengeError) for exc in chain):
        return "challenge_required"
    if any(
        isinstance(
            exc,
            (AuthenticationError, CredentialsNotFoundError, SessionExpiredError),
        )
        for exc in chain
    ):
        return "reauth_required"
    if any(
        isinstance(exc, (RateLimitError, ActionLimitError, AccountCooldownError))
        for exc in chain
    ):
        return "rate_limited"
    if any(isinstance(exc, ProfileNotFoundError) for exc in chain):
        return "not_found"
    if any(
        isinstance(exc, (TimeoutError, PlaywrightTimeoutError))
        or _is_fastmcp_tool_timeout(exc)
        for exc in chain
    ):
        return "timeout"
    if any(
        isinstance(exc, (LinkedInScraperException, LinkedInMCPError)) for exc in chain
    ):
        return "provider_error"
    if any(_is_owner_unreachable(exc) for exc in chain):
        return "provider_error"
    return None


def _outcome(code: OutcomeCode) -> dict[str, object]:
    return {"v": OUTCOME_VERSION, "code": code}


def _readable_message(exception: BaseException, code: OutcomeCode) -> str:
    """Keep shaped tool prose, bypassing only FastMCP's generic mask."""
    for current in _exception_chain(exception):
        message = str(current).strip()
        if not message:
            continue
        if isinstance(current, ToolError) and message.startswith(
            "Error calling tool '"
        ):
            continue
        if isinstance(current, ToolError):
            return redact_proxy_credentials(message)
        if isinstance(
            current,
            (
                LinkedInScraperException,
                LinkedInMCPError,
                TimeoutError,
                PlaywrightTimeoutError,
                McpError,
            ),
        ) or _is_owner_unreachable(current):
            return redact_proxy_credentials(message)
    return {
        "challenge_required": "LinkedIn requires interactive security verification.",
        "reauth_required": "LinkedIn authentication is required.",
        "rate_limited": "LinkedIn temporarily limited this operation.",
        "not_found": "The requested LinkedIn target was not found.",
        "timeout": "The LinkedIn provider operation timed out.",
        "provider_error": "The LinkedIn provider operation failed.",
    }[code]


def _has_owner_auth_marker(result: ToolResult) -> bool:
    if not result.is_error or not result.meta:
        return False
    # Lazy to keep the generic outcome boundary from importing daemon/auth
    # machinery when a direct server starts.
    from linkedin_mcp_server.daemon_auth import MARKER_KEY

    return MARKER_KEY in result.meta


def _has_unknown_owner_outcome(result: ToolResult) -> bool:
    """Recognize the recovery middleware's result-form transport failure."""
    module = sys.modules.get("linkedin_mcp_server.daemon_proxy")
    status = getattr(module, "UNKNOWN_OUTCOME_STATUS", None)
    structured = result.structured_content
    return (
        result.is_error
        and isinstance(status, str)
        and isinstance(structured, dict)
        and structured.get("status") == status
        and structured.get("retry_safe") is False
    )


class OutcomeMiddleware(Middleware):
    """Expose provider failure categories without requiring prose parsing."""

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        try:
            result = await call_next(context)
        except Exception as exc:
            code = outcome_code(exc)
            if code is None:
                raise
            return ToolResult(
                content=[
                    mt.TextContent(type="text", text=_readable_message(exc, code))
                ],
                meta={OUTCOME_KEY: _outcome(code)},
                is_error=True,
            )

        # The detached owner turns an auth exception into a private repair
        # marker before this outer middleware sees it. Add the public contract
        # without removing that marker: the proxy still needs it to repair and
        # replay, while a final unrepaired result needs a policy code.
        result_code: OutcomeCode | None = None
        if _has_owner_auth_marker(result):
            result_code = "reauth_required"
        elif _has_unknown_owner_outcome(result):
            result_code = "provider_error"
        if result_code is not None and OUTCOME_KEY not in (result.meta or {}):
            meta = dict(result.meta or {})
            meta[OUTCOME_KEY] = _outcome(result_code)
            return result.model_copy(update={"meta": meta})
        return result
