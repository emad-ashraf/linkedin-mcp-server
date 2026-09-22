"""Stable provider outcomes exposed at the MCP tool boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import mcp.types as mt
import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.tools import ToolResult

from linkedin_mcp_server.core.exceptions import (
    AuthenticationError,
    ProfileNotFoundError,
    RateLimitError,
    ScrapingError,
    SecurityChallengeError,
)
from linkedin_mcp_server.error_handler import raise_tool_error
from linkedin_mcp_server.outcomes import OUTCOME_KEY, OutcomeMiddleware


def _context() -> MiddlewareContext[mt.CallToolRequestParams]:
    return MiddlewareContext(
        message=mt.CallToolRequestParams(name="failing", arguments={}),
        source="client",
        type="request",
        method="tools/call",
        fastmcp_context=None,
    )


def _text(result: ToolResult) -> str:
    return "\n".join(
        block.text for block in result.content if isinstance(block, mt.TextContent)
    )


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (SecurityChallengeError("verify the account"), "challenge_required"),
        (AuthenticationError("the session ended"), "reauth_required"),
        (RateLimitError("slow down"), "rate_limited"),
        (ProfileNotFoundError("missing member"), "not_found"),
        (TimeoutError("bounded operation expired"), "timeout"),
        (ScrapingError("provider response failed"), "provider_error"),
    ],
)
async def test_each_provider_failure_has_exact_versioned_metadata(
    error: Exception, code: str
):
    middleware = OutcomeMiddleware()

    async def fail(context: MiddlewareContext[mt.CallToolRequestParams]):
        try:
            raise error
        except Exception as exc:
            raise_tool_error(exc, "failing")

    result = await middleware.on_call_tool(_context(), fail)

    assert result.is_error is True
    assert result.meta == {OUTCOME_KEY: {"v": 1, "code": code}}
    assert _text(result)


async def test_classification_walks_middleware_wrapper_causes():
    middleware = OutcomeMiddleware()

    async def wrapped(context: MiddlewareContext[mt.CallToolRequestParams]):
        try:
            raise ScrapingError("the provider failed")
        except ScrapingError as exc:
            raise ToolError("outer middleware wording") from exc

    result = await middleware.on_call_tool(_context(), wrapped)

    assert result.meta == {OUTCOME_KEY: {"v": 1, "code": "provider_error"}}
    assert _text(result) == "outer middleware wording"


async def test_private_auth_result_gets_public_reauth_outcome_without_losing_metadata():
    from linkedin_mcp_server.daemon_auth import MARKER_KEY, MARKER_VERSION

    middleware = OutcomeMiddleware()
    marker = {
        "v": MARKER_VERSION,
        "reason": "stale",
        "replayable": False,
        "browser_open": False,
        "generation": "generation-1",
    }

    async def auth_result(context: MiddlewareContext[mt.CallToolRequestParams]):
        return ToolResult(
            content=[mt.TextContent(type="text", text="sign in again")],
            meta={MARKER_KEY: marker, "kept": True},
            is_error=True,
        )

    result = await middleware.on_call_tool(_context(), auth_result)

    assert result.meta == {
        MARKER_KEY: marker,
        "kept": True,
        OUTCOME_KEY: {"v": 1, "code": "reauth_required"},
    }
    assert result.is_error is True
    assert _text(result) == "sign in again"


async def test_success_and_untyped_error_results_are_unchanged():
    middleware = OutcomeMiddleware()
    success = ToolResult(structured_content={"ok": True}, meta={"kept": 1})
    untyped_error = ToolResult(
        content=[mt.TextContent(type="text", text="ordinary error")],
        meta={"kept": 2},
        is_error=True,
    )

    async def return_success(context: MiddlewareContext[mt.CallToolRequestParams]):
        return success

    async def return_untyped_error(
        context: MiddlewareContext[mt.CallToolRequestParams],
    ):
        return untyped_error

    assert await middleware.on_call_tool(_context(), return_success) is success
    assert (
        await middleware.on_call_tool(_context(), return_untyped_error) is untyped_error
    )


async def test_unknown_exceptions_and_cancellation_remain_control_flow():
    middleware = OutcomeMiddleware()

    async def unknown(context: MiddlewareContext[mt.CallToolRequestParams]):
        raise ValueError("application bug")

    async def cancelled(context: MiddlewareContext[mt.CallToolRequestParams]):
        raise asyncio.CancelledError

    with pytest.raises(ValueError, match="application bug"):
        await middleware.on_call_tool(_context(), unknown)
    with pytest.raises(asyncio.CancelledError):
        await middleware.on_call_tool(_context(), cancelled)


async def test_fastmcp_owned_tool_deadline_is_a_timeout_outcome():
    server = FastMCP("deadline", mask_error_details=True)
    server.add_middleware(OutcomeMiddleware())

    @server.tool(timeout=0.01)
    async def too_slow() -> str:
        await asyncio.sleep(1)
        return "late"

    async with Client(server) as client:
        result = await client.call_tool("too_slow", {}, raise_on_error=False)

    assert result.is_error is True
    assert result.meta == {OUTCOME_KEY: {"v": 1, "code": "timeout"}}
    assert "timed out" in _text(result).lower()


WIRE_ERROR_CASES: tuple[tuple[Callable[[], Exception], str], ...] = (
    (
        lambda: SecurityChallengeError("account verification required"),
        "challenge_required",
    ),
    (lambda: AuthenticationError("sign in again"), "reauth_required"),
    (lambda: RateLimitError("wait before retrying"), "rate_limited"),
    (lambda: ProfileNotFoundError("the member is gone"), "not_found"),
    (lambda: TimeoutError("the provider deadline expired"), "timeout"),
    (lambda: ScrapingError("the provider response failed"), "provider_error"),
)


def _raise(error_factory: Callable[[], Exception]) -> str:
    raise error_factory()


@pytest.mark.parametrize(("error_factory", "code"), WIRE_ERROR_CASES)
async def test_direct_client_receives_every_typed_error_and_readable_fallback(
    error_factory: Callable[[], Exception], code: str
):
    server = FastMCP("direct", mask_error_details=True)
    server.add_middleware(OutcomeMiddleware())

    @server.tool
    async def failing() -> str:
        return _raise(error_factory)

    async with Client(server) as client:
        result = await client.call_tool("failing", {}, raise_on_error=False)

    assert result.is_error is True
    assert result.meta == {OUTCOME_KEY: {"v": 1, "code": code}}
    assert _text(result) not in ("", "Error calling tool 'failing'")
