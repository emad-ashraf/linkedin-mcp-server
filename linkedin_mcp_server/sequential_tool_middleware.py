"""Middleware that serializes MCP tool execution across server processes."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

import mcp.types as mt

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult

from linkedin_mcp_server.config import get_config
from linkedin_mcp_server.config.loaders import EnvironmentKeys
from linkedin_mcp_server.exceptions import AccountCooldownError, BrowserBusyError
from linkedin_mcp_server.pacing import (
    JobStore,
    account_cooldown_disabled,
    cooldown_resume_at,
    hourly_cap_resume_at,
    load_account_budget,
    request_arrived_at,
    tool_call_gap,
)
from linkedin_mcp_server.profile_lease import get_profile_lease

logger = logging.getLogger(__name__)


class SequentialToolExecutionMiddleware(Middleware):
    """Ensure only one tool call at a time drives the shared LinkedIn browser.

    Two layers, because one is not enough:

    * an ``asyncio.Lock`` serializes calls inside this process, where several MCP
      sessions can share one server;
    * the profile lease serializes calls across processes, where each MCP client
      instance spawns its own server against the same Chromium profile. A
      process on the holder's version waits for the handoff; one on another
      version is refused at once, with the holder named.

    Without the second layer two processes open that profile simultaneously and
    the last one to close silently overwrites the other's cookies.

    Serial is not the same as paced, though, and this is also where the gap
    between consecutive calls lives. The account budget is not spent here: it
    is charged per page load at the navigation, since one call can be
    fourteen of them (``pacing.charge_navigation``).
    """

    # Tools that answer from local disk and never reach LinkedIn. Pacing them
    # would charge the account budget for activity that never happened and make
    # a status poll wait out a gap meant for page loads. Named rather than
    # derived: there is no flag on a tool saying whether it navigates, and a
    # wrong guess here is silent -- an unlisted local tool merely pays a gap it
    # did not owe, while a listed navigating one would escape pacing entirely.
    _LOCAL_ONLY_TOOLS = frozenset(
        {
            "get_enrichment_status",
            "get_company_cache",
            "query_company_cache",
            # Writes the queue to disk; the visits happen in run_enrichment_bunch.
            "start_enrichment_job",
            # Closes the browser; the only page it touches is the one going away.
            "close_session",
        }
    )

    # The bulk tools. They are not refused by the hourly cap here: each
    # plans its bunch against the hour's headroom and answers with a status
    # dict and a wait, which is the better answer than an error for a call
    # that only needed to plan zero. They still pay the gap and the cooldown.
    _BULK_TOOLS = frozenset(
        {
            "run_enrichment_bunch",
            "enrich_companies",
            "enrich_company_deep",
        }
    )

    # Tools that leave a trace on another member's side -- an invitation, a
    # message -- and so get the longer gap band. Named like the sets above:
    # the ``destructiveHint`` annotation would say the same, but reading it
    # needs the FastMCP context, which a direct call does not carry. The
    # enrichment tools are not here; they write to local disk only.
    _WRITE_TOOLS = frozenset({"connect_with_person", "send_message"})

    # How often the wait for the gap re-reports progress. A single notification
    # at the start left a client silent for the rest of a gap that can now
    # run to a minute, which is the silence the old five-second gap avoided.
    _PROGRESS_EVERY_SECONDS = 10.0

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # Never charged from here -- the cooldown and the hourly count are
        # only checked before a call is let through -- though a first read
        # does materialise the budget file, as every reader of it does.
        self._store = JobStore()
        # Monotonic instant the next call may start at. 0 lets the first call
        # of the process run immediately -- the gap is between calls, and there
        # is nothing yet to be spaced from.
        self._next_call_at = 0.0

    async def _report_progress(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        *,
        message: str,
    ) -> None:
        fastmcp_context = context.fastmcp_context
        if fastmcp_context is None or fastmcp_context.request_context is None:
            return

        await fastmcp_context.report_progress(
            progress=0,
            total=100,
            message=message,
        )

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        tool_name = context.message.name
        wait_started = time.perf_counter()
        # Recorded before the lock wait: the tool's own timeout starts only
        # once the call is let through, so this is the one clock that sees
        # the queue.
        arrival = request_arrived_at.set(time.monotonic())
        try:
            # Before the lock, so a paused account answers at once instead of
            # queueing behind a call that is itself about to be refused.
            if tool_name not in self._LOCAL_ONLY_TOOLS:
                self._refuse_while_the_account_is_paused(tool_name)
            logger.debug("Waiting for scraper lock for tool '%s'", tool_name)
            await self._report_progress(
                context,
                message="Queued waiting for scraper lock",
            )

            async with self._lock:
                wait_seconds = time.perf_counter() - wait_started
                logger.debug(
                    "Acquired scraper lock for tool '%s' after %.3fs",
                    tool_name,
                    wait_seconds,
                )
                await self._report_progress(
                    context,
                    message="Scraper lock acquired, starting tool",
                )
                # Slept holding the lock on purpose: the gap is between calls
                # to LinkedIn, so letting a queued call run through it would
                # defeat it. The cross-process lease is not held here -- that
                # one is taken inside, after the wait, so no other process is
                # blocked by ours.
                if tool_name not in self._LOCAL_ONLY_TOOLS:
                    await self._space_out_the_call(context, tool_name)
                return await self._run_owning_the_profile(context, call_next, tool_name)
        finally:
            request_arrived_at.reset(arrival)

    async def _space_out_the_call(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        tool_name: str,
    ) -> None:
        """Wait out whatever is left of the gap since the previous call."""
        wait_seconds = self._next_call_at - time.monotonic()
        if wait_seconds <= 0:
            return

        logger.debug(
            "Pacing tool '%s': %.2fs left of the gap since the previous call",
            tool_name,
            wait_seconds,
        )
        while wait_seconds > 0:
            await self._report_progress(
                context,
                message=f"Pacing LinkedIn activity, starting in {wait_seconds:.1f}s",
            )
            await asyncio.sleep(min(wait_seconds, self._PROGRESS_EVERY_SECONDS))
            wait_seconds = self._next_call_at - time.monotonic()

    def _refuse_while_the_account_is_paused(self, tool_name: str) -> None:
        """Refuse the call outright while the account is cooling down.

        Read fresh from disk on every call: the pause is written by whichever
        process saw LinkedIn push back, and the hour's count by every
        navigation, so this process's memory of either is always stale.

        Best-effort in the same direction as recording: a ledger that cannot
        be read is no reason to refuse a call, so an unreadable file lets the
        call through.
        """
        try:
            now = datetime.now(timezone.utc)
            resume_at = None
            reason = ""
            if not account_cooldown_disabled():
                resume_at = cooldown_resume_at(self._store, now)
                reason = "LinkedIn pushed back and the account is cooling down"
            # The bulk tools plan their bunch against the hourly headroom and
            # answer with a status dict and a wait; refusing them here would
            # turn that into an error for a call that only needed to plan zero.
            if resume_at is None and tool_name not in self._BULK_TOOLS:
                # Local and aware, as the navigator hands it: the budget's
                # schedule and start date are read in the operator's time.
                budget = load_account_budget(self._store, datetime.now().astimezone())
                resume_at = hourly_cap_resume_at(budget, now)
                reason = "the hourly action cap is reached"
        except Exception:
            logger.debug("Could not read the account cooldown", exc_info=True)
            return
        if resume_at is None:
            return

        error = AccountCooldownError(resume_at, reason=reason)
        logger.info(
            "Refused tool '%s' until %s: %s", tool_name, error.resume_at, reason
        )
        # A ToolError for the same reason BrowserBusyError is one below.
        raise ToolError(str(error)) from error

    async def _run_owning_the_profile(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
        tool_name: str,
    ) -> ToolResult:
        """Run the tool while this process owns the browser profile."""
        # Imported here so the module stays importable without the driver.
        from linkedin_mcp_server.drivers.browser import (
            note_activity,
            note_call_started,
            release_profile_if_idle_or_requested,
        )

        lease = get_profile_lease()
        acquired = lease.try_acquire()
        if not acquired:
            await self._report_progress(
                context,
                message="Another LinkedIn MCP client is using the browser",
            )
            budget = get_config().browser.browser_wait_seconds
            # Raised as a ToolError here, not via error_handler: an exception
            # thrown in middleware does not pass through raise_tool_error, and
            # mask_error_details would otherwise hide the explanation.
            try:
                acquired = await lease.acquire_or_refuse(timeout=budget)
            except BrowserBusyError as refusal:
                raise ToolError(str(refusal)) from refusal

        if not acquired:
            holder = lease.holder()
            described = holder.describe() if holder else "an unidentified process"
            logger.warning(
                "Tool '%s' gave up waiting for the shared browser, held by %s",
                tool_name,
                described,
            )
            raise ToolError(str(BrowserBusyError(holder=described)))

        hold_started = time.perf_counter()
        try:
            # Marks the browser as in use so the background handoff poll cannot
            # close it out from under this call. Inside the try so the finally
            # always balances it, including if the call is cancelled.
            note_call_started()
            return await call_next(context)
        finally:
            hold_seconds = time.perf_counter() - hold_started
            logger.debug(
                "Released scraper lock for tool '%s' after %.3fs",
                tool_name,
                hold_seconds,
            )
            note_activity()
            # Inside this finally rather than around the whole call: a call
            # that gave up waiting for the lease never reached this point and
            # never touched LinkedIn, so it owes no gap. Still inside the
            # asyncio lock, so the next call in this process sees the deadline
            # before it tests it.
            if tool_name not in self._LOCAL_ONLY_TOOLS:
                self._next_call_at = time.monotonic() + tool_call_gap(
                    os.environ.get(EnvironmentKeys.TOOL_CALL_GAP_SECONDS),
                    write=tool_name in self._WRITE_TOOLS,
                )
            lease.release()
            # Hand the browser over now if someone is waiting, rather than
            # holding it for the rest of this process's lifetime.
            try:
                await release_profile_if_idle_or_requested()
            except Exception:
                logger.debug("Profile handoff check failed", exc_info=True)
