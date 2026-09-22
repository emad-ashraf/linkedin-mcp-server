"""The minimum gap the middleware leaves between two MCP tool calls.

Nothing here mocks the sleep away: the point of the change is elapsed time on
the wire, and a test that only watched a code path would still pass with the
spacing removed. The configured gaps are therefore small, and the assertions
are on measured seconds.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp.exceptions import ToolError

from linkedin_mcp_server.config.loaders import EnvironmentKeys
from linkedin_mcp_server.exceptions import AccountCooldownError
from linkedin_mcp_server.pacing import (
    ACCOUNT_COOLDOWN_FILE,
    JobStore,
    load_account_budget,
    read_account_cooldown,
    record_throttle_signal,
    request_arrived_at,
)
from linkedin_mcp_server.sequential_tool_middleware import (
    SequentialToolExecutionMiddleware,
)

# Small enough that the suite does not crawl, large enough to sit well clear of
# scheduling noise. The configured value is the read minimum, so the floor for
# a call is exactly this; the ceiling is 2.5x it.
GAP = 0.5


def _call_context(tool_name: str = "get_inbox") -> MagicMock:
    context = MagicMock()
    context.message.name = tool_name
    context.fastmcp_context = None
    return context


async def _timed_call(
    middleware: SequentialToolExecutionMiddleware, tool_name: str = "get_inbox"
) -> float:
    started = time.monotonic()
    await middleware.on_call_tool(_call_context(tool_name), AsyncMock())
    return time.monotonic() - started


@pytest.fixture
def paced(monkeypatch):
    monkeypatch.setenv(EnvironmentKeys.TOOL_CALL_GAP_SECONDS, str(GAP))
    return SequentialToolExecutionMiddleware()


class TestCallsAreSpacedApart:
    """`get_inbox` then `get_conversation` back to back is what drew a 429."""

    async def test_the_second_call_waits_out_the_gap(self, paced):
        first = await _timed_call(paced, "get_inbox")
        second = await _timed_call(paced, "get_conversation")

        assert second >= GAP * 0.8, (
            f"the second call started {second:.3f}s in, so nothing spaced it "
            "from the first"
        )
        assert first < GAP * 0.8, (
            "the first call of the process was delayed; the gap belongs "
            "between calls, not in front of each one"
        )

    async def test_the_gap_is_the_configured_one(self, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.TOOL_CALL_GAP_SECONDS, "0.1")
        middleware = SequentialToolExecutionMiddleware()

        await _timed_call(middleware)
        second = await _timed_call(middleware)

        # Would be ~4s on the 5s default, so this fails if the environment is
        # ignored, and ~0s if the spacing is gone.
        assert 0.08 <= second < GAP * 0.8

    async def test_zero_turns_the_spacing_off(self, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.TOOL_CALL_GAP_SECONDS, "0")
        middleware = SequentialToolExecutionMiddleware()

        await _timed_call(middleware)
        second = await _timed_call(middleware)

        assert second < 0.1


class TestTheMiddlewareChargesNoBudgetItself:
    """The account budget is spent per page load, at the navigation (#58).

    One unit per call was the old rule, and it undercounted by the number of
    pages a call loaded: a full-sections profile read was fourteen loads for
    one unit. A call that loads no page therefore owes nothing here, and the
    navigator is where the charge now lives (`tests/scraping/test_navigation`).
    """

    async def test_a_call_that_loads_no_page_spends_nothing(self, paced, tmp_path):
        # The same jobs root the autouse ledger-isolation fixture hands out.
        store = JobStore(tmp_path / "jobs")
        now = datetime.now()

        await _timed_call(paced)
        await _timed_call(paced, "run_enrichment_bunch")
        assert load_account_budget(store, now).ledger.spent(now) == 0


class TestLocalOnlyToolsAreNotPaced:
    """A tool answered from disk never reached LinkedIn, so it owes nothing.

    Both halves matter and both are silent when wrong: a status poll made to
    wait out a page-load gap merely feels broken, while one charged to the
    account budget makes the ledger overstate real activity and stop the next
    bulk job early.
    """

    async def test_a_local_tool_does_not_wait(self, paced):
        await _timed_call(paced)
        second = await _timed_call(paced, "get_enrichment_status")

        assert second < 0.1

    async def test_a_local_tool_does_not_arm_the_gap_for_the_next_call(self, paced):
        await _timed_call(paced, "get_enrichment_status")
        second = await _timed_call(paced)

        assert second < 0.1

    @pytest.mark.parametrize("tool", ["start_enrichment_job", "close_session"])
    async def test_queue_and_session_tools_do_not_wait(self, paced, tool):
        """``start_enrichment_job`` only writes the queue; ``close_session``
        only closes the browser. Neither loads a page, so neither waits."""
        await _timed_call(paced)
        second = await _timed_call(paced, tool)

        assert second < 0.1


class TestBulkToolsAreStillPaced:
    """An enrichment tool records its own units, one per page load, and the
    navigator adds each load's kind to the copy it holds. The gap between
    calls still applies: these tools do reach LinkedIn."""

    async def test_a_bulk_tool_waits_out_the_gap(self, paced):
        await _timed_call(paced, "run_enrichment_bunch")
        second = await _timed_call(paced, "run_enrichment_bunch")

        # Unlike a local-only tool, this one waits: it did reach LinkedIn.
        assert second >= GAP * 0.8


class TestWriteToolsWaitLonger:
    """An invitation or a message is spaced further from the previous call."""

    async def test_a_write_tool_arms_the_longer_gap(self, paced):
        await _timed_call(paced, "send_message")
        second = await _timed_call(paced, "get_inbox")

        # The write band starts at 2.5x the read minimum.
        assert second >= GAP * 2.5 * 0.98

    async def test_a_read_tool_arms_the_shorter_gap(self, paced):
        await _timed_call(paced, "get_inbox")
        second = await _timed_call(paced, "get_conversation")

        assert second < GAP * 2.5 * 1.02


class TestProgressIsReportedThroughTheGap:
    """The reason a longer gap is affordable: the client is never left silent."""

    async def test_the_wait_re_reports_progress(self, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.TOOL_CALL_GAP_SECONDS, "0.3")
        middleware = SequentialToolExecutionMiddleware()
        monkeypatch.setattr(middleware, "_PROGRESS_EVERY_SECONDS", 0.05)
        context = _call_context()
        context.fastmcp_context = MagicMock()
        context.fastmcp_context.report_progress = AsyncMock()

        await middleware.on_call_tool(context, AsyncMock())
        await middleware.on_call_tool(context, AsyncMock())

        pacing = [
            call.kwargs["message"]
            for call in context.fastmcp_context.report_progress.await_args_list
            if call.kwargs["message"].startswith("Pacing LinkedIn activity")
        ]
        # A 0.3s+ gap re-reported every 0.05s is several notifications, not
        # the single one at the start that the old code sent.
        assert len(pacing) >= 3


def _pause_the_account(tmp_path) -> datetime:
    store = JobStore(tmp_path / "jobs")
    now = datetime.now(timezone.utc)
    until = record_throttle_signal(store, now, "http_429")
    assert until is not None
    return until


class TestAPausedAccountRefusesEveryLinkedInCall:
    """The sticky half of issue #57: a signal from one session stops them all.

    The middleware reads the pause from the shared ledger, which is the same
    file the raise sites write, so a 429 seen by any process refuses every
    LinkedIn-touching call from every process until the pause ends.
    """

    async def test_a_linkedin_tool_is_refused_with_the_resume_time(
        self, paced, tmp_path
    ):
        resume_at = _pause_the_account(tmp_path)

        with pytest.raises(ToolError) as raised:
            await paced.on_call_tool(_call_context("get_inbox"), AsyncMock())

        assert "error_type=account_cooldown" in str(raised.value)
        assert resume_at.isoformat(timespec="seconds") in str(raised.value)
        cause = raised.value.__cause__
        assert isinstance(cause, AccountCooldownError)
        assert cause.resume_at == resume_at.isoformat(timespec="seconds")

    async def test_the_refusal_comes_before_the_gap(self, paced, tmp_path):
        await _timed_call(paced, "get_inbox")  # arms the gap
        _pause_the_account(tmp_path)

        started = time.monotonic()
        with pytest.raises(ToolError):
            await paced.on_call_tool(_call_context("get_conversation"), AsyncMock())

        # Faster than the gap it would otherwise have waited out.
        assert time.monotonic() - started < GAP * 0.5

    async def test_the_refusal_comes_before_the_lock(self, paced, tmp_path):
        """A call queued behind a long one is answered while it waits."""
        _pause_the_account(tmp_path)

        async with paced._lock:
            # Refused after the lock, this would sit here until the timeout.
            with pytest.raises(ToolError):
                await asyncio.wait_for(
                    paced.on_call_tool(_call_context("get_inbox"), AsyncMock()),
                    timeout=1.0,
                )

    async def test_a_refused_call_does_not_run_the_tool(self, paced, tmp_path):
        _pause_the_account(tmp_path)
        call_next = AsyncMock()

        with pytest.raises(ToolError):
            await paced.on_call_tool(_call_context("get_inbox"), call_next)

        call_next.assert_not_awaited()

    async def test_a_refused_call_never_takes_the_lease(
        self, paced, tmp_path, monkeypatch
    ):
        """Refused before the profile is touched: the budget is charged per
        navigation, and a call that never takes the lease never navigates."""
        _pause_the_account(tmp_path)
        lease = MagicMock()
        monkeypatch.setattr(
            "linkedin_mcp_server.sequential_tool_middleware.get_profile_lease",
            lambda: lease,
        )
        call_next = AsyncMock()

        with pytest.raises(ToolError):
            await paced.on_call_tool(_call_context("get_inbox"), call_next)

        lease.try_acquire.assert_not_called()
        call_next.assert_not_awaited()

    async def test_a_local_tool_still_answers(self, paced, tmp_path):
        _pause_the_account(tmp_path)
        call_next = AsyncMock()

        await paced.on_call_tool(_call_context("get_enrichment_status"), call_next)

        call_next.assert_awaited_once()

    async def test_the_pause_is_read_fresh_on_every_call(self, paced, tmp_path):
        """Another process wrote it after this one's last call."""
        await _timed_call(paced, "get_inbox")
        _pause_the_account(tmp_path)

        with pytest.raises(ToolError):
            await paced.on_call_tool(_call_context("get_inbox"), AsyncMock())

    async def test_an_expired_pause_lets_the_call_through(self, paced, tmp_path):
        store = JobStore(tmp_path / "jobs")
        _pause_the_account(tmp_path)
        cooldown = read_account_cooldown(store)
        cooldown.until = datetime.now(timezone.utc) - timedelta(seconds=1)
        (store.root / ACCOUNT_COOLDOWN_FILE).write_text(json.dumps(cooldown.to_dict()))
        call_next = AsyncMock()

        await paced.on_call_tool(_call_context("get_inbox"), call_next)

        call_next.assert_awaited_once()

    async def test_the_opt_out_lets_the_call_through(
        self, paced, tmp_path, monkeypatch
    ):
        _pause_the_account(tmp_path)
        monkeypatch.setenv(EnvironmentKeys.ACCOUNT_COOLDOWN_DISABLED, "1")
        call_next = AsyncMock()

        await paced.on_call_tool(_call_context("get_inbox"), call_next)

        call_next.assert_awaited_once()

    async def test_an_unreadable_ledger_lets_the_call_through(self, paced, tmp_path):
        """No evidence of a pause is not a pause."""
        store = JobStore(tmp_path / "jobs")
        store.root.mkdir(parents=True, exist_ok=True)
        (store.root / ACCOUNT_COOLDOWN_FILE).write_text("{not json")
        call_next = AsyncMock()

        await paced.on_call_tool(_call_context("get_inbox"), call_next)

        call_next.assert_awaited_once()


class TestTheHourlyCapRefusesTheSameWay:
    """Forty LinkedIn-touching actions an hour, then the same typed refusal."""

    def _spend(self, tmp_path, count: int, at: datetime | None = None) -> None:
        store = JobStore(tmp_path / "jobs")
        at = at or datetime.now(timezone.utc)
        budget = load_account_budget(store, at)
        for _ in range(count):
            budget.ledger.record(at)
        store.save(budget)

    async def test_the_forty_first_call_is_refused_until_the_oldest_ages_out(
        self, paced, tmp_path
    ):
        oldest = datetime.now(timezone.utc) - timedelta(minutes=50)
        self._spend(tmp_path, 40, at=oldest)

        with pytest.raises(ToolError) as raised:
            await paced.on_call_tool(_call_context("get_inbox"), AsyncMock())

        assert "error_type=account_cooldown" in str(raised.value)
        resume_at = (oldest + timedelta(hours=1)).isoformat(timespec="seconds")
        assert resume_at in str(raised.value)

    async def test_the_fortieth_call_still_runs(self, paced, tmp_path):
        self._spend(tmp_path, 39)
        call_next = AsyncMock()

        await paced.on_call_tool(_call_context("get_inbox"), call_next)

        call_next.assert_awaited_once()

    async def test_actions_older_than_an_hour_release_the_cap(self, paced, tmp_path):
        self._spend(tmp_path, 40, at=datetime.now(timezone.utc) - timedelta(hours=2))
        call_next = AsyncMock()

        await paced.on_call_tool(_call_context("get_inbox"), call_next)

        call_next.assert_awaited_once()

    async def test_the_cap_is_configurable(self, paced, tmp_path, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.HOURLY_ACTIONS_MAX, "2")
        self._spend(tmp_path, 2)

        with pytest.raises(ToolError, match="hourly action cap"):
            await paced.on_call_tool(_call_context("get_inbox"), AsyncMock())

    async def test_the_cap_is_not_a_strike(self, paced, tmp_path):
        self._spend(tmp_path, 40)

        with pytest.raises(ToolError):
            await paced.on_call_tool(_call_context("get_inbox"), AsyncMock())

        cooldown = read_account_cooldown(JobStore(tmp_path / "jobs"))
        assert cooldown.strikes == 0
        assert cooldown.until is None

    async def test_a_bulk_tool_is_let_through_to_plan_against_the_headroom(
        self, paced, tmp_path
    ):
        """It answers with a status dict and a wait, not a ToolError."""
        self._spend(tmp_path, 40)
        call_next = AsyncMock()

        await paced.on_call_tool(_call_context("run_enrichment_bunch"), call_next)

        call_next.assert_awaited_once()

    async def test_the_cap_survives_the_cooldown_opt_out(
        self, paced, tmp_path, monkeypatch
    ):
        monkeypatch.setenv(EnvironmentKeys.ACCOUNT_COOLDOWN_DISABLED, "1")
        self._spend(tmp_path, 40)

        with pytest.raises(ToolError, match="hourly action cap"):
            await paced.on_call_tool(_call_context("get_inbox"), AsyncMock())


class TestArrivalIsRecordedBeforeTheQueue:
    """A bunch's deadline has to see the time its call spent queued.

    The tool timeout starts only once the middleware lets a call through, so a
    call queued behind another session's can already be past the frontend
    proxy's deadline when it starts. The middleware records the arrival for the
    tool to start its deadline from.
    """

    async def test_the_tool_sees_the_arrival_from_before_the_gap(self, paced):
        seen: list[float | None] = []

        async def call_next(context):
            seen.append(request_arrived_at.get())
            return None

        await paced.on_call_tool(_call_context(), AsyncMock())
        await paced.on_call_tool(_call_context(), call_next)

        assert seen[0] is not None
        # The second call sat out the gap between arrival and running, so an
        # arrival stamped after the wait would read as (almost) now.
        assert time.monotonic() - seen[0] >= GAP * 0.8

    async def test_the_arrival_is_reset_after_the_call(self, paced):
        await paced.on_call_tool(_call_context(), AsyncMock())

        assert request_arrived_at.get() is None
