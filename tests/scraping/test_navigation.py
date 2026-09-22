"""Tests for the page navigation lifecycle owner."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import logging

from patchright.async_api import Error as PatchrightError
from patchright.async_api import TimeoutError as PlaywrightTimeoutError

import pytest

from linkedin_mcp_server.config.loaders import EnvironmentKeys
from linkedin_mcp_server.core.exceptions import (
    AuthenticationError,
    ProxyConnectionError,
    RateLimitError,
    TransientBarrierError,
)
from linkedin_mcp_server.exceptions import ActionLimitError
from linkedin_mcp_server.pacing import (
    PROFILE,
    JobStore,
    load_account_budget,
    read_account_cooldown,
)
from linkedin_mcp_server.scraping import navigation as navigation_module
from linkedin_mcp_server.scraping import session as session_module
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.rate_limit import (
    RATE_LIMIT_BACKOFF_DELAY,
    RETRY_AFTER_CEILING,
)
from linkedin_mcp_server.scraping.session import ScrapingSession
from .support.navigation import navigate


def _recorders_registered(page) -> list:
    """Every callback the page was handed, in the order it was handed them."""
    return [call.args[1] for call in page.on.call_args_list]


def _recorders_removed(page) -> list:
    """Every callback the page was asked to drop, in the same order."""
    return [call.args[1] for call in page.remove_listener.call_args_list]


class TestNavigationDiagnostics:
    async def test_goto_with_auth_checks_clicks_remember_me_and_retries(
        self, mock_page
    ):
        navigator = PageNavigator(ScrapingSession(mock_page))

        async def goto_side_effect(*args, **kwargs):
            if mock_page.goto.await_count == 1:
                raise Exception("net::ERR_TOO_MANY_REDIRECTS")
            return None

        mock_page.goto = AsyncMock(side_effect=goto_side_effect)

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                side_effect=[True],
            ) as mock_resolve,
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert mock_page.goto.await_count == 2
        mock_resolve.assert_awaited_once()

    async def test_goto_with_auth_checks_unhooks_outer_listener_before_retry(
        self, mock_page
    ):
        navigator = PageNavigator(ScrapingSession(mock_page))
        listener_events: list[str] = []

        def record_on(event_name, callback):
            listener_events.append(f"on:{event_name}")

        def record_remove(event_name, callback):
            listener_events.append(f"off:{event_name}")

        mock_page.on.side_effect = record_on
        mock_page.remove_listener.side_effect = record_remove

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                side_effect=["account picker", None],
            ),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert listener_events == [
            "on:framenavigated",
            "off:framenavigated",
            "on:framenavigated",
            "off:framenavigated",
        ]

    async def test_goto_with_auth_checks_records_original_failure_before_retry(
        self, mock_page
    ):
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.goto = AsyncMock(
            side_effect=[
                Exception("net::ERR_TOO_MANY_REDIRECTS"),
                Exception("retry failed"),
            ]
        )

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                side_effect=[True, False],
            ),
            patch(
                "linkedin_mcp_server.scraping.navigation.record_page_trace",
                new_callable=AsyncMock,
            ) as mock_trace,
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier",
                new_callable=AsyncMock,
                return_value=None,
            ),
            pytest.raises(Exception, match="retry failed"),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        trace_steps = [call.args[1] for call in mock_trace.await_args_list]
        assert "extractor-navigation-error-before-remember-me-retry" in trace_steps

        trace_call = next(
            call
            for call in mock_trace.await_args_list
            if call.args[1] == "extractor-navigation-error-before-remember-me-retry"
        )
        assert (
            trace_call.kwargs["extra"]["error"]
            == "Exception: net::ERR_TOO_MANY_REDIRECTS"
        )

    async def test_a_hop_on_the_way_reaches_the_failure_log(self, mock_page):
        """Where a failed navigation went is the diagnostic it leaves behind.

        The recorder reads the address off the frame the event carries, so a
        double whose frame never moves records nothing while looking exactly
        like one that works. That the frame and `page.url` agree is not an
        accident this test could catch: patchright's `Page.url` returns
        `self._main_frame.url`, and the frame's `_url` is set before
        `framenavigated` is emitted, so for the main frame the two reads are
        the same value at dispatch time.
        """
        navigator = PageNavigator(ScrapingSession(mock_page))
        checkpoint = "https://www.linkedin.com/checkpoint/challenge/"

        async def goto_then_fail(*args, **kwargs):
            navigate(mock_page, checkpoint)
            raise Exception("net::ERR_ABORTED")

        mock_page.goto = AsyncMock(side_effect=goto_then_fail)

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                navigator,
                "_log_navigation_failure",
                new_callable=AsyncMock,
            ) as mock_log_failure,
            pytest.raises(Exception, match="ERR_ABORTED"),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        logged = mock_log_failure.await_args
        assert logged is not None
        assert logged.args[3] == [checkpoint]

    async def test_goto_with_auth_checks_logs_failure_context(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.goto = AsyncMock(side_effect=Exception("net::ERR_TOO_MANY_REDIRECTS"))

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                navigator,
                "_log_navigation_failure",
                new_callable=AsyncMock,
            ) as mock_log_failure,
            pytest.raises(Exception, match="ERR_TOO_MANY_REDIRECTS"),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        mock_log_failure.assert_awaited_once()
        mock_page.on.assert_called_once()
        mock_page.remove_listener.assert_called_once()


class TestCommittedNavigation:
    """`goto` waits for the commit; the load state is a second, softer wait.

    A LinkedIn SDUI page keeps a request open long after its DOM has
    rendered, so `domcontentloaded` can lag past the budget on a page that is
    already usable (#42). Folding both waits into one `goto` made that lag a
    hard failure and dropped the response the 429 check needs.
    """

    URL = "https://www.linkedin.com/in/testuser/"

    @staticmethod
    def _no_barrier():
        return patch(
            "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
            new_callable=AsyncMock,
            return_value=None,
        )

    @staticmethod
    def _committed_but_never_loaded(page, response=None) -> None:
        """A page whose load state never arrives, the way the browser has it.

        `goto` returns on commit and times out on anything later, so a double
        that answered every `wait_until` alike would let a single combined
        `goto` pass the lag cases it exists to fail.
        """

        async def goto(url, *, wait_until, timeout):
            if wait_until == "commit":
                return response
            raise PlaywrightTimeoutError(f"Timeout {timeout}ms exceeded.")

        page.goto = AsyncMock(side_effect=goto)
        page.wait_for_load_state = AsyncMock(
            side_effect=PlaywrightTimeoutError("Timeout 10000ms exceeded.")
        )

    async def test_goto_commits_and_the_load_state_is_waited_for_separately(
        self, mock_page
    ):
        navigator = PageNavigator(ScrapingSession(mock_page))

        with self._no_barrier():
            await navigator._goto_with_auth_checks(self.URL)

        mock_page.goto.assert_awaited_once_with(
            self.URL, wait_until="commit", timeout=30000
        )
        mock_page.wait_for_load_state.assert_awaited_once_with(
            "domcontentloaded", timeout=int(PageNavigator._LOAD_STATE_TIMEOUT * 1000)
        )

    async def test_a_commit_request_waits_for_nothing_more(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))

        with self._no_barrier():
            await navigator._goto_with_auth_checks(self.URL, wait_until="commit")

        mock_page.goto.assert_awaited_once_with(
            self.URL, wait_until="commit", timeout=30000
        )
        mock_page.wait_for_load_state.assert_not_awaited()

    async def test_a_lagging_load_state_warns_and_proceeds(self, mock_page, caplog):
        navigator = PageNavigator(ScrapingSession(mock_page))
        self._committed_but_never_loaded(mock_page)

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.stabilize_navigation",
                new_callable=AsyncMock,
            ) as stabilize,
            patch(
                "linkedin_mcp_server.scraping.navigation.humanize_after_nav",
                new_callable=AsyncMock,
            ) as humanize,
            patch.object(
                navigator, "_log_navigation_failure", new_callable=AsyncMock
            ) as log_failure,
            patch.object(
                navigator, "_raise_if_auth_barrier", new_callable=AsyncMock
            ) as raise_if_barrier,
            self._no_barrier() as barrier,
            caplog.at_level(logging.WARNING),
        ):
            await navigator._goto_with_auth_checks(self.URL)

        assert any(
            r.levelno == logging.WARNING
            and "lagged behind a committed navigation" in r.getMessage()
            for r in caplog.records
        )
        stabilize.assert_awaited_once()
        humanize.assert_awaited_once_with(mock_page)
        barrier.assert_awaited_once()
        log_failure.assert_not_awaited()
        raise_if_barrier.assert_not_awaited()

    async def test_a_lagging_load_state_keeps_the_response_for_the_429_check(
        self, mock_page
    ):
        """The one wait discarded the response on timeout; the split keeps it."""
        response = MagicMock()
        response.status = 429
        response.headers = {"retry-after": "7"}
        self._committed_but_never_loaded(mock_page, response)
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            pytest.raises(RateLimitError) as raised,
        ):
            await navigator._goto_with_auth_checks(self.URL)

        assert raised.value.suggested_wait_time == 7
        assert navigator._session.rate_limit.rate_limit_hits == 1

    async def test_a_navigation_that_never_commits_still_fails(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.goto = AsyncMock(
            side_effect=PlaywrightTimeoutError("Timeout 30000ms")
        )
        prompt, barrier = _no_prompt_no_barrier()

        with (
            prompt,
            barrier,
            patch.object(
                navigator, "_log_navigation_failure", new_callable=AsyncMock
            ) as log_failure,
            patch(
                "linkedin_mcp_server.scraping.navigation.humanize_after_nav",
                new_callable=AsyncMock,
            ) as humanize,
            pytest.raises(PlaywrightTimeoutError, match="Timeout 30000ms"),
        ):
            await navigator._goto_with_auth_checks(self.URL)

        log_failure.assert_awaited_once()
        humanize.assert_not_awaited()
        mock_page.wait_for_load_state.assert_not_awaited()

    async def test_a_load_state_failure_that_is_not_a_timeout_still_fails(
        self, mock_page
    ):
        """Only the lag is forgiven: a closed page is a navigation failure."""
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.wait_for_load_state = AsyncMock(
            side_effect=PatchrightError(
                "Target page, context or browser has been closed"
            )
        )
        prompt, barrier = _no_prompt_no_barrier()

        with (
            prompt,
            barrier,
            patch.object(
                navigator, "_log_navigation_failure", new_callable=AsyncMock
            ) as log_failure,
            pytest.raises(PatchrightError, match="has been closed"),
        ):
            await navigator._goto_with_auth_checks(self.URL)

        log_failure.assert_awaited_once()

    @pytest.mark.parametrize("lagged", [True, False], ids=["lagged", "on-time"])
    async def test_the_after_goto_trace_says_whether_the_load_state_lagged(
        self, mock_page, lagged
    ):
        navigator = PageNavigator(ScrapingSession(mock_page))
        if lagged:
            self._committed_but_never_loaded(mock_page)

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.record_page_trace",
                new_callable=AsyncMock,
            ) as mock_trace,
            self._no_barrier(),
        ):
            await navigator._goto_with_auth_checks(self.URL)

        after_goto = next(
            call
            for call in mock_trace.await_args_list
            if call.args[1] == "extractor-after-goto"
        )
        assert after_goto.kwargs["extra"]["load_state_lagged"] is lagged


class TestNavigationListenerIdentity:
    """The callback removed has to be the callback that was registered.

    `remove_listener` matches on the object, so removing anything else is a
    silent no-op and the recorder stays attached: one more listener on the
    page per navigation, for the life of the session. Counting the calls
    cannot see that, because the call happens either way. What the page still
    holds afterwards can.
    """

    async def test_a_failed_navigation_leaves_no_recorder_behind(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.goto = AsyncMock(side_effect=Exception("net::ERR_TOO_MANY_REDIRECTS"))

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier",
                new_callable=AsyncMock,
                return_value=None,
            ),
            pytest.raises(Exception, match="ERR_TOO_MANY_REDIRECTS"),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert _recorders_removed(mock_page) == _recorders_registered(mock_page)
        assert mock_page.listeners["framenavigated"] == []

    async def test_a_retry_after_a_failed_navigation_removes_both_recorders(
        self, mock_page
    ):
        """The retry unhooks before it recurses, and the `finally` above it
        then has nothing left to do.

        Removing an object already gone is silent, so a guard that stopped
        working would show up nowhere except in the removals outnumbering the
        registrations.
        """
        navigator = PageNavigator(ScrapingSession(mock_page))

        async def goto_side_effect(*args, **kwargs):
            if mock_page.goto.await_count == 1:
                raise Exception("net::ERR_TOO_MANY_REDIRECTS")
            return None

        mock_page.goto = AsyncMock(side_effect=goto_side_effect)

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert len(_recorders_registered(mock_page)) == 2
        assert _recorders_removed(mock_page) == _recorders_registered(mock_page)
        assert mock_page.listeners["framenavigated"] == []

    async def test_a_retry_behind_a_barrier_removes_both_recorders(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                side_effect=["account picker", None],
            ),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert len(_recorders_registered(mock_page)) == 2
        assert _recorders_removed(mock_page) == _recorders_registered(mock_page)
        assert mock_page.listeners["framenavigated"] == []

    def test_the_watcher_removes_the_recorder_it_registered(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))

        with navigator._watching_navigations():
            navigate(mock_page)

        assert _recorders_removed(mock_page) == _recorders_registered(mock_page)
        assert mock_page.listeners["framenavigated"] == []


class TestRememberMeRetriesOnlyOnce:
    """The second attempt stands on its own, whatever the page keeps showing.

    A prompt that resolves and re-renders on the next load is resolved again,
    and a retry that hands its own permission down recurses until the
    interpreter stops it. That is a `RecursionError` inside a tool call, from
    a page that did nothing but keep asking.
    """

    async def test_a_prompt_behind_a_failing_navigation_retries_once(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.goto = AsyncMock(side_effect=Exception("net::ERR_TOO_MANY_REDIRECTS"))

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_resolve,
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier",
                new_callable=AsyncMock,
                return_value=None,
            ),
            pytest.raises(Exception, match="ERR_TOO_MANY_REDIRECTS") as excinfo,
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert not isinstance(excinfo.value, RecursionError)
        assert mock_page.goto.await_count == 2
        assert mock_resolve.await_count == 1

    async def test_a_prompt_behind_a_standing_barrier_retries_once(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_resolve,
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                return_value="account picker",
            ),
            pytest.raises(AuthenticationError),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        # Target, target again behind the prompt, then the /feed/ second look.
        assert mock_page.goto.await_count == 3
        assert mock_resolve.await_count == 1


class TestABarrierMidScrapeIsBelievedOnlyWhenSeenTwice:
    """With the owner-start probe cached, the first LinkedIn request after a
    restart is a scrape navigation, so this path needs the same second look
    as the probe (issue #9)."""

    FEED = "https://www.linkedin.com/feed/"
    TARGET = "https://www.linkedin.com/in/testuser/"

    def _no_prompt(self):
        return patch(
            "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
            new_callable=AsyncMock,
            return_value=False,
        )

    async def test_a_barrier_that_clears_earns_the_target_one_more_load(
        self, mock_page
    ):
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            self._no_prompt(),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                side_effect=["auth blocker URL: /checkpoint/lg/x", None, None],
            ),
        ):
            await navigator._goto_with_auth_checks(self.TARGET)

        urls = [call.args[0] for call in mock_page.goto.await_args_list]
        assert urls == [self.TARGET, self.FEED, self.TARGET]
        assert mock_page.listeners["framenavigated"] == []

    async def test_a_barrier_seen_twice_raises(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            self._no_prompt(),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                return_value="auth blocker URL: /checkpoint/lg/x",
            ),
            pytest.raises(AuthenticationError),
        ):
            await navigator._goto_with_auth_checks(self.TARGET)

        urls = [call.args[0] for call in mock_page.goto.await_args_list]
        assert urls == [self.TARGET, self.FEED]

    async def test_a_barrier_on_the_retried_target_is_final(self, mock_page):
        # Cleared on /feed/, back on the target: one second look, not a loop.
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            self._no_prompt(),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                side_effect=[
                    "auth blocker URL: /checkpoint/lg/x",
                    None,
                    "auth blocker URL: /checkpoint/lg/x",
                ],
            ),
            pytest.raises(AuthenticationError),
        ):
            await navigator._goto_with_auth_checks(self.TARGET)

        urls = [call.args[0] for call in mock_page.goto.await_args_list]
        assert urls == [self.TARGET, self.FEED, self.TARGET]

    async def test_a_login_redirect_without_li_at_raises_at_once(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.context.cookies = AsyncMock(return_value=[])
        mock_page.url = "https://www.linkedin.com/login"

        with (
            self._no_prompt(),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                return_value="auth blocker URL: https://www.linkedin.com/login",
            ),
            pytest.raises(AuthenticationError),
        ):
            await navigator._goto_with_auth_checks(self.TARGET)

        mock_page.goto.assert_awaited_once()

    async def test_a_barrier_behind_a_failed_load_gets_the_same_look(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.goto = AsyncMock(
            side_effect=[Exception("net::ERR_ABORTED"), None, None]
        )

        with (
            self._no_prompt(),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier",
                new_callable=AsyncMock,
                # The failure log reads it first, then the check, then the
                # second look.
                side_effect=[
                    "auth blocker URL: /checkpoint/lg/x",
                    "auth blocker URL: /checkpoint/lg/x",
                    None,
                ],
            ),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await navigator._goto_with_auth_checks(self.TARGET)

        urls = [call.args[0] for call in mock_page.goto.await_args_list]
        assert urls == [self.TARGET, self.FEED, self.TARGET]

    async def test_a_redirect_loop_reads_no_barrier(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.goto = AsyncMock(side_effect=Exception("net::ERR_TOO_MANY_REDIRECTS"))

        with (
            self._no_prompt(),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier",
                new_callable=AsyncMock,
                return_value="auth blocker URL: /checkpoint/lg/x",
            ),
            pytest.raises(Exception, match="ERR_TOO_MANY_REDIRECTS") as excinfo,
        ):
            await navigator._goto_with_auth_checks(self.TARGET)

        assert not isinstance(excinfo.value, AuthenticationError)
        mock_page.goto.assert_awaited_once()

    async def test_a_picker_only_the_full_read_sees_is_still_expiry(
        self, mock_page, monkeypatch
    ):
        # Sighted by the full detector, so it has to be re-checked by the full
        # detector: the quick one would call it cleared every time and the job
        # readers would report a transient forever instead of a re-login.
        monkeypatch.setattr(
            "linkedin_mcp_server.core.auth.detect_auth_barrier_quick",
            AsyncMock(return_value=None),
        )
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier",
                new_callable=AsyncMock,
                return_value="auth barrier text: welcome back + join now",
            ),
            pytest.raises(AuthenticationError),
        ):
            await navigator._raise_if_auth_barrier(self.TARGET)

    async def test_a_guarded_page_lost_to_a_cleared_barrier_is_not_expiry(
        self, mock_page
    ):
        # The job readers call this after a suspicious reload and would read
        # /feed/ as the list if it returned; it raises, but not as expiry.
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier",
                new_callable=AsyncMock,
                side_effect=["account picker: #rememberme-div", None],
            ),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                return_value=None,
            ),
            pytest.raises(TransientBarrierError, match="Retry the call"),
        ):
            await navigator._raise_if_auth_barrier(self.TARGET)

        urls = [call.args[0] for call in mock_page.goto.await_args_list]
        assert urls == [self.FEED]


class TestWatchingNavigations:
    def test_records_main_frame_hops_without_deduplicating_and_cleans_up(
        self, mock_page
    ):
        navigator = PageNavigator(ScrapingSession(mock_page))

        with navigator._watching_navigations() as hops:
            navigate(mock_page)
            navigate(mock_page)
            for callback in list(mock_page.listeners["framenavigated"]):
                callback(object())

        assert hops == [mock_page.url, mock_page.url]
        assert mock_page.listeners["framenavigated"] == []

    def test_cleans_up_when_the_watched_block_raises(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))

        with pytest.raises(RuntimeError, match="synthetic failure"):
            with navigator._watching_navigations():
                raise RuntimeError("synthetic failure")

        assert mock_page.listeners["framenavigated"] == []


class TestSettleNavigation:
    """The listener decides whether anything happened; the URL cannot."""

    class Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            return self.now

    @staticmethod
    def _sleep(clock, hops, page, schedule=()):
        """Advance the clock per poll, landing each hop at its own moment.

        Each hop replaces the document, which is what a reload and a redirect
        both do. A same-document change is spelled by leaving `time_origin`
        alone instead.
        """
        pending = list(schedule)

        async def sleep(seconds: float) -> None:
            clock.now += seconds
            while pending and pending[0] <= clock.now:
                pending.pop(0)
                hops.append("hop")
                page.time_origin += 1.0

        return sleep

    async def test_a_destroyed_context_reads_as_no_document(self, mock_page):
        """A navigation in flight takes the context the reading needs with it.

        The class patchright raises for that is `Error`, measured, and not a
        `RuntimeError`. A handler narrowed to the latter would turn the
        ordinary case this reading exists for into an unhandled exception,
        so the double is held to the real class.
        """
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.evaluate = AsyncMock(
            side_effect=PatchrightError(
                "Page.evaluate: Execution context was destroyed, "
                "most likely because of a navigation."
            )
        )

        assert await navigator._document_origin() is None

    async def test_a_page_going_nowhere_costs_the_lag_and_not_the_quiet(
        self, mock_page
    ):
        """An ordinary failure has no navigation behind it.

        Charging it the quiet window spends half a second on every DOM error,
        and a call near its tool timeout loses the diagnostic it was about to
        build.
        """
        clock = self.Clock()
        navigator = PageNavigator(ScrapingSession(mock_page))
        hops: list[str] = []

        with (
            patch.object(session_module, "time", clock),
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                side_effect=self._sleep(clock, hops, mock_page),
            ),
        ):
            assert (
                await navigator._settle_navigation(hops, mock_page.time_origin) is False
            )

        assert clock.now < PageNavigator._URL_SETTLE_QUIET
        assert clock.now >= PageNavigator._URL_SETTLE_LAG

    async def test_a_reload_is_a_navigation_though_the_address_holds(self, mock_page):
        """A reload replaces the document and leaves the address alone.

        Comparing addresses calls the replacement the same page, so a picker
        served by a reload was read as search results. The event says so.
        """
        clock = self.Clock()
        navigator = PageNavigator(ScrapingSession(mock_page))
        hops: list[str] = []

        with (
            patch.object(session_module, "time", clock),
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                side_effect=self._sleep(clock, hops, mock_page, [0.05]),
            ),
        ):
            assert (
                await navigator._settle_navigation(hops, mock_page.time_origin) is True
            )

        assert mock_page.wait_for_load_state.await_count == 1

    async def test_a_chain_is_followed_to_its_last_hop(self, mock_page):
        """Hops are counted, not compared.

        A chain that returns to the route it started on reads as one that
        never left, and its last hop is what decides whether this is a
        checkpoint.
        """
        clock = self.Clock()
        navigator = PageNavigator(ScrapingSession(mock_page))
        hops: list[str] = []

        with (
            patch.object(session_module, "time", clock),
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                side_effect=self._sleep(clock, hops, mock_page, [0.05, 0.4]),
            ),
        ):
            assert (
                await navigator._settle_navigation(hops, mock_page.time_origin) is True
            )

        assert len(hops) == 2
        assert clock.now >= 0.4 + PageNavigator._URL_SETTLE_QUIET

    async def test_a_history_change_is_not_a_navigation(self, mock_page):
        """LinkedIn rewrites its own address, and the event cannot tell.

        `pushState`, `replaceState` and a hash change each fire
        `framenavigated` on the main frame, and a search page appends
        `currentJobId` that way by itself. Settling on the event alone charges
        every healthy page the quiet window plus a document wait plus the
        barrier check that follows from it. The document surviving is what
        says nothing was replaced.
        """
        clock = self.Clock()
        navigator = PageNavigator(ScrapingSession(mock_page))
        origin = mock_page.time_origin
        navigate(mock_page, same_document=True)
        hops = ["https://www.linkedin.com/jobs/search/?currentJobId=1"]

        with (
            patch.object(session_module, "time", clock),
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                side_effect=self._sleep(clock, hops, mock_page),
            ),
        ):
            assert await navigator._settle_navigation(hops, origin) is False

        assert clock.now >= PageNavigator._URL_SETTLE_LAG
        assert clock.now < PageNavigator._URL_SETTLE_QUIET
        assert mock_page.wait_for_load_state.await_count == 0

    async def test_a_redirect_behind_a_history_change_is_still_caught(self, mock_page):
        """The address is announced before the checkpoint commits.

        A search page names its selected job the moment a card is chosen, and
        a checkpoint arriving right behind it would be waved through by a
        settler that left on the first hop. The wait is for a replaced
        document, so the second hop is what ends it.
        """
        clock = self.Clock()
        navigator = PageNavigator(ScrapingSession(mock_page))
        origin = mock_page.time_origin
        navigate(mock_page, same_document=True)
        hops = ["https://www.linkedin.com/jobs/search/?currentJobId=1"]

        with (
            patch.object(session_module, "time", clock),
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                side_effect=self._sleep(clock, hops, mock_page, [0.05]),
            ),
        ):
            assert await navigator._settle_navigation(hops, origin) is True

        assert mock_page.wait_for_load_state.await_count == 1


class TestProxyNavigationFailures:
    """A proxy outage during an ordinary tool call is reported as itself."""

    async def test_proxy_error_is_raised_instead_of_a_scraping_failure(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.goto = AsyncMock(
            side_effect=Exception("net::ERR_PROXY_CONNECTION_FAILED at …")
        )

        with pytest.raises(ProxyConnectionError):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

    async def test_proxy_error_is_converted_before_it_reaches_a_trace(self, mock_page):
        # The trace records the raw exception text, which for a proxy failure
        # can quote the proxy URL and put a password into trace.jsonl.
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.goto = AsyncMock(
            side_effect=Exception("net::ERR_TUNNEL_CONNECTION_FAILED")
        )

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.record_page_trace",
                new_callable=AsyncMock,
            ) as mock_trace,
            pytest.raises(ProxyConnectionError),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        recorded = [call.args[1] for call in mock_trace.await_args_list]
        assert "extractor-navigation-error" not in recorded

    async def test_ordinary_navigation_failure_is_unaffected(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.goto = AsyncMock(side_effect=Exception("net::ERR_ABORTED"))

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=False,
            ),
            pytest.raises(Exception) as excinfo,
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert not isinstance(excinfo.value, ProxyConnectionError)


class TestNavigationFailureLogRedaction:
    """The navigation-failure log must not carry proxy credentials.

    It reaches the log even for errors the marker check does not recognise as
    proxy faults, and that log is what users paste into issue reports.
    """

    async def test_credentials_are_redacted_from_the_log(
        self, mock_page, monkeypatch, caplog
    ):
        import logging

        from linkedin_mcp_server.config.schema import AppConfig

        config = AppConfig()
        config.browser.proxy_server = "http://gate.example:7000"
        config.browser.proxy_username = "acctzone9"
        config.browser.proxy_password = "s3cr3t"
        monkeypatch.setattr("linkedin_mcp_server.config.get_config", lambda: config)

        navigator = PageNavigator(ScrapingSession(mock_page))
        # No proxy marker, so it is not converted and reaches the logger.
        mock_page.goto = AsyncMock(
            side_effect=Exception(
                "failed via http://acctzone9:s3cr3t@gate.example:7000"
            )
        )

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=False,
            ),
            caplog.at_level(logging.WARNING),
            pytest.raises(Exception),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert "s3cr3t" not in caplog.text
        assert "acctzone9" not in caplog.text


class TestNavigationFailureCrossesTheToolBoundaryClean:
    """The re-raised exception itself must be credential-free.

    Redacting the extractor's own trace and log is not enough: everything
    downstream logs the exception too, starting with the catch-all in
    error_handler and FastMCP's handler above it.
    """

    async def test_reraised_exception_carries_no_credentials(
        self, mock_page, monkeypatch
    ):
        from linkedin_mcp_server.config.schema import AppConfig

        config = AppConfig()
        config.browser.proxy_server = "http://gate.example:7000"
        config.browser.proxy_username = "acctzone9"
        config.browser.proxy_password = "s3cr3t"
        monkeypatch.setattr("linkedin_mcp_server.config.get_config", lambda: config)

        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.goto = AsyncMock(
            side_effect=Exception(
                "failed via http://acctzone9:s3cr3t@gate.example:7000"
            )
        )

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=False,
            ),
            pytest.raises(Exception) as excinfo,
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert "s3cr3t" not in str(excinfo.value)
        assert "acctzone9" not in str(excinfo.value)
        # The raw error must not survive as a cause either: the handlers
        # downstream print the whole chain.
        assert excinfo.value.__cause__ is None


def _show_the_interstitial(page, status: int) -> None:
    """Leave Chromium's own error page in the tab, the way a refusal does.

    The classifier reads the status off this body, so a test that only makes
    `goto` raise is describing a refusal with no status attached -- which is
    every 4xx and 5xx, not a rate limit.
    """
    page.evaluate = AsyncMock(
        return_value=(
            f"This page isn't working If the problem continues, "
            f"contact the site owner. HTTP ERROR {status} Reload"
        )
    )


def _refused(page, url: str) -> None:
    """Make `goto` raise the way Chromium does for a status it will not commit."""
    page.goto = AsyncMock(
        side_effect=PatchrightError(
            f"Page.goto: net::ERR_HTTP_RESPONSE_CODE_FAILURE at {url}"
        )
    )


def _no_prompt_no_barrier():
    """The ordinary failure path: nothing to resolve, nothing to re-authenticate."""
    return (
        patch(
            "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "linkedin_mcp_server.scraping.navigation.detect_auth_barrier",
            new_callable=AsyncMock,
            return_value=None,
        ),
    )


def _recording_sleep(slept: list[float]):
    return patch(
        "linkedin_mcp_server.scraping.session.asyncio.sleep",
        new_callable=AsyncMock,
        side_effect=lambda delay: slept.append(delay),
    )


def _identity_jitter():
    return patch(
        "linkedin_mcp_server.scraping.session.jitter",
        side_effect=lambda base, *a, **kw: base,
    )


class TestHumanizeAfterNavigation:
    async def test_a_landed_page_gets_cursor_entropy_before_it_is_read(self, mock_page):
        """After the load and before the auth check, so a frozen cursor is
        never what the next read sees."""
        navigator = PageNavigator(ScrapingSession(mock_page))
        order: list[str] = []

        async def humanize(page):
            assert page is mock_page
            order.append("humanize")

        async def barrier(page):
            order.append("barrier")
            return None

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.humanize_after_nav",
                new=humanize,
            ),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new=barrier,
            ),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert order == ["humanize", "barrier"]

    async def test_a_failed_navigation_is_not_humanized(self, mock_page):
        navigator = PageNavigator(ScrapingSession(mock_page))
        mock_page.goto = AsyncMock(side_effect=Exception("net::ERR_ABORTED"))
        prompt, barrier = _no_prompt_no_barrier()

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.humanize_after_nav",
                new_callable=AsyncMock,
            ) as humanize,
            prompt,
            barrier,
            pytest.raises(Exception, match="ERR_ABORTED"),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        humanize.assert_not_awaited()


class TestHttp429Navigation:
    """A 429 that never becomes a page the loaded-page detector could read."""

    async def test_navigation_failure_is_classified_as_rate_limit(
        self, mock_page, tmp_path
    ):
        """The live shape: Chromium refuses the 429 and `goto` raises."""
        url = "https://www.linkedin.com/messaging/thread/2-abc/"
        _refused(mock_page, url)
        _show_the_interstitial(mock_page, 429)
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            pytest.raises(RateLimitError) as raised,
        ):
            await navigator._goto_with_auth_checks(url)

        # No `Retry-After` is readable on this path, so the default stands.
        assert raised.value.suggested_wait_time == 300
        assert "HTTP 429" in str(raised.value)
        # The driver's own text is not carried into anything that logs it.
        assert raised.value.__cause__ is None
        # Classified, not retried: a retry here is one more request into a
        # limit that is still live.
        assert mock_page.goto.await_count == 1
        assert navigator._session.rate_limit.rate_limit_hits == 1
        # And the account is paused for every session, in the shared ledger
        # (isolated to tmp_path by conftest), not only this navigator.
        cooldown = read_account_cooldown(JobStore(tmp_path / "jobs"))
        assert cooldown.last_signal is not None
        assert cooldown.last_signal["signal"] == "http_429"
        assert cooldown.strikes == 1

    async def test_a_refusal_that_is_not_a_429_is_not_a_rate_limit(self, mock_page):
        """The whole reason the interstitial is read.

        Chromium raises the same net error for every status the navigation
        stack refuses. Classifying on the token alone told someone who
        mistyped a username to wait five minutes, and skipped the not-found
        branch in `error_handler` on the way.
        """
        url = "https://www.linkedin.com/in/nosuchuser/"
        _refused(mock_page, url)
        _show_the_interstitial(mock_page, 404)
        navigator = PageNavigator(ScrapingSession(mock_page))
        slept: list[float] = []
        prompt, barrier = _no_prompt_no_barrier()

        # Before the fix this raised `RateLimitError`; the navigation error
        # itself is what the section reader downstream absorbs and reports.
        with (
            _recording_sleep(slept),
            prompt,
            barrier,
            pytest.raises(PatchrightError, match="ERR_HTTP_RESPONSE_CODE_FAILURE"),
        ):
            await navigator._goto_with_auth_checks(url)

        # And it does not pay the rate-limit backoff on the way out.
        assert not slept
        assert navigator._session.rate_limit.rate_limit_hits == 0

    async def test_a_429_elsewhere_on_the_page_is_not_the_status(self, mock_page):
        """The word boundary: a 429 inside a longer number is not a status."""
        url = "https://www.linkedin.com/in/testuser/"
        _refused(mock_page, url)
        mock_page.evaluate = AsyncMock(
            return_value="This page isn't working HTTP ERROR 404 ref 14290"
        )
        navigator = PageNavigator(ScrapingSession(mock_page))
        prompt, barrier = _no_prompt_no_barrier()

        with (
            prompt,
            barrier,
            pytest.raises(PatchrightError, match="ERR_HTTP_RESPONSE_CODE_FAILURE"),
        ):
            await navigator._goto_with_auth_checks(url)

    async def test_an_unreadable_body_is_not_a_rate_limit(self, mock_page):
        """Fails closed: no evidence is not evidence of a limit."""
        url = "https://www.linkedin.com/in/testuser/"
        _refused(mock_page, url)
        mock_page.evaluate = AsyncMock(side_effect=PatchrightError("no document"))
        navigator = PageNavigator(ScrapingSession(mock_page))
        prompt, barrier = _no_prompt_no_barrier()

        with (
            prompt,
            barrier,
            pytest.raises(PatchrightError, match="ERR_HTTP_RESPONSE_CODE_FAILURE"),
        ):
            await navigator._goto_with_auth_checks(url)

    async def test_the_interstitial_alone_is_not_a_rate_limit(self, mock_page):
        """Both halves of the signal, or neither: a 429 on the page behind a
        different net error is a page that happens to say 429."""
        url = "https://www.linkedin.com/in/testuser/"
        mock_page.goto = AsyncMock(
            side_effect=PatchrightError("Page.goto: net::ERR_ABORTED at " + url)
        )
        _show_the_interstitial(mock_page, 429)
        navigator = PageNavigator(ScrapingSession(mock_page))
        prompt, barrier = _no_prompt_no_barrier()

        with prompt, barrier, pytest.raises(PatchrightError, match="ERR_ABORTED"):
            await navigator._goto_with_auth_checks(url)

    async def test_navigation_failure_backs_off_with_jitter(self, mock_page):
        url = "https://www.linkedin.com/messaging/thread/2-abc/"
        _refused(mock_page, url)
        _show_the_interstitial(mock_page, 429)
        slept: list[float] = []
        navigator = PageNavigator(ScrapingSession(mock_page))

        with _recording_sleep(slept), pytest.raises(RateLimitError):
            await navigator._goto_with_auth_checks(url)

        base = RATE_LIMIT_BACKOFF_DELAY
        assert len(slept) == 1
        assert base * 0.5 <= slept[0] <= base * 1.5

    async def test_second_rate_limit_of_a_scrape_backs_off_longer(self, mock_page):
        url = "https://www.linkedin.com/in/testuser/"
        _refused(mock_page, url)
        _show_the_interstitial(mock_page, 429)
        slept: list[float] = []
        navigator = PageNavigator(ScrapingSession(mock_page))

        with _identity_jitter(), _recording_sleep(slept):
            for _ in range(2):
                with pytest.raises(RateLimitError):
                    await navigator._goto_with_auth_checks(url)

        base = RATE_LIMIT_BACKOFF_DELAY
        assert slept == [base, base * 2]
        assert navigator._session.rate_limit.rate_limit_hits == 2

    async def test_the_backoff_is_capped_after_the_jitter(self, mock_page):
        """Jittering the cap first let a 30s maximum sleep 45s."""
        url = "https://www.linkedin.com/in/testuser/"
        _refused(mock_page, url)
        _show_the_interstitial(mock_page, 429)
        slept: list[float] = []
        navigator = PageNavigator(ScrapingSession(mock_page))
        # Hits already spent push the exponent to the doubling cap.
        navigator._session.rate_limit.rate_limit_hits = 8

        with (
            patch(
                "linkedin_mcp_server.scraping.session.jitter",
                side_effect=lambda base, *a, **kw: base * 1.5,
            ),
            _recording_sleep(slept),
            pytest.raises(RateLimitError),
        ):
            await navigator._goto_with_auth_checks(url)

        assert slept == [30.0]

    async def test_committed_429_honors_retry_after(self, mock_page):
        """The other shape: Chromium commits the 429 and `goto` returns it."""
        response = MagicMock()
        response.status = 429
        response.headers = {"retry-after": "120"}
        mock_page.goto = AsyncMock(return_value=response)
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
            pytest.raises(RateLimitError) as raised,
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert raised.value.suggested_wait_time == 120
        assert "120s" in str(raised.value)
        # The backoff is the local pause, never the header: nothing here
        # sleeps for `Retry-After`.
        sleep.assert_awaited_once()
        assert all(call.args[0] < 120 for call in sleep.await_args_list)

    async def test_committed_429_without_retry_after_keeps_the_default_wait(
        self, mock_page
    ):
        response = MagicMock()
        response.status = 429
        response.headers = {}
        mock_page.goto = AsyncMock(return_value=response)
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            pytest.raises(RateLimitError) as raised,
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert raised.value.suggested_wait_time == 300

    async def test_committed_429_relays_a_capped_retry_after(self, mock_page):
        response = MagicMock()
        response.status = 429
        response.headers = {"retry-after": "86400"}
        mock_page.goto = AsyncMock(return_value=response)
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            pytest.raises(RateLimitError) as raised,
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert raised.value.suggested_wait_time == RETRY_AFTER_CEILING

    async def test_committed_200_is_not_a_rate_limit(self, mock_page):
        response = MagicMock()
        response.status = 200
        response.headers = {}
        mock_page.goto = AsyncMock(return_value=response)
        navigator = PageNavigator(ScrapingSession(mock_page))

        with patch(
            "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
            new_callable=AsyncMock,
            return_value=None,
        ) as barrier:
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        barrier.assert_awaited_once()
        assert navigator._session.rate_limit.rate_limit_hits == 0

    async def test_a_driver_that_returns_no_response_is_not_a_rate_limit(
        self, mock_page
    ):
        """`goto` may hand back nothing at all (a scripted page double does);
        no response is no status."""
        mock_page.goto = AsyncMock(return_value=None)
        navigator = PageNavigator(ScrapingSession(mock_page))

        with patch(
            "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert navigator._session.rate_limit.rate_limit_hits == 0


class TestConfigurableBackoff:
    """Each backoff constant is a default an environment variable replaces.

    Read at call time, so `monkeypatch.setenv` inside the test is enough; no
    module reload. Each asserts a value the default cannot produce.
    """

    @staticmethod
    def _hard_limited(mock_page) -> PageNavigator:
        _refused(mock_page, "https://www.linkedin.com/in/testuser/")
        _show_the_interstitial(mock_page, 429)
        return PageNavigator(ScrapingSession(mock_page))

    async def test_backoff_delay_is_read_from_the_environment(
        self, mock_page, monkeypatch
    ):
        monkeypatch.setenv("RATE_LIMIT_BACKOFF_DELAY_SECONDS", "0.5")
        slept: list[float] = []
        navigator = self._hard_limited(mock_page)

        with _identity_jitter(), _recording_sleep(slept), pytest.raises(RateLimitError):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert slept == [0.5]

    async def test_backoff_max_caps_the_first_hit(self, mock_page, monkeypatch):
        """A cap below the base delay is visible on the very first hit."""
        monkeypatch.setenv("RATE_LIMIT_BACKOFF_MAX_SECONDS", "1")
        slept: list[float] = []
        navigator = self._hard_limited(mock_page)

        with _identity_jitter(), _recording_sleep(slept), pytest.raises(RateLimitError):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert slept == [1.0]

    async def test_backoff_max_doublings_stops_the_escalation(
        self, mock_page, monkeypatch
    ):
        """With no doublings allowed, the second hit waits the base delay."""
        monkeypatch.setenv("RATE_LIMIT_BACKOFF_MAX_DOUBLINGS", "0")
        slept: list[float] = []
        navigator = self._hard_limited(mock_page)

        with _identity_jitter(), _recording_sleep(slept):
            for _ in range(2):
                with pytest.raises(RateLimitError):
                    await navigator._goto_with_auth_checks(
                        "https://www.linkedin.com/in/testuser/"
                    )

        base = RATE_LIMIT_BACKOFF_DELAY
        assert slept == [base, base]

    async def test_garbage_falls_back_to_the_default_with_a_warning(
        self, mock_page, monkeypatch, caplog
    ):
        monkeypatch.setenv("RATE_LIMIT_BACKOFF_DELAY_SECONDS", "soon")
        slept: list[float] = []
        navigator = self._hard_limited(mock_page)

        with (
            caplog.at_level(logging.WARNING, logger="linkedin_mcp_server.limits"),
            _identity_jitter(),
            _recording_sleep(slept),
            pytest.raises(RateLimitError),
        ):
            await navigator._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert slept == [RATE_LIMIT_BACKOFF_DELAY]
        assert any(
            "RATE_LIMIT_BACKOFF_DELAY_SECONDS" in r.getMessage()
            and "'soon'" in r.getMessage()
            for r in caplog.records
        )


class TestEveryNavigationIsCharged:
    """One unit of the account budget per page load, taken here (#58).

    The middleware used to charge one unit per tool call, whatever the call
    loaded: a full-sections profile read was fourteen page loads for one
    unit. This is the one path every page load takes, so it is where the
    ledger and LinkedIn's own count agree.
    """

    URL = "https://www.linkedin.com/in/testuser/"

    @staticmethod
    def _ledger(tmp_path):
        # The root the autouse isolation fixture hands the navigator.
        return load_account_budget(JobStore(tmp_path / "jobs"), datetime.now())

    async def test_each_page_load_costs_one_unit_of_its_kind(self, mock_page, tmp_path):
        navigator = PageNavigator(ScrapingSession(mock_page))
        now = datetime.now()

        with TestCommittedNavigation._no_barrier():
            await navigator._goto_with_auth_checks(self.URL)
            await navigator._goto_with_auth_checks(self.URL + "details/skills/")

        budget = self._ledger(tmp_path)
        assert budget.ledger.spent(now) == 2
        assert budget.ledger.spent_kind(PROFILE, now) == 2

    async def test_a_remember_me_retry_is_the_same_page_load(self, mock_page, tmp_path):
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                side_effect=["account picker", None],
            ),
        ):
            await navigator._goto_with_auth_checks(self.URL)

        assert mock_page.goto.await_count == 2
        assert self._ledger(tmp_path).ledger.spent(datetime.now()) == 1

    async def test_a_capped_load_is_refused_before_the_page_is_asked_for(
        self, mock_page, tmp_path, monkeypatch
    ):
        monkeypatch.setenv(EnvironmentKeys.PROFILE_LOADS_MAX, "1")
        navigator = PageNavigator(ScrapingSession(mock_page))

        with TestCommittedNavigation._no_barrier():
            await navigator._goto_with_auth_checks(self.URL)
            with pytest.raises(ActionLimitError, match="limit_exceeded"):
                await navigator._goto_with_auth_checks(self.URL)

        assert mock_page.goto.await_count == 1
        assert self._ledger(tmp_path).ledger.spent(datetime.now()) == 1

    async def test_a_ledger_that_cannot_be_written_costs_the_count_not_the_page(
        self, mock_page, caplog
    ):
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            TestCommittedNavigation._no_barrier(),
            patch.object(
                navigation_module, "JobStore", side_effect=OSError("read-only")
            ),
            caplog.at_level(logging.DEBUG, logger=navigation_module.__name__),
        ):
            await navigator._goto_with_auth_checks(self.URL)

        mock_page.goto.assert_awaited_once()
        assert any("Could not charge" in r.getMessage() for r in caplog.records)
