"""Human-paced action budgeting for bulk work spread over days.

LinkedIn restricts accounts on *behavioral pattern*, not raw volume alone --
150 profiles viewed steadily across a workday reads differently from 150
viewed in half an hour. Bulk jobs therefore cannot run as one long loop; they
have to be a sequence of small bunches, paced apart, stopped overnight and at
weekends, and resumable across process restarts.

This module holds the scheduling arithmetic for that, deliberately free of any
browser or MCP dependency so it can be reasoned about and tested directly.
Every function takes ``now`` explicitly rather than reading the clock, so the
tests do not sleep.

The model mirrors what the established LinkedIn automation tools converged on:

* A **rolling 24-hour** action budget. Not a midnight reset -- each action
  ages out exactly 24 hours after it happened. A midnight reset lets a job
  spend its whole budget at 23:00 and again at 00:01, which is precisely the
  burst shape that gets flagged.
* **Working hours** (09:00-18:00 local, weekends off, lunch skipped, via
  ``Schedule.business_hours()``), because a member who views profiles at
  04:00 on a Sunday is not browsing. A per-work job still defaults to 24/7,
  and the shared account budget once did too: a restrictive default used to
  refuse whole bunches in the evening for operators who had never asked for
  a working-hours limit. That no longer applies, since the schedule no
  longer blocks reads -- it halves their caps outside the window and gates
  only invitations and messages -- so the account budget now defaults to
  business hours; ``WORKING_HOURS_DISABLED`` opts out, and an existing
  ledger keeps whatever schedule it stored.
* **Randomized** gaps and a jittered daily cap, so the traffic carries no
  fixed period and no suspiciously round daily total.
* A **warm-up ramp**, because the pattern change matters as much as the level:
  an account that has never automated jumping straight to 100 views/day is a
  step function.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import math
import os
import random
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from linkedin_mcp_server.common_utils import secure_mkdir, secure_write_text
from linkedin_mcp_server.config.loaders import TRUTHY_VALUES, EnvironmentKeys
from linkedin_mcp_server.exceptions import ActionLimitError
from linkedin_mcp_server.limits import env_float, env_int, env_int_list

try:  # POSIX
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    _HAS_FCNTL = False

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 24 * 60 * 60
WEEK_SECONDS = 7 * WINDOW_SECONDS

#: Monotonic instant (``time.monotonic()``) at which the current MCP tool
#: request reached the middleware -- before it queued for the scraper lock, not
#: after it was let through. A bunch starts its deadline from this so the time
#: spent queued counts against it. ``None`` when no middleware is driving the
#: call (a direct call, a test), in which case the bunch starts from now.
request_arrived_at: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "linkedin_mcp_request_arrived_at", default=None
)

# Every number below is a default; the named environment variable replaces it
# at call time through the accessor next to it. Nothing reads these module
# constants for behaviour, so tests and operators can set the variable without
# racing the import. None of them is an official LinkedIn number -- LinkedIn
# publishes none; they follow what the major automation tools converged on.

# Ceiling across all action types (DAILY_ACTIONS_MAX) and the view-only
# default below it (DAILY_ACTIONS_DEFAULT), since views are cheaper than invites.
MAX_DAILY_ACTIONS = 150
DEFAULT_DAILY_ACTIONS = 100

# Rolling caps per kind of activity, one page load or one submitted write
# each (#58). The variable next to each may lower it and never raise it:
# these are the ceiling, and the daily cap above used to be raisable from the
# environment, which is how a live ledger came to read ``daily_cap: 250``.
PROFILE_LOADS_MAX = 80  # PROFILE_LOADS_MAX, per 24 h
SEARCH_PAGES_MAX = 60  # SEARCH_PAGES_MAX, per 24 h
INVITES_MAX = 20  # INVITES_MAX, per 24 h
INVITES_WEEKLY_MAX = 100  # INVITES_WEEKLY_MAX, per 7 d
MESSAGES_MAX = 50  # MESSAGES_MAX, per 24 h

# Kinds a navigation is filed under, by URL. Only ``profile`` and ``search``
# carry a cap; the rest are counted so the ledger says what the account did.
PROFILE, COMPANY, SEARCH, MESSAGING, FEED, OTHER = (
    "profile",
    "company",
    "search",
    "messaging",
    "feed",
    "other",
)
# Kinds counted at the write, not at a URL: an invite submitted, a message sent.
INVITES, MESSAGES = "invites", "messages"

# Slice shaved off the daily cap by the per-day draw (DAILY_CAP_JITTER):
# uniform(1 - jitter, 1.0), so the total is never a round number.
DAILY_CAP_JITTER = 0.15

# Warm-up ramp for a fresh job: the cap on days before each threshold
# (WARMUP_CAPS / WARMUP_DAYS, comma-separated, same length).
WARMUP_CAPS = (10, 20, 50)
WARMUP_DAYS = (7, 14, 21)

# Gap inside a bunch (STEP_DELAY_MIN_SECONDS / STEP_DELAY_MAX_SECONDS).
DEFAULT_STEP_DELAY = (8.0, 25.0)

# Bunches are spaced to spread the daily budget across the working window,
# jittered by BUNCH_PAUSE_JITTER, then clamped to BUNCH_PAUSE_MIN_SECONDS /
# BUNCH_PAUSE_MAX_SECONDS so the spacing stays plausible either way.
MIN_BUNCH_PAUSE = 60.0
MAX_BUNCH_PAUSE = 3600.0
BUNCH_PAUSE_JITTER = 0.25

# Most profiles one run_enrichment_bunch call may visit (BUNCH_SIZE_MAX) and
# most navigations one enrich_companies call may run (BUNCH_SEARCHES_MAX).
BUNCH_SIZE_MAX = 25
BUNCH_SEARCHES_MAX = 20

# Spacing between two consecutive MCP tool calls, in seconds, drawn
# log-uniformly from one of two bands: READ for tools that only look, WRITE
# for the ones that leave a trace on another member (an invitation, a
# message). Log-uniform rather than uniform so short pauses stay the common
# case while the long ones still happen, which is the skew a person's pauses
# have; a flat draw has a fixed mean and no tail. TOOL_CALL_GAP_SECONDS
# replaces the read minimum and scales every other bound with it, so one knob
# moves the whole shape. Zero turns the spacing off.
#
# This used to be five seconds, chosen so an interactive MCP client did not
# read a silent server as hung. The middleware now reports progress while it
# waits out the gap, so silence is no longer what a longer gap costs, and five
# seconds was a burst LinkedIn answered with 429s and checkpoints (issue #57).
READ_TOOL_CALL_GAP = (8.0, 20.0)
WRITE_TOOL_CALL_GAP = (20.0, 60.0)
DEFAULT_TOOL_CALL_GAP = READ_TOOL_CALL_GAP[0]

# The sticky account cooldown (see `record_throttle_signal`): how long each
# consecutive throttle signal pauses the account, how long without one before
# the count starts over, how close two signals have to be to count as one
# incident, and how close two *half* signals have to be to make one strike.
# ACCOUNT_COOLDOWN_DISABLED turns it off.
COOLDOWN_STEPS_SECONDS = (30 * 60, 2 * 3600, 4 * 3600, 8 * 3600)
COOLDOWN_STRIKES_RESET_SECONDS = 24 * 3600
COOLDOWN_SAME_INCIDENT_SECONDS = 60
COOLDOWN_HALF_SIGNAL_WINDOW_SECONDS = 10 * 60

# LinkedIn-touching actions allowed in any rolling hour across every session
# of the account (HOURLY_ACTIONS_MAX). Counted from the same ledger as the
# daily budget, so an enrichment bunch counts each profile it visited.
HOURLY_ACTIONS_MAX = 40
HOURLY_WINDOW_SECONDS = 3600


def max_daily_actions() -> int:
    """The ceiling on the daily cap; the environment may lower it, never raise it."""
    return _env_at_most(EnvironmentKeys.DAILY_ACTIONS_MAX, MAX_DAILY_ACTIONS)


# Clamps already warned about, as (key, value). The ceilings are read on
# every page load, and an operator with DAILY_ACTIONS_MAX=250 in the
# environment would otherwise see the same line once per navigation.
_clamp_warned: set[tuple[str, int]] = set()


def _env_at_most(key: str, ceiling: int) -> int:
    value = env_int(key, ceiling, minimum=1)
    if value > ceiling:
        if (key, value) not in _clamp_warned:
            _clamp_warned.add((key, value))
            logger.warning("Clamping %s=%d to the ceiling %d", key, value, ceiling)
        return ceiling
    return value


def kind_caps(kind: str) -> tuple[tuple[int, int], ...]:
    """The ``(limit, window seconds)`` pairs `kind` is held to; none for most."""
    if kind == PROFILE:
        return (
            (
                _env_at_most(EnvironmentKeys.PROFILE_LOADS_MAX, PROFILE_LOADS_MAX),
                WINDOW_SECONDS,
            ),
        )
    if kind == SEARCH:
        return (
            (
                _env_at_most(EnvironmentKeys.SEARCH_PAGES_MAX, SEARCH_PAGES_MAX),
                WINDOW_SECONDS,
            ),
        )
    if kind == INVITES:
        return (
            (_env_at_most(EnvironmentKeys.INVITES_MAX, INVITES_MAX), WINDOW_SECONDS),
            (
                _env_at_most(EnvironmentKeys.INVITES_WEEKLY_MAX, INVITES_WEEKLY_MAX),
                WEEK_SECONDS,
            ),
        )
    if kind == MESSAGES:
        return (
            (_env_at_most(EnvironmentKeys.MESSAGES_MAX, MESSAGES_MAX), WINDOW_SECONDS),
        )
    return ()


def working_hours_enforced() -> bool:
    """False only under ``WORKING_HOURS_DISABLED``; the schedule then only paces bunches."""
    raw = os.environ.get(EnvironmentKeys.WORKING_HOURS_DISABLED, "")
    return raw.strip().lower() not in TRUTHY_VALUES


def navigation_kind(url: str) -> str:
    """File a LinkedIn URL under the kind its page load is counted as."""
    path = urlparse(url).path
    # A member's detail pages (`/in/<slug>/details/...`) and overlays sit
    # under the profile path, so one prefix files them all.
    if path.startswith("/in/"):
        return PROFILE
    if path.startswith("/company/"):
        return COMPANY
    # Not `/jobs/search/`: that is a job board page, filed as `other` on
    # purpose, and only the people and company search pages spend this cap.
    if path.startswith("/search/"):
        return SEARCH
    if path.startswith("/messaging/"):
        return MESSAGING
    if path.startswith("/feed/"):
        return FEED
    return OTHER


def default_daily_actions() -> int:
    """The daily cap a job or budget gets when none is given."""
    return env_int(
        EnvironmentKeys.DAILY_ACTIONS_DEFAULT, DEFAULT_DAILY_ACTIONS, minimum=1
    )


def daily_cap_jitter() -> float:
    return _fraction(EnvironmentKeys.DAILY_CAP_JITTER, DAILY_CAP_JITTER)


def bunch_pause_jitter() -> float:
    return _fraction(EnvironmentKeys.BUNCH_PAUSE_JITTER, BUNCH_PAUSE_JITTER)


def _fraction(key: str, default: float) -> float:
    return min(env_float(key, default), 1.0)


def warmup_ramp() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The (caps, day thresholds) pairs of the warm-up ramp."""
    caps = env_int_list(EnvironmentKeys.WARMUP_CAPS, WARMUP_CAPS)
    days = env_int_list(EnvironmentKeys.WARMUP_DAYS, WARMUP_DAYS)
    if len(caps) != len(days):
        logger.warning(
            "Ignoring %s=%r and %s=%r of different lengths; using %s / %s",
            EnvironmentKeys.WARMUP_CAPS,
            caps,
            EnvironmentKeys.WARMUP_DAYS,
            days,
            WARMUP_CAPS,
            WARMUP_DAYS,
        )
        return WARMUP_CAPS, WARMUP_DAYS
    if list(days) != sorted(days):
        logger.warning(
            "Ignoring %s=%r that is not sorted ascending; using %s",
            EnvironmentKeys.WARMUP_DAYS,
            days,
            WARMUP_DAYS,
        )
        return WARMUP_CAPS, WARMUP_DAYS
    return caps, days


def step_delay_range() -> tuple[float, float]:
    return _seconds_range(
        EnvironmentKeys.STEP_DELAY_MIN_SECONDS,
        EnvironmentKeys.STEP_DELAY_MAX_SECONDS,
        DEFAULT_STEP_DELAY,
    )


def bunch_pause_range() -> tuple[float, float]:
    return _seconds_range(
        EnvironmentKeys.BUNCH_PAUSE_MIN_SECONDS,
        EnvironmentKeys.BUNCH_PAUSE_MAX_SECONDS,
        (MIN_BUNCH_PAUSE, MAX_BUNCH_PAUSE),
    )


def _seconds_range(
    min_key: str, max_key: str, default: tuple[float, float]
) -> tuple[float, float]:
    low = env_float(min_key, default[0])
    high = env_float(max_key, default[1])
    if low > high:
        logger.warning(
            "Ignoring %s=%s above %s=%s; using %s", min_key, low, max_key, high, default
        )
        return default
    return low, high


def bunch_size_max() -> int:
    return env_int(EnvironmentKeys.BUNCH_SIZE_MAX, BUNCH_SIZE_MAX, minimum=1)


def bunch_searches_max() -> int:
    return env_int(EnvironmentKeys.BUNCH_SEARCHES_MAX, BUNCH_SEARCHES_MAX, minimum=1)


@dataclass(frozen=True)
class Schedule:
    """When automated work is allowed to run, in the operator's local time.

    The default is permissive (24/7): working-hours restriction is opt-in, not
    imposed. ``business_hours()`` is the 09-18, weekdays, lunch-skipped preset
    for callers who do want the account to look like it only browses in office
    hours. A restrictive default surprised users by refusing to run in the
    evening when they had never asked for a working-hours limit.
    """

    work_start: int = 0
    work_end: int = 24
    lunch_start: int | None = None
    lunch_end: int | None = None
    # Monday is 0, matching datetime.weekday().
    days_off: tuple[int, ...] = ()

    @classmethod
    def business_hours(cls) -> Schedule:
        """The opt-in 09-18, weekdays, lunch-skipped preset."""
        return cls(
            work_start=9, work_end=18, lunch_start=12, lunch_end=13, days_off=(5, 6)
        )

    def is_open(self, now: datetime) -> bool:
        """True when `now` falls inside the working window."""
        if now.weekday() in self.days_off:
            return False
        if not (self.work_start <= now.hour < self.work_end):
            return False
        return not self._in_lunch(now)

    def _in_lunch(self, now: datetime) -> bool:
        if self.lunch_start is None or self.lunch_end is None:
            return False
        return self.lunch_start <= now.hour < self.lunch_end

    def next_open(self, now: datetime) -> datetime:
        """The first instant at or after `now` when work may run.

        Walks forward in hour steps rather than solving analytically -- the
        window is small and irregular (lunch, weekends), and a loop that is
        obviously correct beats arithmetic that is nearly correct.
        """
        if self.is_open(now):
            return now

        candidate = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        # 14 days is far past any weekend-plus-holiday gap this can produce;
        # if nothing opens by then the schedule is misconfigured.
        for _ in range(24 * 14):
            if self.is_open(candidate):
                return candidate
            candidate += timedelta(hours=1)

        raise ValueError(
            "Schedule never opens -- check work_start/work_end/days_off "
            f"(start={self.work_start}, end={self.work_end}, off={self.days_off})"
        )

    def seconds_until_close(self, now: datetime) -> float:
        """Working seconds left today, lunch excluded. 0 when closed."""
        if not self.is_open(now):
            return 0.0

        # work_end == 24 means "midnight" (the always-open default); hour=24 is
        # not a valid time, so express it as start-of-next-day.
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.work_end >= 24:
            close = midnight + timedelta(days=1)
        else:
            close = now.replace(hour=self.work_end, minute=0, second=0, microsecond=0)
        remaining = (close - now).total_seconds()

        # Lunch still ahead today is not usable time.
        if self.lunch_start is not None and self.lunch_end is not None:
            if now.hour < self.lunch_start:
                remaining -= (self.lunch_end - self.lunch_start) * 3600

        return max(remaining, 0.0)


@dataclass
class Ledger:
    """Timestamps of actions performed, as a rolling 24-hour window."""

    actions: list[float] = field(default_factory=list)
    #: The same instants filed by kind, for the per-kind caps. Kept apart from
    #: ``actions`` so the daily cap and each kind's cap are read on their own:
    #: an enrichment bunch writes ``actions`` itself, one per page load, and
    #: the navigation that page load is adds only its kind.
    kinds: dict[str, list[float]] = field(default_factory=dict)

    def prune(self, now: datetime) -> None:
        cutoff = now.timestamp() - WINDOW_SECONDS
        self.actions = [t for t in self.actions if t > cutoff]
        # A week is the longest window any kind is held to.
        week_cutoff = now.timestamp() - WEEK_SECONDS
        self.kinds = {
            kind: [t for t in stamps if t > week_cutoff]
            for kind, stamps in self.kinds.items()
        }

    def record(self, now: datetime) -> None:
        self.actions.append(now.timestamp())

    def record_kind(self, kind: str, now: datetime) -> None:
        self.kinds.setdefault(kind, []).append(now.timestamp())

    def spent_kind(self, kind: str, now: datetime, window: int = WINDOW_SECONDS) -> int:
        cutoff = now.timestamp() - window
        return sum(1 for t in self.kinds.get(kind, ()) if t > cutoff)

    def kind_expiry(
        self, kind: str, now: datetime, window: int = WINDOW_SECONDS
    ) -> float:
        """Seconds until the oldest `kind` entry inside `window` ages out."""
        cutoff = now.timestamp() - window
        inside = [t for t in self.kinds.get(kind, ()) if t > cutoff]
        if not inside:
            return 0.0
        return max(min(inside) + window - now.timestamp(), 0.0)

    def spent(self, now: datetime) -> int:
        self.prune(now)
        return len(self.actions)

    def remaining(self, now: datetime, cap: int) -> int:
        return max(cap - self.spent(now), 0)

    def next_expiry(self, now: datetime) -> float:
        """Seconds until the oldest action ages out of the window.

        This is how long a budget-exhausted job must wait before it regains
        even one unit of headroom.
        """
        self.prune(now)
        if not self.actions:
            return 0.0
        return max(min(self.actions) + WINDOW_SECONDS - now.timestamp(), 0.0)


def warmup_cap(base_cap: int, started_on: date, today: date) -> int:
    """Ramp a fresh job up to `base_cap` over four weeks.

    The published warm-up schedules all share this shape: a fortnight of
    visibly low volume, then a climb. The exact numbers matter less than not
    presenting LinkedIn with a step change.
    """
    days = (today - started_on).days
    if days < 0:
        days = 0
    caps, thresholds = warmup_ramp()
    for cap, threshold in zip(caps, thresholds):
        if days < threshold:
            return min(cap, base_cap)
    return base_cap


def jittered_cap(cap: int, today: date, salt: str = "") -> int:
    """Shave a stable, per-day random slice off the cap.

    A job that stops at exactly 100 every single day advertises itself. The
    draw is seeded by date so every call within a day agrees -- otherwise the
    effective cap would wobble between calls and the job could overshoot.
    """
    rng = random.Random(f"{salt}:{today.isoformat()}")
    return max(1, int(cap * rng.uniform(1.0 - daily_cap_jitter(), 1.0)))


def next_bunch_delay(
    remaining_budget: int,
    bunch_size: int,
    now: datetime,
    schedule: Schedule,
    rng: random.Random | None = None,
) -> float:
    """Seconds to wait before running the next bunch.

    Spreads whatever budget is left across the working time left today, so the
    job finishes the day's allowance around closing time instead of racing
    through it by lunch.
    """
    rng = rng or random.Random()
    min_pause, max_pause = bunch_pause_range()

    if remaining_budget <= 0:
        return max_pause

    open_seconds = schedule.seconds_until_close(now)
    if open_seconds <= 0:
        return max_pause

    bunches_left = max(math.ceil(remaining_budget / max(bunch_size, 1)), 1)
    base = open_seconds / bunches_left
    jitter = bunch_pause_jitter()
    jittered = base * rng.uniform(1.0 - jitter, 1.0 + jitter)
    return max(min_pause, min(jittered, max_pause))


def step_delay(
    delay_range: tuple[float, float] | None = None,
    rng: random.Random | None = None,
) -> float:
    """A randomized gap between two profile loads inside one bunch."""
    rng = rng or random.Random()
    low, high = delay_range or step_delay_range()
    return rng.uniform(low, high)


def tool_call_gap(
    raw: str | None = None,
    rng: random.Random | None = None,
    *,
    write: bool = False,
) -> float:
    """Seconds to leave between the end of one tool call and the next.

    ``raw`` is the configured read minimum in seconds as it arrives from the
    environment, unparsed; anything unusable falls back to the default rather
    than removing the spacing, since a typo must not be the way pacing is
    turned off. An explicit ``0`` is that way. ``write`` picks the longer band
    for a call that leaves a trace on another member.
    """
    minimum = _configured_tool_call_gap(raw)
    if minimum <= 0:
        return 0.0
    scale = minimum / READ_TOOL_CALL_GAP[0]
    low, high = WRITE_TOOL_CALL_GAP if write else READ_TOOL_CALL_GAP
    return _log_uniform(low * scale, high * scale, rng or random.Random())


def _log_uniform(low: float, high: float, rng: random.Random) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _configured_tool_call_gap(raw: str | None) -> float:
    if raw is None or not raw.strip():
        return DEFAULT_TOOL_CALL_GAP
    try:
        seconds = float(raw)
    except ValueError:
        logger.warning(
            "Ignoring non-numeric tool-call gap %r; using %ss",
            raw,
            DEFAULT_TOOL_CALL_GAP,
        )
        return DEFAULT_TOOL_CALL_GAP
    if seconds < 0:
        logger.warning(
            "Ignoring negative tool-call gap %r; using %ss",
            raw,
            DEFAULT_TOOL_CALL_GAP,
        )
        return DEFAULT_TOOL_CALL_GAP
    return seconds


@dataclass
class Job:
    """A resumable bulk job: its queue, its results, and its action ledger."""

    name: str
    started_on: date
    pending: list[str] = field(default_factory=list)
    done: dict[str, Any] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    # Consecutive empty-page visits per pending username, so a profile that
    # never loads can be struck out instead of blocking the queue for good.
    strikes: dict[str, int] = field(default_factory=dict)
    ledger: Ledger = field(default_factory=Ledger)
    daily_cap: int = field(default_factory=default_daily_actions)
    schedule: Schedule = field(default_factory=Schedule)
    warmup: bool = True

    def effective_cap(self, now: datetime) -> int:
        """Today's cap after the warm-up ramp and the daily jitter."""
        cap = min(self.daily_cap, max_daily_actions())
        if self.warmup:
            cap = warmup_cap(cap, self.started_on, now.date())
        return jittered_cap(cap, now.date(), salt=self.name)

    def remaining_today(self, now: datetime) -> int:
        return self.ledger.remaining(now, self.effective_cap(now))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "started_on": self.started_on.isoformat(),
            "pending": self.pending,
            "done": self.done,
            "failed": self.failed,
            "strikes": self.strikes,
            "actions": self.ledger.actions,
            "kinds": self.ledger.kinds,
            "daily_cap": self.daily_cap,
            "warmup": self.warmup,
            "schedule": {
                "work_start": self.schedule.work_start,
                "work_end": self.schedule.work_end,
                "lunch_start": self.schedule.lunch_start,
                "lunch_end": self.schedule.lunch_end,
                "days_off": list(self.schedule.days_off),
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Job:
        sched_raw = raw.get("schedule") or {}
        schedule = Schedule(
            work_start=sched_raw.get("work_start", 9),
            work_end=sched_raw.get("work_end", 18),
            lunch_start=sched_raw.get("lunch_start", 12),
            lunch_end=sched_raw.get("lunch_end", 13),
            days_off=tuple(sched_raw.get("days_off", (5, 6))),
        )
        return cls(
            name=raw["name"],
            started_on=date.fromisoformat(raw["started_on"]),
            pending=list(raw.get("pending", [])),
            done=dict(raw.get("done", {})),
            failed=dict(raw.get("failed", {})),
            strikes=dict(raw.get("strikes", {})),
            ledger=Ledger(
                actions=list(raw.get("actions", [])),
                kinds={k: list(v) for k, v in (raw.get("kinds") or {}).items()},
            ),
            # Clamped on read as well as in effective_cap: a ledger written
            # while the environment could still raise the ceiling reads 250.
            daily_cap=min(
                raw.get("daily_cap", default_daily_actions()), max_daily_actions()
            ),
            schedule=schedule,
            warmup=raw.get("warmup", True),
        )


class JobStore:
    """Reads and writes jobs as JSON, one file per job.

    Persisted after every single profile rather than at the end of a bunch: a
    crash or a kill mid-bunch should cost at most one duplicated page view,
    never the day's progress.

    The default root is under the home directory, not ``USER_DATA_DIR``, so
    the account budget it holds is shared by every browser profile of one
    user; the limits in ``limits.py`` are per process but the ledger they
    bound is not.
    """

    def __init__(self, root: Path | str = "~/.linkedin-mcp/jobs") -> None:
        self.root = Path(root).expanduser()

    def locked(self) -> contextlib.AbstractContextManager[None]:
        """Hold the root's lock across a read-modify-write of one record.

        ``load`` and ``save`` each take it on their own, which keeps a reader
        off a half-written file; a caller that loads, changes and saves has
        to hold it across all three or another process's save lands in
        between and one of the two writes is lost.
        """
        return _ledger_lock(self.root)

    def _path(self, name: str) -> Path:
        # The filename is the name verbatim, so distinct names cannot collide.
        # Stripping unsafe characters instead (the old behaviour) aliased
        # "a/b" and "ab" to one file, silently merging two jobs' state -- and
        # let a user job overwrite the private budget record. Reject anything
        # outside the safe set rather than lossily map it.
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(
                f"Job name {name!r} may contain only letters, digits, '-' or "
                "'_' (no spaces, slashes or other characters), so that distinct "
                "names cannot collide on disk."
            )
        return self.root / f"{name}.json"

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def load(self, name: str) -> Job:
        path = self._path(name)
        with self.locked():
            if not path.exists():
                raise FileNotFoundError(f"No job named {name!r} at {path}")
            return Job.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, job: Job) -> None:
        path = self._path(job.name)
        # Write-then-rename through a per-process temp name: a crash mid-write
        # leaves the previous good file rather than a truncated one, and two
        # processes saving at once cannot rename each other's half-written
        # temp file into place, which one shared temp name let them do.
        with self.locked():
            secure_write_text(path, json.dumps(job.to_dict(), indent=2))

    def list_jobs(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.stem for p in self.root.glob("*.json") if p.name != ACCOUNT_COOLDOWN_FILE
        )


#: Roots this process currently holds the ledger lock on, with a depth, so
#: a caller holding it across a load-modify-save does not deadlock on the
#: lock the load and the save take for themselves. ``flock`` conflicts
#: between two descriptors of one process exactly as between two processes.
_LEDGER_LOCK_DEPTH: dict[Path, int] = {}


@contextmanager
def _ledger_lock(root: Path) -> Iterator[None]:
    """An exclusive, blocking lock on the records under ``root``.

    One lock file for the whole directory rather than one per record: the
    critical sections are a few milliseconds of JSON, and the cooldown file
    is read on every tool call by every process, so contention is cheap and
    a single file keeps the ordering obvious. Released by the kernel when
    the process dies, so a crash cannot wedge it. Re-entrant within the
    process; the middleware already serialises tool calls inside one.

    On Windows there is no ``fcntl``; the writes are still atomic renames, so
    a reader never sees a torn file, but two writers can lose an update.
    """
    root = root.resolve()
    if _LEDGER_LOCK_DEPTH.get(root, 0) > 0:
        _LEDGER_LOCK_DEPTH[root] += 1
        try:
            yield
        finally:
            _LEDGER_LOCK_DEPTH[root] -= 1
        return

    secure_mkdir(root)
    fd = os.open(root / LEDGER_LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if _HAS_FCNTL:
            fcntl.flock(fd, fcntl.LOCK_EX)
        _LEDGER_LOCK_DEPTH[root] = 1
        try:
            yield
        finally:
            del _LEDGER_LOCK_DEPTH[root]
    finally:
        if _HAS_FCNTL:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# LinkedIn counts activity per *account*, not per job -- it has no idea two
# scrapes belong to different "jobs" of ours. So the safety budget has to be
# one shared ledger that every LinkedIn-touching operation draws down,
# regardless of which queue it serves. This well-known job holds that single
# account-wide budget: its ledger, cap, warm-up and schedule. Its queue stays
# empty; per-work queues live in their own jobs.
ACCOUNT_BUDGET_JOB = "__account_budget__"

# The account cooldown lives next to the budget, not in it. It sat inside the
# budget record once, and the enrichment tools -- which load the budget at
# the top of a bunch and save that same object after every profile -- wrote
# the stale copy back over a pause another process had just recorded.
# Reproduced. A record nobody holds in memory cannot be clobbered that way.
ACCOUNT_COOLDOWN_FILE = "__account_cooldown__.json"
LEDGER_LOCK_FILE = ".ledger.lock"
#: The job name the cooldown file would answer to; reserved like the budget.
ACCOUNT_COOLDOWN_JOB = "__account_cooldown__"


def load_account_budget(
    store: JobStore,
    now: datetime,
    *,
    daily_cap: int | None = None,
    warmup: bool | None = None,
    schedule: Schedule | None = None,
) -> Job:
    """Load the one shared account budget, creating or reconfiguring it.

    A non-None ``daily_cap``/``warmup``/``schedule`` updates the stored budget
    (last writer wins) and is persisted, so a caller can set the account-wide
    cap once and every subsystem then honours it. With all three None this is a
    pure read (still materialising a default budget on first use, on business
    hours -- see the module docstring for why that default is safe here).
    """
    with store.locked():
        if store.exists(ACCOUNT_BUDGET_JOB):
            budget = store.load(ACCOUNT_BUDGET_JOB)
            changed = False
            if daily_cap is not None and daily_cap != budget.daily_cap:
                budget.daily_cap = daily_cap
                changed = True
            if warmup is not None and warmup != budget.warmup:
                budget.warmup = warmup
                changed = True
            if schedule is not None and schedule != budget.schedule:
                budget.schedule = schedule
                changed = True
            if changed:
                store.save(budget)
            return budget

        budget = Job(
            name=ACCOUNT_BUDGET_JOB,
            started_on=now.date(),
            daily_cap=daily_cap if daily_cap is not None else default_daily_actions(),
            warmup=warmup if warmup is not None else False,
            schedule=schedule if schedule is not None else Schedule.business_hours(),
        )
        store.save(budget)
        return budget


#: The account budget a bulk tool holds in memory for the length of its call.
#: Such a tool records ``actions`` itself, one per page load, and saves its
#: copy after every profile -- so a navigation inside it must add its kind to
#: that copy, not to the file the tool is about to overwrite. ``None`` when no
#: such tool is running, and the charge goes straight to disk. Set by the tool
#: once it has loaded its budget and reset by the same tool on its way out.
account_budget_in_use: contextvars.ContextVar[Job | None] = contextvars.ContextVar(
    "linkedin_mcp_account_budget_in_use", default=None
)

#: Set alongside ``account_budget_in_use`` by a bulk tool called with
#: ``ignore_schedule``: its page loads run on the full caps, not the halved
#: off-hours ones, since the caller asked for the catch-up knowingly.
schedule_ignored: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "linkedin_mcp_schedule_ignored", default=False
)


def _window_label(window: int) -> str:
    return "7 d" if window >= WEEK_SECONDS else "24 h"


def _effective_caps(
    budget: Job, kind: str, now: datetime, *, ignore_schedule: bool
) -> tuple[bool, list[tuple[int, int]]]:
    """Whether the schedule is closed, and `kind`'s caps as they apply now.

    A read outside the window is let through on half its cap: a member who
    browses at 03:00 is odd, one who sends invitations then is odder.
    """
    off_hours = (
        working_hours_enforced()
        and not ignore_schedule
        and not budget.schedule.is_open(now)
    )
    caps = [
        (max(limit // 2, 1) if off_hours else limit, window)
        for limit, window in kind_caps(kind)
    ]
    return off_hours, caps


def kind_headroom(
    budget: Job, kind: str, now: datetime, *, ignore_schedule: bool = False
) -> tuple[int, float]:
    """Units of `kind` left under its tightest cap, and seconds until one frees.

    For a bulk tool to plan against before it starts, the way it plans
    against the daily cap: a bunch must not start what it cannot finish.
    Uncapped kinds report the daily cap's own ceiling and no wait.
    """
    off_hours, caps = _effective_caps(
        budget, kind, now, ignore_schedule=ignore_schedule
    )
    headroom: int | None = None
    wait = 0.0
    for limit, window in caps:
        left = limit - budget.ledger.spent_kind(kind, now, window)
        if headroom is None or left < headroom:
            headroom = left
            wait = budget.ledger.kind_expiry(kind, now, window)
    if headroom is None:
        return MAX_DAILY_ACTIONS, 0.0
    if off_hours:
        # The full cap is back the moment the schedule reopens.
        wait = min(wait, (budget.schedule.next_open(now) - now).total_seconds())
    return max(headroom, 0), wait


def refuse_if_limited(
    budget: Job, kind: str, now: datetime, *, ignore_schedule: bool = False
) -> None:
    """Raise ``ActionLimitError`` before one more `kind` would break a cap.

    Outside the schedule a write waits for it to reopen, and a read is let
    through on half its cap. Nothing is charged for a refusal.
    """
    off_hours, caps = _effective_caps(
        budget, kind, now, ignore_schedule=ignore_schedule
    )
    if off_hours and kind in (INVITES, MESSAGES):
        raise ActionLimitError(
            kind,
            limit=0,
            window="working hours",
            resume_at=budget.schedule.next_open(now),
        )
    for limit, window in caps:
        if budget.ledger.spent_kind(kind, now, window) < limit:
            continue
        resume_at = now + timedelta(
            seconds=budget.ledger.kind_expiry(kind, now, window)
        )
        if off_hours:
            # The full cap is back the moment the schedule reopens.
            resume_at = min(resume_at, budget.schedule.next_open(now))
        raise ActionLimitError(
            kind, limit=limit, window=_window_label(window), resume_at=resume_at
        )


def charge_navigation(store: JobStore, url: str, now: datetime) -> None:
    """Spend one page load of the shared account budget, refusing it first.

    Called once per navigation, which is what LinkedIn counts: a fourteen
    section profile read is fourteen loads, not one call.
    """
    kind = navigation_kind(url)
    held = account_budget_in_use.get()
    if held is not None:
        refuse_if_limited(held, kind, now, ignore_schedule=schedule_ignored.get())
        held.ledger.record_kind(kind, now)
        return
    # Held across the read and the write: another process's navigation
    # landing in the same instant would otherwise overwrite this one with
    # its own copy of the ledger, and one of the two is lost.
    with store.locked():
        budget = load_account_budget(store, now)
        refuse_if_limited(budget, kind, now, ignore_schedule=schedule_ignored.get())
        budget.ledger.record_kind(kind, now)
        budget.ledger.record(now)
        # Nothing else on this path prunes, and the file would carry every
        # page load of the account's life.
        budget.ledger.prune(now)
        store.save(budget)


def refuse_action(store: JobStore, kind: str, now: datetime) -> None:
    """Refuse a write before it is attempted when its cap or the schedule says so.

    Best-effort on the ledger, like the navigation charge: a home directory
    that cannot be read refuses nothing, since nothing has been sent yet.
    """
    try:
        budget = load_account_budget(store, now)
    except Exception:
        logger.warning(
            "Could not read the account budget; allowing the %s", kind, exc_info=True
        )
        return
    refuse_if_limited(budget, kind, now)


def record_action(store: JobStore, kind: str, now: datetime) -> None:
    """Count one write that was submitted, or may have been.

    Best-effort: the write has happened, and a ledger that cannot be saved
    must cost the count, never turn a delivered message into a tool error.
    """
    try:
        with store.locked():
            budget = load_account_budget(store, now)
            budget.ledger.record_kind(kind, now)
            budget.ledger.prune(now)
            store.save(budget)
    except Exception:
        logger.warning(
            "Could not count the %s against the account budget", kind, exc_info=True
        )


# --- Sticky account cooldown and the hourly cap ----------------------------
#
# A throttle signal (an HTTP 429, a checkpoint, a rate-limit page) used to
# raise `RateLimitError` and nothing else, so the calling agent, and every
# other session sharing the daemon, retried at once: client logs showed ten
# calls in under five seconds right before an incident (issue #57). One signal
# now pauses the *account*, in a file every session reads, and the pause
# escalates with each further signal until a day passes without one.


def account_cooldown_disabled() -> bool:
    raw = os.environ.get(EnvironmentKeys.ACCOUNT_COOLDOWN_DISABLED, "")
    return raw.strip().lower() in TRUTHY_VALUES


def hourly_actions_max() -> int:
    return env_int(EnvironmentKeys.HOURLY_ACTIONS_MAX, HOURLY_ACTIONS_MAX, minimum=1)


def _utc(now: datetime) -> datetime:
    """`now` as an aware UTC instant.

    A naive value is taken as local time, which is what ``datetime.now()``
    hands the rest of this module.
    """
    return now.astimezone(timezone.utc)


def _parse_utc(raw: Any) -> datetime | None:
    """An ISO-8601 instant from the cooldown file, or None for anything else.

    Tolerant on purpose: the file is documented as hand-editable, and a key
    nulled or mistyped by hand must read as "no cooldown", never as a broken
    ledger that stops every tool.
    """
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    # A hand-edited value without an offset is read as UTC, which is what the
    # server writes; comparing it as local time would shift the pause by the
    # operator's offset in whichever direction they did not expect.
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class AccountCooldown:
    """The contents of ``__account_cooldown__.json``."""

    until: datetime | None = None
    strikes: int = 0
    last_signal: dict[str, str] | None = None
    #: When the last unconfirmed half signal landed; see `record_throttle_signal`.
    half_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "until": self.until.isoformat() if self.until else None,
            "strikes": self.strikes,
            "last_signal": self.last_signal,
            "half_at": self.half_at.isoformat() if self.half_at else None,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> AccountCooldown:
        if not isinstance(raw, dict):
            return cls()
        strikes = raw.get("strikes")
        last = raw.get("last_signal")
        return cls(
            until=_parse_utc(raw.get("until")),
            strikes=strikes if isinstance(strikes, int) and strikes > 0 else 0,
            last_signal=last if isinstance(last, dict) else None,
            half_at=_parse_utc(raw.get("half_at")),
        )


def read_account_cooldown(store: JobStore) -> AccountCooldown:
    """The cooldown as it is on disk right now; empty when unreadable.

    Read fresh on every call rather than cached: the pause is written by
    whichever process saw LinkedIn push back, so any copy held in memory is
    stale by definition.
    """
    path = store.root / ACCOUNT_COOLDOWN_FILE
    try:
        with store.locked():
            raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return AccountCooldown()
    except (OSError, ValueError):
        logger.warning("Ignoring unreadable %s; treating as no cooldown", path)
        return AccountCooldown()
    return AccountCooldown.from_dict(raw)


def _write_account_cooldown(store: JobStore, cooldown: AccountCooldown) -> None:
    secure_write_text(
        store.root / ACCOUNT_COOLDOWN_FILE, json.dumps(cooldown.to_dict(), indent=2)
    )


def cooldown_resume_at(store: JobStore, now: datetime) -> datetime | None:
    """When the current account pause ends, or None when nothing is paused."""
    until = read_account_cooldown(store).until
    return until if until is not None and until > _utc(now) else None


def record_throttle_signal(
    store: JobStore, now: datetime, signal: str, *, half: bool = False
) -> datetime | None:
    """Pause the account after LinkedIn pushed back, and say until when.

    Each signal within a day of the previous one is one more strike, and the
    pause steps up through ``COOLDOWN_STEPS_SECONDS`` with the count; a day
    without a signal starts the count over. Signals within a minute of each
    other are one incident and one strike: a 429 is usually seen by more
    than one layer of the same call. A pause already in force is never
    shortened, because another session may have recorded a later signal and
    the longer of the two is the one LinkedIn is judging the account by.

    A ``half`` signal is one that is ambiguous with a slow network, such as a
    messaging payload that never arrived. On its own it pauses nothing; two
    within ``COOLDOWN_HALF_SIGNAL_WINDOW_SECONDS`` make one strike. Returns
    None when nothing was paused.

    The whole read-modify-write runs under the ledger lock, so two processes
    recording at once escalate once each rather than both writing strike 1.
    """
    now = _utc(now)
    with store.locked():
        cooldown = read_account_cooldown(store)
        if half:
            if (
                cooldown.half_at is None
                or (now - cooldown.half_at).total_seconds()
                > COOLDOWN_HALF_SIGNAL_WINDOW_SECONDS
            ):
                cooldown.half_at = now
                _write_account_cooldown(store, cooldown)
                logger.info(
                    "Possible throttle (%s); a second one within %d min pauses "
                    "the account",
                    signal,
                    COOLDOWN_HALF_SIGNAL_WINDOW_SECONDS // 60,
                )
                return None
        # Spent by this strike, whether it completed a pair or a full signal
        # arrived instead: a half after a full strike is not strike two.
        cooldown.half_at = None

        strikes = cooldown.strikes
        since_last = None
        if cooldown.last_signal is not None:
            last_at = _parse_utc(cooldown.last_signal.get("at"))
            if last_at is not None:
                since_last = (now - last_at).total_seconds()
        if since_last is None or since_last > COOLDOWN_STRIKES_RESET_SECONDS:
            strikes = 0
        same_incident = (
            since_last is not None and since_last <= COOLDOWN_SAME_INCIDENT_SECONDS
        )
        if not same_incident:
            strikes += 1
        strikes = max(strikes, 1)
        step = COOLDOWN_STEPS_SECONDS[min(strikes, len(COOLDOWN_STEPS_SECONDS)) - 1]
        until = now + timedelta(seconds=step)
        in_force = cooldown.until
        if in_force is not None and in_force <= now:
            in_force = None
        if in_force is not None and (same_incident or in_force > until):
            until = in_force

        cooldown.until = until
        cooldown.strikes = strikes
        cooldown.last_signal = {"signal": signal, "at": now.isoformat()}
        _write_account_cooldown(store, cooldown)
    logger.warning(
        "LinkedIn pushed back (%s); pausing the account until %s (strike %d)",
        signal,
        until.isoformat(timespec="seconds"),
        strikes,
    )
    return until


def note_throttle_signal(signal: str, *, half: bool = False) -> None:
    """Record a throttle signal against the default account ledger.

    For the places that detect throttling and hold no store of their own.
    Best-effort, like every other write to the ledger: a home directory that
    cannot be written must not turn one refused page into a second error.
    """
    if account_cooldown_disabled():
        return
    try:
        record_throttle_signal(
            JobStore(), datetime.now(timezone.utc), signal, half=half
        )
    except Exception:
        logger.debug("Could not record the throttle signal", exc_info=True)


def _actions_in_the_last_hour(budget: Job, now: datetime) -> list[float]:
    cutoff = _utc(now).timestamp() - HOURLY_WINDOW_SECONDS
    return sorted(t for t in budget.ledger.actions if t > cutoff)


def hourly_headroom(budget: Job, now: datetime, cap: int | None = None) -> int:
    """Actions the rolling-hour cap still admits right now.

    For the bulk tools, which plan a bunch up front: a bunch planned past
    the headroom would run through the cap in its middle, where nothing
    checks it.
    """
    cap = hourly_actions_max() if cap is None else cap
    return max(cap - len(_actions_in_the_last_hour(budget, now)), 0)


def hourly_cap_resume_at(
    budget: Job, now: datetime, cap: int | None = None, *, needed: int = 1
) -> datetime | None:
    """When the rolling-hour cap next admits ``needed`` actions, or None if now.

    Not a strike: the cap is this server holding itself back, not LinkedIn
    pushing back, so it neither escalates nor persists anything. The answer
    is the moment enough of the hour's oldest actions have aged out of it.
    """
    cap = hourly_actions_max() if cap is None else cap
    recent = _actions_in_the_last_hour(budget, now)
    # How many of the hour's actions have to expire before `needed` fit.
    must_expire = len(recent) - cap + needed
    if must_expire <= 0:
        return None
    if not recent:
        # Nothing to wait for: `needed` exceeds the cap itself, which no
        # amount of waiting fixes. The callers name that as configuration.
        return None
    # The k-th oldest is the one whose expiry frees the last of them;
    # everything older has expired by then anyway. Clamped: past the last
    # entry the answer is "when the whole hour has drained", and the cap
    # itself is the limit after that.
    k = min(must_expire, len(recent))
    return datetime.fromtimestamp(
        recent[k - 1] + HOURLY_WINDOW_SECONDS, tz=timezone.utc
    )


def seconds_until_hourly_release(budget: Job, now: datetime, needed: int = 1) -> float:
    """How long until the rolling-hour cap admits ``needed`` more actions.

    For the bulk tools' ``next_run_after``; 0 when they are admitted now.
    """
    resume_at = hourly_cap_resume_at(budget, now, needed=needed)
    if resume_at is None:
        return 0.0
    return max((resume_at - _utc(now)).total_seconds(), 0.0)
