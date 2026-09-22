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

import contextvars
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from linkedin_mcp_server.config.loaders import TRUTHY_VALUES, EnvironmentKeys
from linkedin_mcp_server.exceptions import ActionLimitError
from linkedin_mcp_server.limits import env_float, env_int, env_int_list

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

# Minimum spacing between two consecutive MCP tool calls, in seconds
# (TOOL_CALL_GAP_SECONDS), jittered by TOOL_CALL_GAP_JITTER. Zero turns the
# spacing off. Five seconds is the compromise between vendor spacing (a minute
# or more, for unattended campaigns) and an interactive MCP client where a
# minute of silence reads as a hung server: it holds a burst to about a dozen
# page loads a minute while staying inside what a person waits through.
DEFAULT_TOOL_CALL_GAP = 5.0
TOOL_CALL_GAP_JITTER = 0.2


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


def tool_call_gap_jitter() -> float:
    return _fraction(EnvironmentKeys.TOOL_CALL_GAP_JITTER, TOOL_CALL_GAP_JITTER)


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


def tool_call_gap(raw: str | None = None, rng: random.Random | None = None) -> float:
    """Seconds to leave between the end of one tool call and the next.

    ``raw`` is the configured gap in seconds as it arrives from the
    environment, unparsed; anything unusable falls back to the default rather
    than removing the spacing, since a typo must not be the way pacing is
    turned off. An explicit ``0`` is that way.
    """
    base = _configured_tool_call_gap(raw)
    if base <= 0:
        return 0.0
    spread = base * tool_call_gap_jitter()
    return step_delay((base - spread, base + spread), rng=rng)


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
        if not path.exists():
            raise FileNotFoundError(f"No job named {name!r} at {path}")
        return Job.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, job: Job) -> None:
        path = self._path(job.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a crash mid-write leaves the previous good file
        # rather than a truncated one.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)

    def list_jobs(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))


# LinkedIn counts activity per *account*, not per job -- it has no idea two
# scrapes belong to different "jobs" of ours. So the safety budget has to be
# one shared ledger that every LinkedIn-touching operation draws down,
# regardless of which queue it serves. This well-known job holds that single
# account-wide budget: its ledger, cap, warm-up and schedule. Its queue stays
# empty; per-work queues live in their own jobs.
ACCOUNT_BUDGET_JOB = "__account_budget__"


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
    budget = held if held is not None else load_account_budget(store, now)
    refuse_if_limited(budget, kind, now, ignore_schedule=schedule_ignored.get())
    budget.ledger.record_kind(kind, now)
    if held is None:
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
        budget = load_account_budget(store, now)
        budget.ledger.record_kind(kind, now)
        budget.ledger.prune(now)
        store.save(budget)
    except Exception:
        logger.warning(
            "Could not count the %s against the account budget", kind, exc_info=True
        )
