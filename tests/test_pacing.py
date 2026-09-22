"""Tests for the bulk-job pacing arithmetic.

Every function under test takes ``now`` explicitly, so nothing here sleeps.
"""

import fcntl
import json
import logging
import os
import random
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from linkedin_mcp_server.config.loaders import EnvironmentKeys
from linkedin_mcp_server.exceptions import ActionLimitError
from linkedin_mcp_server.limits import env_float
from linkedin_mcp_server.pacing import (
    ACCOUNT_BUDGET_JOB,
    ACCOUNT_COOLDOWN_FILE,
    COOLDOWN_STEPS_SECONDS,
    HOURLY_WINDOW_SECONDS,
    INVITES,
    Job,
    JobStore,
    LEDGER_LOCK_FILE,
    Ledger,
    MAX_BUNCH_PAUSE,
    MESSAGES,
    MIN_BUNCH_PAUSE,
    PROFILE,
    READ_TOOL_CALL_GAP,
    SEARCH,
    Schedule,
    WEEK_SECONDS,
    WINDOW_SECONDS,
    WRITE_TOOL_CALL_GAP,
    account_budget_in_use,
    charge_navigation,
    cooldown_resume_at,
    hourly_cap_resume_at,
    hourly_headroom,
    jittered_cap,
    kind_headroom,
    load_account_budget,
    max_daily_actions,
    navigation_kind,
    next_bunch_delay,
    note_throttle_signal,
    read_account_cooldown,
    record_action,
    record_throttle_signal,
    seconds_until_hourly_release,
    refuse_action,
    refuse_if_limited,
    schedule_ignored,
    step_delay,
    tool_call_gap,
    warmup_cap,
)

# 2026-08-05 is a Wednesday; 2026-08-08 a Saturday.
WED_10AM = datetime(2026, 8, 5, 10, 0)
WED_NOON = datetime(2026, 8, 5, 12, 30)
WED_3AM = datetime(2026, 8, 5, 3, 0)
WED_8PM = datetime(2026, 8, 5, 20, 0)
SAT_10AM = datetime(2026, 8, 8, 10, 0)


BH = Schedule.business_hours  # the opt-in 09-18 preset


class TestScheduleDefaultIsPermissive:
    def test_default_is_open_any_hour_any_day(self):
        # Working hours are opt-in: the bare default never blocks.
        assert Schedule().is_open(WED_3AM)
        assert Schedule().is_open(WED_NOON)
        assert Schedule().is_open(SAT_10AM)
        assert Schedule().is_open(WED_8PM)

    def test_default_next_open_is_always_now(self):
        assert Schedule().next_open(WED_3AM) == WED_3AM

    def test_default_seconds_until_close_is_until_midnight(self):
        # 22:00 -> next midnight is 2h.
        assert Schedule().seconds_until_close(WED_8PM) == 4 * 3600


class TestBusinessHours:
    def test_open_during_working_hours(self):
        assert BH().is_open(WED_10AM)

    def test_closed_overnight(self):
        assert not BH().is_open(WED_3AM)
        assert not BH().is_open(WED_8PM)

    def test_closed_at_lunch(self):
        assert not BH().is_open(WED_NOON)

    def test_closed_at_weekend(self):
        assert not BH().is_open(SAT_10AM)

    def test_next_open_returns_now_when_already_open(self):
        assert BH().next_open(WED_10AM) == WED_10AM

    def test_next_open_skips_to_morning(self):
        opens = BH().next_open(WED_3AM)
        assert opens.hour == 9
        assert opens.date() == WED_3AM.date()

    def test_next_open_skips_lunch(self):
        opens = BH().next_open(WED_NOON)
        assert opens.hour == 13

    def test_next_open_skips_the_weekend(self):
        opens = BH().next_open(SAT_10AM)
        assert opens.weekday() == 0  # Monday
        assert opens.hour == 9

    def test_next_open_after_close_lands_next_morning(self):
        opens = BH().next_open(WED_8PM)
        assert opens.day == WED_8PM.day + 1
        assert opens.hour == 9

    def test_seconds_until_close_excludes_lunch_still_ahead(self):
        # 10:00 -> 18:00 is 8h, minus the 1h lunch not yet taken.
        assert BH().seconds_until_close(WED_10AM) == 7 * 3600

    def test_seconds_until_close_keeps_afternoon_whole(self):
        # 14:00 is past lunch, so nothing is deducted.
        afternoon = datetime(2026, 8, 5, 14, 0)
        assert BH().seconds_until_close(afternoon) == 4 * 3600

    def test_seconds_until_close_is_zero_when_shut(self):
        assert BH().seconds_until_close(SAT_10AM) == 0.0

    def test_a_schedule_that_never_opens_is_rejected(self):
        every_day_off = Schedule(days_off=(0, 1, 2, 3, 4, 5, 6))
        with pytest.raises(ValueError, match="never opens"):
            every_day_off.next_open(WED_10AM)


class TestLedger:
    def test_actions_age_out_after_24h(self):
        ledger = Ledger()
        ledger.record(WED_10AM)
        # One second past the window.
        later = WED_10AM + timedelta(seconds=WINDOW_SECONDS + 1)
        assert ledger.spent(later) == 0

    def test_actions_inside_the_window_still_count(self):
        ledger = Ledger()
        ledger.record(WED_10AM)
        later = WED_10AM + timedelta(hours=23)
        assert ledger.spent(later) == 1

    def test_budget_refills_gradually_not_at_midnight(self):
        """The rolling window is the whole point.

        Ten actions spent over ten minutes do not all free up at 00:00, and
        they do not all free up at once either -- each returns on its own 24h
        anniversary. A midnight-reset budget would let this job spend twice in
        two minutes across the boundary, which is the exact burst shape that
        gets accounts flagged.
        """
        ledger = Ledger()
        for minute in range(10):
            ledger.record(WED_10AM + timedelta(minutes=minute))

        just_before_midnight = datetime(2026, 8, 5, 23, 59)
        assert ledger.remaining(just_before_midnight, cap=10) == 0

        # One minute past the first anniversary: only the two oldest are back.
        assert ledger.remaining(WED_10AM + timedelta(hours=24, minutes=1), 10) == 2

        # Ten minutes past, and the whole batch has aged out.
        assert ledger.remaining(WED_10AM + timedelta(hours=24, minutes=10), 10) == 10

    def test_next_expiry_is_when_the_oldest_ages_out(self):
        ledger = Ledger()
        ledger.record(WED_10AM)
        one_hour_in = WED_10AM + timedelta(hours=1)
        assert ledger.next_expiry(one_hour_in) == pytest.approx(23 * 3600, abs=1)

    def test_next_expiry_is_zero_when_empty(self):
        assert Ledger().next_expiry(WED_10AM) == 0.0


class TestWarmup:
    @pytest.mark.parametrize(
        ("day", "expected"),
        [(0, 10), (6, 10), (7, 20), (13, 20), (14, 50), (20, 50), (21, 100)],
    )
    def test_ramp_steps(self, day, expected):
        start = date(2026, 8, 5)
        assert warmup_cap(100, start, start + timedelta(days=day)) == expected

    def test_ramp_never_exceeds_the_configured_cap(self):
        start = date(2026, 8, 5)
        assert warmup_cap(5, start, start + timedelta(days=30)) == 5

    def test_clock_skew_backwards_is_treated_as_day_zero(self):
        start = date(2026, 8, 5)
        assert warmup_cap(100, start, start - timedelta(days=3)) == 10


class TestJitteredCap:
    def test_is_stable_within_a_day(self):
        day = date(2026, 8, 5)
        assert jittered_cap(100, day, "job") == jittered_cap(100, day, "job")

    def test_varies_across_days(self):
        caps = {
            jittered_cap(100, date(2026, 8, 5) + timedelta(days=d), "job")
            for d in range(14)
        }
        assert len(caps) > 1, "a constant daily total is the pattern to avoid"

    def test_stays_within_a_sane_band(self):
        for d in range(60):
            cap = jittered_cap(100, date(2026, 8, 5) + timedelta(days=d), "job")
            assert 85 <= cap <= 100

    def test_never_rounds_down_to_zero(self):
        assert jittered_cap(1, date(2026, 8, 5), "job") >= 1


class TestNextBunchDelay:
    def test_spreads_budget_across_the_remaining_window(self):
        # Business-hours preset: 7 usable hours, 100 left, bunches of 5 ->
        # 20 bunches -> ~21 min.
        delay = next_bunch_delay(100, 5, WED_10AM, BH())
        assert MIN_BUNCH_PAUSE <= delay <= MAX_BUNCH_PAUSE
        assert 15 * 60 <= delay <= 27 * 60

    def test_backs_right_off_when_budget_is_gone(self):
        assert next_bunch_delay(0, 5, WED_10AM, BH()) == MAX_BUNCH_PAUSE

    def test_backs_right_off_when_the_window_is_shut(self):
        assert next_bunch_delay(50, 5, SAT_10AM, BH()) == MAX_BUNCH_PAUSE

    def test_a_tiny_remaining_budget_still_respects_the_floor(self):
        # One bunch left across 7 hours would otherwise suggest a 7-hour wait;
        # the clamp keeps it bounded.
        delay = next_bunch_delay(1, 5, WED_10AM, BH())
        assert delay <= MAX_BUNCH_PAUSE


class TestJobRoundTrip:
    def test_survives_serialization(self):
        job = Job(
            name="egypt-gulf",
            started_on=date(2026, 8, 5),
            pending=["a", "b"],
            done={"c": {"url": "x"}},
            failed={"d": "boom"},
            ledger=Ledger(actions=[1.0, 2.0]),
            daily_cap=80,
            schedule=Schedule(work_start=8, work_end=17, days_off=(6,)),
            warmup=False,
        )
        restored = Job.from_dict(job.to_dict())
        assert restored.to_dict() == job.to_dict()
        assert restored.schedule.work_start == 8
        assert restored.schedule.days_off == (6,)
        assert restored.warmup is False

    def test_a_job_file_without_strikes_loads_with_none(self):
        # Job files written before strikes existed carry no such key.
        job = Job.from_dict({"name": "j", "started_on": "2020-01-01"})
        assert job.strikes == {}

    def test_effective_cap_applies_warmup_then_jitter(self):
        job = Job(name="j", started_on=date(2026, 8, 5), daily_cap=100)
        # Day 0 of the ramp caps at 10, jitter can only shave it.
        assert 1 <= job.effective_cap(WED_10AM) <= 10

    def test_effective_cap_is_bounded_by_the_global_ceiling(self):
        job = Job(name="j", started_on=date(2020, 1, 1), daily_cap=150, warmup=False)
        assert job.effective_cap(WED_10AM) <= 150


class TestAccountBudget:
    def test_first_call_materialises_a_default_persisted_budget(self, tmp_path):
        store = JobStore(tmp_path)
        budget = load_account_budget(store, WED_10AM)
        assert budget.name == ACCOUNT_BUDGET_JOB
        assert store.exists(ACCOUNT_BUDGET_JOB)  # persisted, not ephemeral
        assert budget.warmup is False  # a shared budget is not a fresh account

    def test_reconfigures_cap_and_persists(self, tmp_path):
        store = JobStore(tmp_path)
        load_account_budget(store, WED_10AM, daily_cap=100)
        load_account_budget(store, WED_10AM, daily_cap=40)
        assert store.load(ACCOUNT_BUDGET_JOB).daily_cap == 40

    def test_pure_read_does_not_change_config(self, tmp_path):
        store = JobStore(tmp_path)
        load_account_budget(store, WED_10AM, daily_cap=40, warmup=True)
        again = load_account_budget(store, WED_10AM)  # all-None -> read
        assert again.daily_cap == 40
        assert again.warmup is True

    def test_the_ledger_is_shared_across_loads(self, tmp_path):
        """Two callers (person + company enrichment) draw down ONE ledger."""
        store = JobStore(tmp_path)
        a = load_account_budget(store, WED_10AM)
        a.ledger.record(WED_10AM)
        store.save(a)
        b = load_account_budget(store, WED_10AM)
        assert b.ledger.spent(WED_10AM) == 1


class TestJobStore:
    def test_save_and_load(self, tmp_path):
        store = JobStore(tmp_path / "jobs")
        job = Job(name="egypt-gulf", started_on=date(2026, 8, 5), pending=["a"])
        store.save(job)
        assert store.exists("egypt-gulf")
        assert store.load("egypt-gulf").pending == ["a"]

    def test_missing_job_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            JobStore(tmp_path / "jobs").load("nope")

    def test_list_is_empty_before_anything_is_written(self, tmp_path):
        assert JobStore(tmp_path / "unborn").list_jobs() == []

    def test_unsafe_names_are_rejected_not_lossily_mapped(self, tmp_path):
        # Path traversal and separators are refused outright, so nothing is
        # written outside the store and no two names alias to one file.
        store = JobStore(tmp_path / "jobs")
        for bad in ("../../etc/passwd", "a/b", "egypt gulf", "../.."):
            with pytest.raises(ValueError, match="may contain only"):
                store.save(Job(name=bad, started_on=date(2026, 8, 5)))
        assert not (tmp_path / "etc").exists()

    def test_distinct_names_do_not_collide(self, tmp_path):
        # "a-b" and "ab" are different files (the old strip-based path mapped
        # "a/b" and "ab" to the same one).
        store = JobStore(tmp_path / "jobs")
        store.save(Job(name="a-b", started_on=date(2026, 8, 5), pending=["x"]))
        store.save(Job(name="ab", started_on=date(2026, 8, 5), pending=["y"]))
        assert store.load("a-b").pending == ["x"]
        assert store.load("ab").pending == ["y"]

    def test_no_tmp_file_is_left_behind(self, tmp_path):
        store = JobStore(tmp_path / "jobs")
        store.save(Job(name="j", started_on=date(2026, 8, 5)))
        assert list((tmp_path / "jobs").glob("*.tmp")) == []


class TestToolCallGap:
    """The gap the middleware leaves between two MCP tool calls."""

    def test_read_gaps_fill_the_documented_band(self):
        gaps = [tool_call_gap() for _ in range(200)]

        low, high = READ_TOOL_CALL_GAP
        assert (low, high) == (8.0, 20.0)
        assert all(low <= gap <= high for gap in gaps)
        # Randomised, not a constant: a fixed period is the pattern being
        # avoided. Both halves of the band are used, not one edge of it.
        assert min(gaps) < (low + high) / 2 < max(gaps)

    def test_write_gaps_fill_the_longer_band(self):
        gaps = [tool_call_gap(write=True) for _ in range(200)]

        low, high = WRITE_TOOL_CALL_GAP
        assert (low, high) == (20.0, 60.0)
        assert all(low <= gap <= high for gap in gaps)
        assert min(gaps) < (low + high) / 2 < max(gaps)

    def test_a_write_waits_longer_than_a_read(self):
        reads = [tool_call_gap() for _ in range(200)]
        writes = [tool_call_gap(write=True) for _ in range(200)]

        assert min(writes) >= max(reads)
        assert sum(writes) / len(writes) > sum(reads) / len(reads)

    def test_the_draw_is_log_uniform_not_uniform(self):
        """Seeded, so the shape is asserted rather than sampled.

        Under a log-uniform draw the geometric midpoint of the band is the
        median; under a uniform one the arithmetic midpoint is. The two sit
        far enough apart on an 8-20 band (12.6 vs 14) to tell them apart.
        """
        rng = random.Random(57)
        gaps = sorted(tool_call_gap(rng=rng) for _ in range(2000))

        low, high = READ_TOOL_CALL_GAP
        median = gaps[len(gaps) // 2]
        assert abs(median - (low * high) ** 0.5) < abs(median - (low + high) / 2)

    def test_a_configured_gap_sets_the_read_minimum_and_scales_the_rest(self):
        reads = [tool_call_gap("16") for _ in range(200)]
        writes = [tool_call_gap("16", write=True) for _ in range(200)]

        # Twice the default minimum doubles every bound.
        assert all(16.0 <= gap <= 40.0 for gap in reads)
        assert all(40.0 <= gap <= 120.0 for gap in writes)
        assert min(reads) < 20.0
        assert min(writes) < 50.0

    def test_zero_is_the_way_to_turn_the_spacing_off(self):
        assert tool_call_gap("0") == 0.0

    @pytest.mark.parametrize("raw", ["", "  ", None, "soon", "-5"])
    def test_anything_unusable_falls_back_to_the_default(self, raw):
        # A typo must not be a second way of disabling the pacing, which is why
        # only an explicit 0 does that.
        assert tool_call_gap(raw) > 0


class TestConfigurableLimits:
    """Every pacing constant is a default an environment variable replaces.

    Each variable gets one test that moves the behaviour and one that shows a
    garbage value falling back to the default with a warning, since a typo
    must not be the way pacing is turned off.
    """

    START = date(2020, 1, 1)  # long past any warm-up ramp

    def _warned(self, caplog, key):
        return any(key in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf"])
    def test_a_non_finite_float_falls_back_with_a_warning(
        self, monkeypatch, caplog, raw
    ):
        """``float("nan")`` parses and compares below nothing, so it would
        sail past the minimum check; ``inf`` would make a pause never end."""
        monkeypatch.setenv(EnvironmentKeys.NAV_DELAY_SECONDS, raw)
        with caplog.at_level(logging.WARNING):
            assert env_float(EnvironmentKeys.NAV_DELAY_SECONDS, 2.0) == 2.0
        assert self._warned(caplog, EnvironmentKeys.NAV_DELAY_SECONDS)

    # DAILY_ACTIONS_MAX

    def test_daily_max_lowers_the_ceiling(self, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.DAILY_ACTIONS_MAX, "120")
        monkeypatch.setenv(EnvironmentKeys.DAILY_CAP_JITTER, "0")
        job = Job(name="j", started_on=self.START, daily_cap=200, warmup=True)
        assert job.effective_cap(WED_10AM) == 120

    def test_daily_max_cannot_raise_the_ceiling(self, monkeypatch, caplog):
        """150 is the ceiling, and the environment used to lift it: the live
        ledger this was written against read ``daily_cap: 250``."""
        monkeypatch.setenv(EnvironmentKeys.DAILY_ACTIONS_MAX, "200")
        monkeypatch.setenv(EnvironmentKeys.DAILY_CAP_JITTER, "0")
        job = Job(name="j", started_on=self.START, daily_cap=200, warmup=True)
        with caplog.at_level(logging.WARNING):
            assert job.effective_cap(WED_10AM) == 150
        assert self._warned(caplog, EnvironmentKeys.DAILY_ACTIONS_MAX)

    def test_the_clamp_warns_once_per_value_not_once_per_read(
        self, monkeypatch, caplog
    ):
        """The ceiling is read on every page load; measured with
        DAILY_ACTIONS_MAX=250, one warning per navigation."""
        monkeypatch.setenv(EnvironmentKeys.DAILY_ACTIONS_MAX, "250")
        with caplog.at_level(logging.WARNING):
            assert max_daily_actions() == 150
            assert max_daily_actions() == 150
            monkeypatch.setenv(EnvironmentKeys.DAILY_ACTIONS_MAX, "300")
            assert max_daily_actions() == 150
        clamps = [r for r in caplog.records if "Clamping" in r.getMessage()]
        assert [r.getMessage() for r in clamps] == [
            "Clamping DAILY_ACTIONS_MAX=250 to the ceiling 150",
            "Clamping DAILY_ACTIONS_MAX=300 to the ceiling 150",
        ]

    def test_a_persisted_cap_above_the_ceiling_is_clamped_on_read(self, tmp_path):
        store = JobStore(tmp_path)
        path = store.root / f"{ACCOUNT_BUDGET_JOB}.json"
        path.write_text(
            '{"name": "__account_budget__", "started_on": "2026-08-01", '
            '"daily_cap": 250}',
            encoding="utf-8",
        )
        assert store.load(ACCOUNT_BUDGET_JOB).daily_cap == 150

    def test_daily_max_garbage_falls_back_with_a_warning(self, monkeypatch, caplog):
        monkeypatch.setenv(EnvironmentKeys.DAILY_ACTIONS_MAX, "lots")
        job = Job(name="j", started_on=self.START, daily_cap=200, warmup=False)
        with caplog.at_level(logging.WARNING):
            assert job.effective_cap(WED_10AM) <= 150
        assert self._warned(caplog, EnvironmentKeys.DAILY_ACTIONS_MAX)

    # DAILY_ACTIONS_DEFAULT

    def test_daily_default_is_what_an_unconfigured_job_gets(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(EnvironmentKeys.DAILY_ACTIONS_DEFAULT, "40")
        assert Job(name="j", started_on=self.START).daily_cap == 40
        assert Job.from_dict({"name": "j", "started_on": "2020-01-01"}).daily_cap == 40
        budget = load_account_budget(JobStore(tmp_path), WED_10AM)
        assert budget.daily_cap == 40

    def test_daily_default_garbage_falls_back_with_a_warning(self, monkeypatch, caplog):
        monkeypatch.setenv(EnvironmentKeys.DAILY_ACTIONS_DEFAULT, "0")
        with caplog.at_level(logging.WARNING):
            assert Job(name="j", started_on=self.START).daily_cap == 100
        assert self._warned(caplog, EnvironmentKeys.DAILY_ACTIONS_DEFAULT)

    # DAILY_CAP_JITTER

    def test_cap_jitter_zero_leaves_the_cap_whole(self, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.DAILY_CAP_JITTER, "0")
        assert all(
            jittered_cap(100, date(2026, 8, 5) + timedelta(days=d), "job") == 100
            for d in range(30)
        )

    def test_cap_jitter_widens_the_band(self, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.DAILY_CAP_JITTER, "0.5")
        caps = [
            jittered_cap(100, date(2026, 8, 5) + timedelta(days=d), "job")
            for d in range(60)
        ]
        assert min(caps) < 85
        assert all(50 <= cap <= 100 for cap in caps)

    def test_cap_jitter_above_one_is_clamped(self, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.DAILY_CAP_JITTER, "5")
        for d in range(60):
            assert 1 <= jittered_cap(100, date(2026, 8, 5) + timedelta(days=d)) <= 100

    def test_cap_jitter_garbage_falls_back_with_a_warning(self, monkeypatch, caplog):
        monkeypatch.setenv(EnvironmentKeys.DAILY_CAP_JITTER, "some")
        with caplog.at_level(logging.WARNING):
            for d in range(60):
                cap = jittered_cap(100, date(2026, 8, 5) + timedelta(days=d), "job")
                assert 85 <= cap <= 100
        assert self._warned(caplog, EnvironmentKeys.DAILY_CAP_JITTER)

    # WARMUP_CAPS / WARMUP_DAYS

    def test_warmup_ramp_follows_the_configured_steps(self, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.WARMUP_CAPS, "5,15")
        monkeypatch.setenv(EnvironmentKeys.WARMUP_DAYS, "3,6")
        start = date(2026, 8, 5)
        assert warmup_cap(100, start, start) == 5
        assert warmup_cap(100, start, start + timedelta(days=3)) == 15
        assert warmup_cap(100, start, start + timedelta(days=6)) == 100

    def test_warmup_ramp_of_mismatched_lengths_falls_back_with_a_warning(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv(EnvironmentKeys.WARMUP_CAPS, "5,15,30")
        monkeypatch.setenv(EnvironmentKeys.WARMUP_DAYS, "3,6")
        start = date(2026, 8, 5)
        with caplog.at_level(logging.WARNING):
            assert warmup_cap(100, start, start) == 10
        assert self._warned(caplog, EnvironmentKeys.WARMUP_CAPS)

    def test_warmup_ramp_out_of_order_days_falls_back_with_a_warning(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv(EnvironmentKeys.WARMUP_CAPS, "10,20,50")
        monkeypatch.setenv(EnvironmentKeys.WARMUP_DAYS, "21,14,7")
        start = date(2026, 8, 5)
        with caplog.at_level(logging.WARNING):
            assert warmup_cap(100, start, start) == 10
        assert self._warned(caplog, EnvironmentKeys.WARMUP_DAYS)

    def test_warmup_ramp_garbage_falls_back_with_a_warning(self, monkeypatch, caplog):
        monkeypatch.setenv(EnvironmentKeys.WARMUP_CAPS, "ten,twenty,fifty")
        start = date(2026, 8, 5)
        with caplog.at_level(logging.WARNING):
            assert warmup_cap(100, start, start + timedelta(days=7)) == 20
        assert self._warned(caplog, EnvironmentKeys.WARMUP_CAPS)

    # STEP_DELAY_MIN_SECONDS / STEP_DELAY_MAX_SECONDS

    def test_step_delay_range_is_the_configured_one(self, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.STEP_DELAY_MIN_SECONDS, "1")
        monkeypatch.setenv(EnvironmentKeys.STEP_DELAY_MAX_SECONDS, "2")
        assert all(1.0 <= step_delay() <= 2.0 for _ in range(100))

    def test_step_delay_inverted_range_falls_back_with_a_warning(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv(EnvironmentKeys.STEP_DELAY_MIN_SECONDS, "30")
        monkeypatch.setenv(EnvironmentKeys.STEP_DELAY_MAX_SECONDS, "2")
        with caplog.at_level(logging.WARNING):
            assert all(8.0 <= step_delay() <= 25.0 for _ in range(100))
        assert self._warned(caplog, EnvironmentKeys.STEP_DELAY_MIN_SECONDS)

    def test_step_delay_garbage_falls_back_with_a_warning(self, monkeypatch, caplog):
        monkeypatch.setenv(EnvironmentKeys.STEP_DELAY_MAX_SECONDS, "soon")
        with caplog.at_level(logging.WARNING):
            assert all(8.0 <= step_delay() <= 25.0 for _ in range(100))
        assert self._warned(caplog, EnvironmentKeys.STEP_DELAY_MAX_SECONDS)

    # BUNCH_PAUSE_MIN_SECONDS / BUNCH_PAUSE_MAX_SECONDS

    def test_bunch_pause_range_is_the_configured_one(self, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.BUNCH_PAUSE_MIN_SECONDS, "10")
        monkeypatch.setenv(EnvironmentKeys.BUNCH_PAUSE_MAX_SECONDS, "20")
        assert next_bunch_delay(0, 5, WED_10AM, BH()) == 20.0
        assert 10.0 <= next_bunch_delay(100, 5, WED_10AM, BH()) <= 20.0

    def test_bunch_pause_inverted_range_falls_back_with_a_warning(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv(EnvironmentKeys.BUNCH_PAUSE_MIN_SECONDS, "500")
        monkeypatch.setenv(EnvironmentKeys.BUNCH_PAUSE_MAX_SECONDS, "20")
        with caplog.at_level(logging.WARNING):
            assert next_bunch_delay(0, 5, WED_10AM, BH()) == MAX_BUNCH_PAUSE
        assert self._warned(caplog, EnvironmentKeys.BUNCH_PAUSE_MIN_SECONDS)

    def test_bunch_pause_garbage_falls_back_with_a_warning(self, monkeypatch, caplog):
        monkeypatch.setenv(EnvironmentKeys.BUNCH_PAUSE_MAX_SECONDS, "-1")
        with caplog.at_level(logging.WARNING):
            assert next_bunch_delay(0, 5, WED_10AM, BH()) == MAX_BUNCH_PAUSE
        assert self._warned(caplog, EnvironmentKeys.BUNCH_PAUSE_MAX_SECONDS)

    # BUNCH_PAUSE_JITTER

    def test_bunch_pause_jitter_zero_makes_the_spacing_exact(self, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.BUNCH_PAUSE_JITTER, "0")
        # 7 usable hours, 100 left, bunches of 5 -> 20 bunches -> 1260s flat.
        assert next_bunch_delay(100, 5, WED_10AM, BH()) == 7 * 3600 / 20

    def test_bunch_pause_jitter_garbage_falls_back_with_a_warning(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv(EnvironmentKeys.BUNCH_PAUSE_JITTER, "wide")
        with caplog.at_level(logging.WARNING):
            delays = {next_bunch_delay(100, 5, WED_10AM, BH()) for _ in range(50)}
        assert all(0.75 * 1260 <= d <= 1.25 * 1260 for d in delays)
        assert len(delays) > 1
        assert self._warned(caplog, EnvironmentKeys.BUNCH_PAUSE_JITTER)

    # HOURLY_ACTIONS_MAX

    def test_hourly_max_moves_the_cap(self, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.HOURLY_ACTIONS_MAX, "2")
        budget = Job(name=ACCOUNT_BUDGET_JOB, started_on=self.START)
        budget.ledger.record(WED_10AM)
        assert hourly_cap_resume_at(budget, WED_10AM) is None
        budget.ledger.record(WED_10AM)
        assert hourly_cap_resume_at(budget, WED_10AM) is not None

    def test_hourly_max_garbage_falls_back_with_a_warning(self, monkeypatch, caplog):
        monkeypatch.setenv(EnvironmentKeys.HOURLY_ACTIONS_MAX, "0")
        budget = Job(name=ACCOUNT_BUDGET_JOB, started_on=self.START)
        for _ in range(39):
            budget.ledger.record(WED_10AM)
        with caplog.at_level(logging.WARNING):
            assert hourly_cap_resume_at(budget, WED_10AM) is None
        assert self._warned(caplog, EnvironmentKeys.HOURLY_ACTIONS_MAX)


class TestAccountBudgetSchedule:
    """The account budget defaults to business hours; a stored one is kept."""

    def test_a_fresh_account_budget_is_on_business_hours(self, tmp_path):
        budget = load_account_budget(JobStore(tmp_path), WED_10AM)
        assert budget.schedule == BH()
        # And it is what the file says, not only what this call returned.
        assert load_account_budget(JobStore(tmp_path), WED_10AM).schedule == BH()

    def test_an_existing_ledger_keeps_its_stored_schedule(self, tmp_path):
        store = JobStore(tmp_path)
        load_account_budget(store, WED_10AM, schedule=Schedule())
        assert load_account_budget(store, WED_10AM).schedule == Schedule()

    def test_a_per_work_job_still_defaults_to_always_open(self):
        assert Job(name="j", started_on=date(2020, 1, 1)).schedule == Schedule()


class TestNavigationKind:
    @pytest.mark.parametrize(
        ("url", "kind"),
        [
            ("https://www.linkedin.com/in/testuser/", "profile"),
            ("https://www.linkedin.com/in/testuser/details/experience/", "profile"),
            ("https://www.linkedin.com/in/me/", "profile"),
            ("https://www.linkedin.com/company/acme/people/", "company"),
            ("https://www.linkedin.com/search/results/people/?keywords=x", "search"),
            ("https://www.linkedin.com/messaging/compose/?x=1", "messaging"),
            ("https://www.linkedin.com/feed/", "feed"),
            ("https://www.linkedin.com/jobs/view/123/", "other"),
            ("https://www.linkedin.com/preload/custom-invite/?vanityName=u", "other"),
        ],
    )
    def test_files_a_url_under_its_kind(self, url, kind):
        assert navigation_kind(url) == kind


class TestLedgerKinds:
    def test_kinds_are_counted_apart_from_actions(self):
        ledger = Ledger()
        ledger.record_kind(PROFILE, WED_10AM)
        ledger.record_kind(SEARCH, WED_10AM)
        assert ledger.spent_kind(PROFILE, WED_10AM) == 1
        assert ledger.spent_kind(SEARCH, WED_10AM) == 1
        assert ledger.spent(WED_10AM) == 0

    def test_a_kind_ages_out_of_its_window(self):
        ledger = Ledger()
        ledger.record_kind(INVITES, WED_10AM)
        a_day_on = WED_10AM + timedelta(seconds=WINDOW_SECONDS + 1)
        assert ledger.spent_kind(INVITES, a_day_on) == 0
        assert ledger.spent_kind(INVITES, a_day_on, WEEK_SECONDS) == 1
        assert (
            ledger.spent_kind(INVITES, WED_10AM + timedelta(days=8), WEEK_SECONDS) == 0
        )

    def test_kinds_survive_a_round_trip(self, tmp_path):
        store = JobStore(tmp_path)
        budget = load_account_budget(store, WED_10AM)
        budget.ledger.record_kind(PROFILE, WED_10AM)
        store.save(budget)
        assert (
            load_account_budget(store, WED_10AM).ledger.spent_kind(PROFILE, WED_10AM)
            == 1
        )

    def test_pruning_keeps_a_week_of_kinds(self):
        ledger = Ledger()
        ledger.record_kind(INVITES, WED_10AM)
        ledger.record(WED_10AM)
        ledger.prune(WED_10AM + timedelta(days=2))
        assert ledger.kinds[INVITES] == [WED_10AM.timestamp()]
        assert ledger.actions == []
        ledger.prune(WED_10AM + timedelta(days=8))
        assert ledger.kinds[INVITES] == []


def _budget(schedule: Schedule | None = None) -> Job:
    return Job(
        name=ACCOUNT_BUDGET_JOB,
        started_on=date(2020, 1, 1),
        schedule=schedule or Schedule(),
    )


def _spend(budget: Job, kind: str, count: int, at: datetime) -> None:
    for _ in range(count):
        budget.ledger.record_kind(kind, at)


class TestPerKindCaps:
    """Each kind refuses at N+1 and says when the oldest unit frees up."""

    @pytest.mark.parametrize(
        ("kind", "cap"), [(PROFILE, 80), (SEARCH, 60), (INVITES, 20), (MESSAGES, 50)]
    )
    def test_the_daily_cap_refuses_one_past_it(self, kind, cap):
        budget = _budget()
        _spend(budget, kind, cap - 1, WED_10AM)
        refuse_if_limited(budget, kind, WED_10AM)  # the Nth is allowed

        budget.ledger.record_kind(kind, WED_10AM)
        with pytest.raises(ActionLimitError) as excinfo:
            refuse_if_limited(budget, kind, WED_10AM + timedelta(hours=1))

        error = excinfo.value
        assert error.error_type == "limit_exceeded"
        assert error.kind == kind
        assert error.limit == cap
        assert error.window == "24 h"
        assert error.resume_at == WED_10AM + timedelta(hours=24)
        assert "limit_exceeded" in str(error)

    def test_invites_are_also_capped_per_week(self):
        budget = _budget()
        # 20 a day on each of the last five days: none today, 100 this week.
        for day in range(5):
            _spend(budget, INVITES, 20, WED_10AM - timedelta(days=day + 1))
        now = WED_10AM + timedelta(hours=1)
        assert budget.ledger.spent_kind(INVITES, now) == 0

        with pytest.raises(ActionLimitError) as excinfo:
            refuse_if_limited(budget, INVITES, now)

        assert excinfo.value.limit == 100
        assert excinfo.value.window == "7 d"
        # The oldest twenty were sent five days ago and free up two days on.
        assert excinfo.value.resume_at == WED_10AM - timedelta(days=5) + timedelta(
            days=7
        )

    def test_uncapped_kinds_are_counted_but_never_refused(self):
        budget = _budget()
        _spend(budget, "company", 500, WED_10AM)
        refuse_if_limited(budget, "company", WED_10AM)

    @pytest.mark.parametrize(
        ("key", "kind"),
        [
            (EnvironmentKeys.PROFILE_LOADS_MAX, PROFILE),
            (EnvironmentKeys.SEARCH_PAGES_MAX, SEARCH),
            (EnvironmentKeys.INVITES_MAX, INVITES),
            (EnvironmentKeys.MESSAGES_MAX, MESSAGES),
        ],
    )
    def test_the_environment_lowers_a_cap(self, monkeypatch, key, kind):
        monkeypatch.setenv(key, "3")
        budget = _budget()
        _spend(budget, kind, 3, WED_10AM)
        with pytest.raises(ActionLimitError) as excinfo:
            refuse_if_limited(budget, kind, WED_10AM)
        assert excinfo.value.limit == 3

    @pytest.mark.parametrize(
        ("key", "kind", "default"),
        [
            (EnvironmentKeys.PROFILE_LOADS_MAX, PROFILE, 80),
            (EnvironmentKeys.SEARCH_PAGES_MAX, SEARCH, 60),
            (EnvironmentKeys.INVITES_MAX, INVITES, 20),
            (EnvironmentKeys.MESSAGES_MAX, MESSAGES, 50),
        ],
    )
    def test_the_environment_cannot_raise_a_cap(
        self, monkeypatch, caplog, key, kind, default
    ):
        monkeypatch.setenv(key, str(default * 2))
        budget = _budget()
        _spend(budget, kind, default, WED_10AM)
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(ActionLimitError) as excinfo,
        ):
            refuse_if_limited(budget, kind, WED_10AM)
        assert excinfo.value.limit == default
        assert any(key in r.getMessage() for r in caplog.records)

    def test_the_weekly_invite_cap_lowers_but_never_raises(self, monkeypatch, caplog):
        monkeypatch.setenv(EnvironmentKeys.INVITES_WEEKLY_MAX, "30")
        budget = _budget()
        _spend(budget, INVITES, 30, WED_10AM - timedelta(days=2))
        with pytest.raises(ActionLimitError) as excinfo:
            refuse_if_limited(budget, INVITES, WED_10AM)
        assert (excinfo.value.limit, excinfo.value.window) == (30, "7 d")

        monkeypatch.setenv(EnvironmentKeys.INVITES_WEEKLY_MAX, "300")
        _spend(budget, INVITES, 70, WED_10AM - timedelta(days=2))
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(ActionLimitError) as excinfo,
        ):
            refuse_if_limited(budget, INVITES, WED_10AM)
        assert excinfo.value.limit == 100
        assert any(
            EnvironmentKeys.INVITES_WEEKLY_MAX in r.getMessage() for r in caplog.records
        )


class TestWorkingHours:
    """Writes wait for the schedule; reads run on half their cap outside it."""

    @pytest.mark.parametrize("kind", [INVITES, MESSAGES])
    def test_a_write_outside_the_schedule_is_refused_until_it_reopens(self, kind):
        with pytest.raises(ActionLimitError) as excinfo:
            refuse_if_limited(_budget(BH()), kind, WED_8PM)
        error = excinfo.value
        assert error.error_type == "limit_exceeded"
        assert error.limit == 0
        assert error.window == "working hours"
        assert error.resume_at == datetime(2026, 8, 6, 9, 0)
        assert "working hours" in str(error)

    def test_resume_at_skips_lunch(self):
        with pytest.raises(ActionLimitError) as excinfo:
            refuse_if_limited(_budget(BH()), INVITES, WED_NOON)
        assert excinfo.value.resume_at == datetime(2026, 8, 5, 13, 0)

    def test_resume_at_skips_the_weekend(self):
        with pytest.raises(ActionLimitError) as excinfo:
            refuse_if_limited(_budget(BH()), MESSAGES, SAT_10AM)
        assert excinfo.value.resume_at == datetime(2026, 8, 10, 9, 0)  # Monday

    def test_resume_at_keeps_the_schedule_timezone(self):
        """The schedule is local time, so the answer carries the zone `now`
        came in, and the ISO text in the message names it."""
        local = WED_8PM.astimezone()
        with pytest.raises(ActionLimitError) as excinfo:
            refuse_if_limited(_budget(BH()), INVITES, local)
        assert excinfo.value.resume_at.tzinfo == local.tzinfo
        assert excinfo.value.resume_at.isoformat(timespec="minutes") in str(
            excinfo.value
        )

    def test_a_write_inside_the_schedule_passes(self):
        refuse_if_limited(_budget(BH()), INVITES, WED_10AM)

    @pytest.mark.parametrize(("kind", "half"), [(PROFILE, 40), (SEARCH, 30)])
    def test_reads_outside_the_schedule_run_on_half_the_cap(self, kind, half):
        budget = _budget(BH())
        _spend(budget, kind, half - 1, WED_8PM)
        refuse_if_limited(budget, kind, WED_8PM)

        budget.ledger.record_kind(kind, WED_8PM)
        with pytest.raises(ActionLimitError) as excinfo:
            refuse_if_limited(budget, kind, WED_8PM)
        assert excinfo.value.limit == half
        # The full cap is back at 09:00, well before the oldest unit ages out.
        assert excinfo.value.resume_at == datetime(2026, 8, 6, 9, 0)

        # And the same ledger is fine once the window opens.
        refuse_if_limited(budget, kind, datetime(2026, 8, 6, 9, 0))

    def test_the_permissive_default_schedule_never_refuses(self):
        budget = _budget()
        refuse_if_limited(budget, INVITES, SAT_10AM)
        _spend(budget, PROFILE, 79, WED_3AM)
        refuse_if_limited(budget, PROFILE, WED_3AM)

    @pytest.mark.parametrize("raw", ["1", "true", "YES"])
    def test_the_opt_out_lifts_the_gate_and_the_halving(self, monkeypatch, raw):
        monkeypatch.setenv(EnvironmentKeys.WORKING_HOURS_DISABLED, raw)
        budget = _budget(BH())
        refuse_if_limited(budget, INVITES, WED_8PM)
        _spend(budget, PROFILE, 79, WED_8PM)
        refuse_if_limited(budget, PROFILE, WED_8PM)

    def test_ignore_schedule_lifts_the_halving_for_one_caller(self):
        budget = _budget(BH())
        _spend(budget, PROFILE, 45, WED_8PM)
        with pytest.raises(ActionLimitError):
            refuse_if_limited(budget, PROFILE, WED_8PM)
        refuse_if_limited(budget, PROFILE, WED_8PM, ignore_schedule=True)

    def test_a_navigation_under_schedule_ignored_runs_on_the_full_cap(self, tmp_path):
        store = JobStore(tmp_path)
        held = load_account_budget(store, WED_8PM, schedule=BH())
        _spend(held, PROFILE, 45, WED_8PM)
        token = account_budget_in_use.set(held)
        ignored = schedule_ignored.set(True)
        try:
            charge_navigation(store, "https://www.linkedin.com/in/u/", WED_8PM)
        finally:
            schedule_ignored.reset(ignored)
            account_budget_in_use.reset(token)
        assert held.ledger.spent_kind(PROFILE, WED_8PM) == 46

    def test_the_opt_out_needs_a_truthy_value(self, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.WORKING_HOURS_DISABLED, "0")
        with pytest.raises(ActionLimitError):
            refuse_if_limited(_budget(BH()), INVITES, WED_8PM)


class TestKindHeadroom:
    """What a bulk tool plans against before it starts."""

    def test_headroom_is_the_tightest_cap_less_what_is_spent(self):
        budget = _budget()
        _spend(budget, PROFILE, 30, WED_10AM - timedelta(hours=2))
        headroom, wait = kind_headroom(budget, PROFILE, WED_10AM)
        assert headroom == 50
        assert wait == pytest.approx(22 * 3600, abs=1)

    def test_headroom_never_goes_negative(self):
        budget = _budget()
        _spend(budget, PROFILE, 90, WED_10AM)
        assert kind_headroom(budget, PROFILE, WED_10AM)[0] == 0

    def test_the_weekly_invite_cap_can_be_the_tight_one(self):
        budget = _budget()
        _spend(budget, INVITES, 95, WED_10AM - timedelta(days=3))
        headroom, wait = kind_headroom(budget, INVITES, WED_10AM)
        assert headroom == 5
        assert wait == pytest.approx(4 * 24 * 3600, abs=1)

    def test_off_hours_halves_the_headroom_and_waits_for_the_window(self):
        budget = _budget(BH())
        _spend(budget, PROFILE, 30, WED_8PM)
        headroom, wait = kind_headroom(budget, PROFILE, WED_8PM)
        assert headroom == 10
        assert wait == 13 * 3600  # 09:00 next morning, not 24 h on
        assert kind_headroom(budget, PROFILE, WED_8PM, ignore_schedule=True) == (
            50,
            pytest.approx(24 * 3600, abs=1),
        )

    def test_an_uncapped_kind_reports_the_daily_ceiling(self):
        assert kind_headroom(_budget(), "company", WED_10AM) == (150, 0.0)


class TestChargeNavigation:
    def test_a_page_load_costs_one_action_and_one_of_its_kind(self, tmp_path):
        store = JobStore(tmp_path)
        charge_navigation(store, "https://www.linkedin.com/in/u/", WED_10AM)
        charge_navigation(
            store, "https://www.linkedin.com/search/results/all/", WED_10AM
        )

        budget = load_account_budget(store, WED_10AM)
        assert budget.ledger.spent(WED_10AM) == 2
        assert budget.ledger.spent_kind(PROFILE, WED_10AM) == 1
        assert budget.ledger.spent_kind(SEARCH, WED_10AM) == 1

    def test_a_refused_load_is_not_charged(self, tmp_path, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.PROFILE_LOADS_MAX, "1")
        store = JobStore(tmp_path)
        charge_navigation(store, "https://www.linkedin.com/in/u/", WED_10AM)
        with pytest.raises(ActionLimitError):
            charge_navigation(store, "https://www.linkedin.com/in/v/", WED_10AM)

        budget = load_account_budget(store, WED_10AM)
        assert budget.ledger.spent(WED_10AM) == 1
        assert budget.ledger.spent_kind(PROFILE, WED_10AM) == 1

    def test_a_held_budget_takes_the_kind_and_keeps_its_own_actions(self, tmp_path):
        """A bulk tool records `actions` itself, one per page load, and saves
        its copy; the navigation adds the kind to that copy and nothing to
        disk, or the tool's next save would throw the kind away."""
        store = JobStore(tmp_path)
        held = load_account_budget(store, WED_10AM)
        token = account_budget_in_use.set(held)
        try:
            charge_navigation(store, "https://www.linkedin.com/in/u/", WED_10AM)
        finally:
            account_budget_in_use.reset(token)

        assert held.ledger.spent_kind(PROFILE, WED_10AM) == 1
        assert held.ledger.spent(WED_10AM) == 0
        on_disk = load_account_budget(store, WED_10AM)
        assert on_disk.ledger.spent(WED_10AM) == 0
        assert on_disk.ledger.spent_kind(PROFILE, WED_10AM) == 0

    def test_a_held_budget_is_still_capped(self, tmp_path, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.PROFILE_LOADS_MAX, "1")
        store = JobStore(tmp_path)
        held = load_account_budget(store, WED_10AM)
        held.ledger.record_kind(PROFILE, WED_10AM)
        token = account_budget_in_use.set(held)
        try:
            with pytest.raises(ActionLimitError):
                charge_navigation(store, "https://www.linkedin.com/in/u/", WED_10AM)
        finally:
            account_budget_in_use.reset(token)


class TestWriteActions:
    def test_a_write_is_refused_before_and_counted_after(self, tmp_path, monkeypatch):
        monkeypatch.setenv(EnvironmentKeys.INVITES_MAX, "1")
        store = JobStore(tmp_path)
        refuse_action(store, INVITES, WED_10AM)
        record_action(store, INVITES, WED_10AM)
        with pytest.raises(ActionLimitError) as excinfo:
            refuse_action(store, INVITES, WED_10AM)
        assert excinfo.value.resume_at == WED_10AM + timedelta(hours=24)
        # Refusing is not counting: still the one invite on disk.
        assert (
            load_account_budget(store, WED_10AM).ledger.spent_kind(INVITES, WED_10AM)
            == 1
        )

    def test_a_ledger_that_cannot_be_saved_costs_the_count_not_the_write(
        self, tmp_path, caplog
    ):
        """The message has left; a full home directory must not turn a
        delivered message into a tool error."""
        store = JobStore(tmp_path)
        with (
            patch.object(store, "save", side_effect=OSError("disk full")),
            caplog.at_level(logging.WARNING),
        ):
            record_action(store, MESSAGES, WED_10AM)
        assert any(
            "Could not count the messages" in r.getMessage() for r in caplog.records
        )

    def test_a_ledger_that_cannot_be_read_refuses_nothing(self, tmp_path, caplog):
        store = JobStore(tmp_path)
        with (
            patch.object(store, "exists", return_value=True),
            patch.object(store, "load", side_effect=OSError("unreadable")),
            caplog.at_level(logging.WARNING),
        ):
            refuse_action(store, INVITES, WED_10AM)
        assert any("allowing the invites" in r.getMessage() for r in caplog.records)

    def test_a_write_is_not_an_action_of_the_daily_budget(self, tmp_path):
        """Its page loads were charged one by one as they happened."""
        store = JobStore(tmp_path)
        record_action(store, MESSAGES, WED_10AM)
        assert load_account_budget(store, WED_10AM).ledger.spent(WED_10AM) == 0


# Aware, because the cooldown is stored as ISO-8601 UTC and compared as such.
T0 = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)


def _strike(store: JobStore, now: datetime, signal: str = "http_429") -> datetime:
    """A full signal, which always answers with the pause it set."""
    until = record_throttle_signal(store, now, signal)
    assert until is not None
    return until


class TestAccountCooldown:
    """One throttle signal pauses the account for every session (issue #57).

    The pause lives in its own file, ``__account_cooldown__.json``, not in
    the budget record: the enrichment tools hold a loaded budget for a whole
    bunch and save it after every profile, and a pause written into that
    record by another process was overwritten by the next such save.
    """

    def test_nothing_is_paused_until_a_signal_arrives(self, tmp_path):
        store = JobStore(tmp_path)
        assert cooldown_resume_at(store, T0) is None
        assert not (tmp_path / ACCOUNT_COOLDOWN_FILE).exists()

    def test_a_signal_pauses_the_account_for_half_an_hour(self, tmp_path):
        store = JobStore(tmp_path)
        until = _strike(store, T0, "http_429")

        assert until == T0 + timedelta(minutes=30)
        # Persisted, and read back fresh as any other process would.
        assert cooldown_resume_at(store, T0 + timedelta(minutes=29)) == until
        assert cooldown_resume_at(store, until) is None
        cooldown = read_account_cooldown(store)
        assert cooldown.strikes == 1
        assert cooldown.last_signal == {"signal": "http_429", "at": T0.isoformat()}

    def test_the_file_is_the_documented_shape(self, tmp_path):
        store = JobStore(tmp_path)
        record_throttle_signal(store, T0, "http_429")

        raw = json.loads((tmp_path / ACCOUNT_COOLDOWN_FILE).read_text())
        assert raw == {
            "until": (T0 + timedelta(minutes=30)).isoformat(),
            "strikes": 1,
            "last_signal": {"signal": "http_429", "at": T0.isoformat()},
            "half_at": None,
        }

    def test_the_budget_record_carries_no_cooldown(self, tmp_path):
        """The blocker: a stale budget save must not be able to clear it."""
        store = JobStore(tmp_path)
        budget = load_account_budget(store, T0)  # loaded before the signal
        record_throttle_signal(store, T0, "http_429")

        budget.ledger.record(T0)
        store.save(budget)  # the enrichment tools' per-profile save

        assert cooldown_resume_at(store, T0) == T0 + timedelta(minutes=30)
        assert "cooldown" not in json.dumps(budget.to_dict())

    def test_the_pause_escalates_and_caps_at_eight_hours(self, tmp_path):
        store = JobStore(tmp_path)
        expected = [timedelta(minutes=30), timedelta(hours=2), timedelta(hours=4)]
        expected += [timedelta(hours=8)] * 2
        assert COOLDOWN_STEPS_SECONDS == (1800, 7200, 14400, 28800)

        now = T0
        for step in expected:
            # Each signal lands once the previous pause has run out, so the
            # escalation is the strike count alone and not a pause extended.
            until = _strike(store, now, "http_429")
            assert until == now + step
            now = until

    def test_strikes_start_over_after_a_day_without_a_signal(self, tmp_path):
        store = JobStore(tmp_path)
        record_throttle_signal(store, T0, "http_429")
        record_throttle_signal(store, T0 + timedelta(hours=1), "http_429")
        assert read_account_cooldown(store).strikes == 2

        later = T0 + timedelta(hours=1) + timedelta(hours=24, seconds=1)
        until = _strike(store, later, "checkpoint")

        assert until == later + timedelta(minutes=30)
        assert read_account_cooldown(store).strikes == 1

    def test_a_day_is_measured_from_the_last_signal_not_the_first(self, tmp_path):
        store = JobStore(tmp_path)
        record_throttle_signal(store, T0, "http_429")
        record_throttle_signal(store, T0 + timedelta(hours=20), "http_429")

        until = _strike(store, T0 + timedelta(hours=25), "http_429")

        assert until == T0 + timedelta(hours=25) + timedelta(hours=4)

    def test_signals_within_a_minute_are_one_incident(self, tmp_path):
        """A 429 is seen by more than one layer of the same call."""
        store = JobStore(tmp_path)
        record_throttle_signal(store, T0, "http_429")
        until = _strike(store, T0 + timedelta(seconds=30), "empty_about")

        assert read_account_cooldown(store).strikes == 1
        assert until == T0 + timedelta(minutes=30)

    def test_a_longer_pause_already_in_force_is_kept(self, tmp_path):
        """Two sessions record signals; the longer pause is the account's."""
        store = JobStore(tmp_path)
        record_throttle_signal(store, T0, "http_429")
        record_throttle_signal(store, T0 + timedelta(minutes=2), "http_429")
        two_hours = T0 + timedelta(minutes=2, hours=2)
        assert read_account_cooldown(store).until == two_hours

        # A third signal, from a clock a little behind: 4h from there is
        # still longer, but the point is that nothing here can go backwards.
        cooldown = read_account_cooldown(store)
        cooldown.until = T0 + timedelta(hours=9)
        (tmp_path / ACCOUNT_COOLDOWN_FILE).write_text(json.dumps(cooldown.to_dict()))
        until = _strike(store, T0 + timedelta(minutes=4), "http_429")
        assert until == T0 + timedelta(hours=9)

    def test_deleting_the_keys_from_the_file_clears_the_pause(self, tmp_path):
        """The documented manual release: no CLI flag, just the JSON."""
        store = JobStore(tmp_path)
        record_throttle_signal(store, T0, "http_429")
        path = tmp_path / ACCOUNT_COOLDOWN_FILE
        raw = json.loads(path.read_text())
        del raw["until"]
        del raw["strikes"]
        path.write_text(json.dumps(raw))

        assert cooldown_resume_at(store, T0) is None
        # And the escalation starts over, since strikes went too.
        assert _strike(store, T0, "x") == T0 + timedelta(minutes=30)

    @pytest.mark.parametrize(
        "content",
        [
            "{not json",
            "null",
            "[]",
            '{"until": null, "strikes": null, "last_signal": null}',
            '{"until": "yesterday", "strikes": "many", "last_signal": "429"}',
            '{"until": 12345, "strikes": -3}',
        ],
    )
    def test_a_hand_edited_file_that_is_garbage_reads_as_no_cooldown(
        self, tmp_path, content
    ):
        store = JobStore(tmp_path)
        (tmp_path / ACCOUNT_COOLDOWN_FILE).write_text(content)

        assert cooldown_resume_at(store, T0) is None
        assert read_account_cooldown(store).strikes == 0
        # And recording over it works, rather than tripping on the garbage.
        assert _strike(store, T0, "x") == T0 + timedelta(minutes=30)

    def test_a_naive_now_is_read_as_local_time(self, tmp_path):
        store = JobStore(tmp_path)
        naive = datetime.now()
        until = _strike(store, naive, "http_429")

        assert until.tzinfo is not None
        assert cooldown_resume_at(store, naive) == until

    def test_a_half_signal_alone_pauses_nothing(self, tmp_path):
        """A payload that never arrived is also what a slow proxy looks like."""
        store = JobStore(tmp_path)

        assert record_throttle_signal(store, T0, "payload_timeout", half=True) is None

        assert cooldown_resume_at(store, T0) is None
        assert read_account_cooldown(store).strikes == 0
        assert read_account_cooldown(store).half_at == T0

    def test_two_half_signals_within_ten_minutes_are_one_strike(self, tmp_path):
        store = JobStore(tmp_path)
        record_throttle_signal(store, T0, "payload_timeout", half=True)
        second = T0 + timedelta(minutes=9)

        until = record_throttle_signal(store, second, "payload_timeout", half=True)

        assert until == second + timedelta(minutes=30)
        cooldown = read_account_cooldown(store)
        assert cooldown.strikes == 1
        assert cooldown.half_at is None  # spent; a third starts a new pair

    def test_two_half_signals_further_apart_stay_half(self, tmp_path):
        store = JobStore(tmp_path)
        record_throttle_signal(store, T0, "payload_timeout", half=True)
        second = T0 + timedelta(minutes=11)

        assert (
            record_throttle_signal(store, second, "payload_timeout", half=True) is None
        )

        assert cooldown_resume_at(store, second) is None
        assert read_account_cooldown(store).half_at == second

    def test_a_full_signal_spends_a_pending_half(self, tmp_path):
        """A half after a full strike must not read as strike two."""
        store = JobStore(tmp_path)
        record_throttle_signal(store, T0, "payload_timeout", half=True)
        _strike(store, T0 + timedelta(minutes=2))  # a 429 arrives instead
        assert read_account_cooldown(store).half_at is None

        later = T0 + timedelta(minutes=5)
        assert (
            record_throttle_signal(store, later, "payload_timeout", half=True) is None
        )
        assert read_account_cooldown(store).strikes == 1
        assert read_account_cooldown(store).half_at == later

    def test_the_default_ledger_helper_writes_the_same_record(
        self, tmp_path, monkeypatch
    ):
        """What the raise sites call: no store, no clock, never raises."""
        from linkedin_mcp_server import pacing

        monkeypatch.setattr(pacing, "JobStore", lambda: JobStore(tmp_path))
        note_throttle_signal("http_429")

        cooldown = read_account_cooldown(JobStore(tmp_path))
        assert cooldown.strikes == 1
        assert cooldown.last_signal is not None
        assert cooldown.last_signal["signal"] == "http_429"

    def test_the_default_ledger_helper_passes_half_through(self, tmp_path, monkeypatch):
        from linkedin_mcp_server import pacing

        monkeypatch.setattr(pacing, "JobStore", lambda: JobStore(tmp_path))
        note_throttle_signal("payload_timeout", half=True)

        cooldown = read_account_cooldown(JobStore(tmp_path))
        assert cooldown.strikes == 0
        assert cooldown.half_at is not None

    def test_the_default_ledger_helper_is_the_opt_out(self, tmp_path, monkeypatch):
        from linkedin_mcp_server import pacing

        monkeypatch.setattr(pacing, "JobStore", lambda: JobStore(tmp_path))
        monkeypatch.setenv(EnvironmentKeys.ACCOUNT_COOLDOWN_DISABLED, "1")
        note_throttle_signal("http_429")

        assert not (tmp_path / ACCOUNT_COOLDOWN_FILE).exists()

    def test_the_default_ledger_helper_swallows_an_unwritable_home(
        self, tmp_path, monkeypatch
    ):
        from linkedin_mcp_server import pacing

        def broken():
            raise OSError("read-only file system")

        monkeypatch.setattr(pacing, "JobStore", broken)
        note_throttle_signal("http_429")  # must not raise

    def test_the_cooldown_file_is_not_a_job(self, tmp_path):
        store = JobStore(tmp_path / "jobs")
        record_throttle_signal(store, T0, "http_429")
        load_account_budget(store, T0)

        assert store.list_jobs() == [ACCOUNT_BUDGET_JOB]


class TestLedgerLock:
    """Two processes writing one record must not lose each other's update."""

    def test_load_and_save_take_the_lock_and_release_it(self, tmp_path):
        store = JobStore(tmp_path)
        load_account_budget(store, T0)
        assert (tmp_path / LEDGER_LOCK_FILE).exists()
        # Released: a second exclusive lock from this process is granted at
        # once. flock conflicts between two descriptors of one process, so a
        # leaked lock would block here.
        fd = os.open(tmp_path / LEDGER_LOCK_FILE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def test_the_lock_is_reentrant_within_the_process(self, tmp_path):
        store = JobStore(tmp_path)
        with store.locked():
            budget = load_account_budget(store, T0)  # locks again inside
            budget.ledger.record(T0)
            store.save(budget)
            record_throttle_signal(store, T0, "http_429")
        assert load_account_budget(store, T0).ledger.spent(T0) == 1

    def test_a_held_lock_blocks_another_process(self, tmp_path):
        """A real second process, because flock is per open file description."""
        store = JobStore(tmp_path)
        load_account_budget(store, T0)
        script = (
            "import fcntl, os, sys\n"
            f"fd = os.open({str(tmp_path / LEDGER_LOCK_FILE)!r}, os.O_RDWR)\n"
            "try:\n"
            "    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "except OSError:\n"
            "    sys.exit(3)\n"
            "sys.exit(0)\n"
        )
        with store.locked():
            held = subprocess.run([sys.executable, "-c", script])
        free = subprocess.run([sys.executable, "-c", script])
        assert held.returncode == 3
        assert free.returncode == 0

    def test_saves_use_a_per_process_temp_name(self, tmp_path):
        """One shared ``.json.tmp`` let two savers rename each other's
        half-written file into place."""
        store = JobStore(tmp_path)
        budget = load_account_budget(store, T0)
        seen: list[str] = []
        original = os.replace

        def spy(src, dst):
            seen.append(os.path.basename(src))
            return original(src, dst)

        with patch("linkedin_mcp_server.common_utils.os.replace", spy):
            store.save(budget)
            store.save(budget)
        assert len(seen) == 2 and seen[0] != seen[1]
        assert list(tmp_path.glob("*.tmp")) == []


class TestHourlyCap:
    """Forty LinkedIn-touching actions a rolling hour, from the shared ledger."""

    def _budget_with(self, count: int, at: datetime) -> Job:
        budget = Job(name=ACCOUNT_BUDGET_JOB, started_on=at.date())
        for _ in range(count):
            budget.ledger.record(at)
        return budget

    def test_the_default_cap_is_forty(self):
        assert hourly_cap_resume_at(self._budget_with(39, T0), T0) is None
        assert hourly_cap_resume_at(self._budget_with(40, T0), T0) is not None

    def test_release_is_when_the_oldest_ages_out(self):
        budget = self._budget_with(39, T0)
        budget.ledger.record(T0 + timedelta(minutes=10))
        now = T0 + timedelta(minutes=20)

        resume = hourly_cap_resume_at(budget, now)

        # 40 inside the hour; the slot frees when the oldest of them does.
        assert resume == T0 + timedelta(seconds=HOURLY_WINDOW_SECONDS)
        just_after = T0 + timedelta(seconds=HOURLY_WINDOW_SECONDS + 1)
        assert hourly_cap_resume_at(budget, just_after) is None

    def test_actions_older_than_an_hour_do_not_count(self):
        budget = self._budget_with(40, T0)
        assert hourly_cap_resume_at(budget, T0 + timedelta(hours=1, seconds=1)) is None

    def test_a_backlog_beyond_the_cap_releases_one_slot_at_a_time(self):
        """Fifty inside the hour: the eleventh-oldest expiring frees the slot,
        not the oldest, because forty-nine would still be inside."""
        budget = Job(name=ACCOUNT_BUDGET_JOB, started_on=T0.date())
        for i in range(50):
            budget.ledger.record(T0 + timedelta(seconds=i))

        resume = hourly_cap_resume_at(budget, T0 + timedelta(minutes=30), cap=40)

        assert resume == T0 + timedelta(seconds=10 + HOURLY_WINDOW_SECONDS)

    def test_the_cap_is_not_a_strike(self, tmp_path):
        store = JobStore(tmp_path)
        budget = load_account_budget(store, T0)
        for _ in range(40):
            budget.ledger.record(T0)
        store.save(budget)

        assert hourly_cap_resume_at(budget, T0) is not None
        assert read_account_cooldown(store).strikes == 0
        assert cooldown_resume_at(store, T0) is None

    def test_release_waits_for_as_many_slots_as_are_needed(self):
        """A bunch that needs two must wait for the second-oldest to age out."""
        budget = Job(name=ACCOUNT_BUDGET_JOB, started_on=T0.date())
        for i in range(3):
            budget.ledger.record(T0 + timedelta(minutes=i))
        now = T0 + timedelta(minutes=30)

        assert hourly_cap_resume_at(budget, now, cap=3, needed=1) == T0 + timedelta(
            seconds=HOURLY_WINDOW_SECONDS
        )
        assert hourly_cap_resume_at(budget, now, cap=3, needed=2) == T0 + timedelta(
            minutes=1, seconds=HOURLY_WINDOW_SECONDS
        )
        # One of headroom is enough for one, not for two.
        assert hourly_cap_resume_at(budget, now, cap=4, needed=1) is None
        assert hourly_cap_resume_at(budget, now, cap=4, needed=2) == T0 + timedelta(
            seconds=HOURLY_WINDOW_SECONDS
        )

    def test_needing_more_than_the_cap_does_not_index_past_the_hour(self, monkeypatch):
        """Reproduced: cap=3, needed=5 raised IndexError, as did cap=1,
        needed=2 on an empty ledger. Reachable whenever HOURLY_ACTIONS_MAX
        sits below one profile's cost."""
        empty = Job(name=ACCOUNT_BUDGET_JOB, started_on=T0.date())
        assert hourly_cap_resume_at(empty, T0, cap=1, needed=2) is None
        assert seconds_until_hourly_release(empty, T0, needed=2) == 0.0

        budget = Job(name=ACCOUNT_BUDGET_JOB, started_on=T0.date())
        for i in range(3):
            budget.ledger.record(T0 + timedelta(minutes=i))
        now = T0 + timedelta(minutes=30)
        # Past the last entry the answer is when the whole hour has drained.
        assert hourly_cap_resume_at(budget, now, cap=3, needed=5) == T0 + timedelta(
            minutes=2, seconds=HOURLY_WINDOW_SECONDS
        )
        monkeypatch.setenv(EnvironmentKeys.HOURLY_ACTIONS_MAX, "3")
        assert seconds_until_hourly_release(budget, now, needed=5) == pytest.approx(
            32 * 60
        )

    def test_headroom_is_what_the_hour_still_admits(self):
        budget = self._budget_with(30, T0)
        assert hourly_headroom(budget, T0) == 10
        assert hourly_headroom(budget, T0, cap=25) == 0
        assert hourly_headroom(budget, T0 + timedelta(hours=1, seconds=1)) == 40
