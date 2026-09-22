"""Fixtures shared by the scraping unit tests."""

from __future__ import annotations

import pytest

from linkedin_mcp_server.scraping import navigation as navigation_module


@pytest.fixture(autouse=True)
def reprobe_reads_the_same_barrier(monkeypatch):
    """The /feed/ re-probe in core.auth sees whatever a test patched here.

    Tests patch the names ``navigation`` imported, while the second look runs
    inside ``core.auth.barrier_confirmed`` and would otherwise read the mock
    page for real, where a page that "keeps showing a picker" shows nothing.
    Only the quick detector needs this: the navigator hands the full one over
    itself when that is what sighted the barrier.
    """

    async def delegate(page):
        return await navigation_module.detect_auth_barrier_quick(page)

    monkeypatch.setattr(
        "linkedin_mcp_server.core.auth.detect_auth_barrier_quick", delegate
    )
