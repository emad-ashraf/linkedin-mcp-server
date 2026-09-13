"""Tests for the facet resolver shared by people and company search."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import logging

import pytest

from linkedin_mcp_server.company_cache import CompanyCache
from linkedin_mcp_server.scraping import facets as facets_module
from linkedin_mcp_server.scraping.capture import (
    CaptureMode,
    CapturePlan,
    SectionCapture,
)
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    ExtractedSection,
    FilterValidationError,
)
from linkedin_mcp_server.scraping.facets import FacetResolver
from linkedin_mcp_server.scraping.link_metadata import Reference
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession


def _resolver(page) -> FacetResolver:
    """Wire the resolver the way the facade does."""
    session = ScrapingSession(page)
    navigator = PageNavigator(session)
    capture = SectionCapture(session, navigator, PageContentReader(session))
    return FacetResolver(session, navigator, capture)


def extracted(
    text: str,
    references: list[Reference] | None = None,
    error: dict | None = None,
) -> ExtractedSection:
    """Create an ExtractedSection for tests."""
    return ExtractedSection(text=text, references=references or [], error=error)


def _sleep():
    return patch(
        "linkedin_mcp_server.scraping.session.asyncio.sleep", new_callable=AsyncMock
    )


class TestResolveGeoUrn:
    async def test_resolve_geo_urn_reads_geoid_and_caches(self, mock_page):
        """Drives the dropdown once: types the name, clicks the top suggestion,
        reads geoId from the URL, and caches it (second call does not re-drive)."""
        resolver = _resolver(mock_page)
        box = AsyncMock()
        suggestion = AsyncMock()
        # The location box answers the first selector; the dropdown the next.
        mock_page.query_selector = AsyncMock(side_effect=[box, suggestion])
        mock_page.url = "https://www.linkedin.com/jobs/search/?geoId=106155005&foo=1"
        with (
            patch.object(
                resolver._navigator, "_goto_with_auth_checks", new_callable=AsyncMock
            ) as goto,
            patch(
                "linkedin_mcp_server.scraping.facets.human_type",
                new_callable=AsyncMock,
            ) as typed,
            _sleep(),
        ):
            first = await resolver.resolve_geo_urn("Egypt")
            assert first == "106155005"
            assert resolver._geo_cache["egypt"] == "106155005"
            # The typeahead page is a navigation, so the results page that
            # follows it gets paced like one after a company lookup.
            assert resolver.navigated is True

            # Second call (case-insensitive) is served from cache: the dropdown
            # is not driven again, so no further navigation happens.
            assert await resolver.resolve_geo_urn("EGYPT") == "106155005"
            assert goto.await_count == 1

        goto.assert_awaited_once_with("https://www.linkedin.com/jobs/search/?keywords=")
        box.fill.assert_awaited_once_with("")
        typed.assert_awaited_once_with(mock_page, "Egypt")
        suggestion.click.assert_awaited_once()

    async def test_no_suggestion_is_a_cached_miss(self, mock_page):
        resolver = _resolver(mock_page)
        box = AsyncMock()
        # The location box is found; the dropdown then offers nothing.
        mock_page.query_selector = AsyncMock(side_effect=[box, None])
        mock_page.url = "https://www.linkedin.com/jobs/search/?keywords="
        with (
            patch.object(
                resolver._navigator, "_goto_with_auth_checks", new_callable=AsyncMock
            ) as goto,
            patch(
                "linkedin_mcp_server.scraping.facets.human_type",
                new_callable=AsyncMock,
            ),
            _sleep(),
        ):
            assert await resolver.resolve_geo_urn("Nowhereland") is None
            assert await resolver.resolve_geo_urn("nowhereland") is None

        assert resolver._geo_cache["nowhereland"] == ""
        assert goto.await_count == 1

    async def test_no_location_box_is_a_miss_without_typing(self, mock_page):
        resolver = _resolver(mock_page)
        mock_page.query_selector = AsyncMock(return_value=None)
        with (
            patch.object(
                resolver._navigator, "_goto_with_auth_checks", new_callable=AsyncMock
            ),
            patch(
                "linkedin_mcp_server.scraping.facets.human_type",
                new_callable=AsyncMock,
            ) as typed,
        ):
            assert await resolver.resolve_geo_urn("Egypt") is None

        typed.assert_not_awaited()


class TestResolveCompanyUrn:
    """``current_company`` accepts a name, a /company/ URL, or the numeric id."""

    SEARCH = "https://www.linkedin.com/search/results/companies/?keywords=SAP"
    ABOUT = "https://www.linkedin.com/company/sap/about/"

    @staticmethod
    def _resolver(mock_page, tmp_path) -> FacetResolver:
        resolver = _resolver(mock_page)
        resolver._company_cache = CompanyCache(tmp_path)
        return resolver

    @staticmethod
    def _urn_ref(urn: str) -> Reference:
        return {
            "kind": "company_urn",
            "url": f"/search/results/people/?currentCompany=%5B%22{urn}%22%5D",
            "value": urn,
        }

    @staticmethod
    def _company_ref(slug: str) -> Reference:
        return {"kind": "company", "url": f"/company/{slug}/", "text": slug}

    @staticmethod
    def _nav(resolver: FacetResolver, **kwargs):
        """Patch the one seam both hops go through: ``extract_page`` (the About
        page) is the compatibility adapter over ``capture`` (the search)."""
        return patch.object(
            resolver._capture, "capture", new_callable=AsyncMock, **kwargs
        )

    async def test_digits_pass_through_without_navigating(self, mock_page, tmp_path):
        resolver = self._resolver(mock_page, tmp_path)
        with self._nav(resolver) as nav:
            assert await resolver.resolve_company_urn("1115") == "1115"

        nav.assert_not_awaited()
        assert resolver._company_urn_cache == {}

    async def test_the_disk_cache_opens_on_first_use_only(self, mock_page):
        resolver = _resolver(mock_page)
        with self._nav(resolver) as nav:
            assert await resolver.resolve_company_urn("1115") == "1115"

        nav.assert_not_awaited()
        assert resolver._company_cache is None

    async def test_disk_cache_hit_costs_no_navigation(self, mock_page, tmp_path):
        """A company the enrichment tools already researched is keyed by its
        normalised name, so the spelling need not match."""
        resolver = self._resolver(mock_page, tmp_path)
        assert resolver._company_cache is not None
        resolver._company_cache.record_firmographics(
            "SAP, Inc.", datetime.now(), source="company_page", company_urn="1115"
        )
        with self._nav(resolver) as nav:
            assert await resolver.resolve_company_urn("sap") == "1115"

        nav.assert_not_awaited()
        assert resolver._company_urn_cache["sap"] == "1115"
        assert resolver.navigated is False

    async def test_unresolvable_name_raises_and_is_remembered(
        self, mock_page, tmp_path
    ):
        resolver = self._resolver(mock_page, tmp_path)
        with self._nav(resolver, return_value=extracted("No results found")) as nav:
            with pytest.raises(FilterValidationError) as excinfo:
                await resolver.resolve_company_urn("Nowhere Corp")
            # The miss is cached, so a repeated name in a batch does not search
            # again -- and still raises rather than silently passing the name.
            with pytest.raises(FilterValidationError):
                await resolver.resolve_company_urn("nowhere corp")

        assert "Nowhere Corp" in str(excinfo.value)
        assert "get_company_profile" in str(excinfo.value)
        assert nav.await_count == 1
        assert resolver._company_cache is not None
        assert resolver._company_cache.get("Nowhere Corp") is None

    async def test_name_resolves_via_search_then_about(self, mock_page, tmp_path):
        """Search finds the slug, the About page carries the id, and both
        caches learn it so the next call (any spelling) is free."""
        resolver = self._resolver(mock_page, tmp_path)
        pages = {
            self.SEARCH: extracted(
                "SAP\nSoftware",
                [self._company_ref("sap"), self._company_ref("sap-labs")],
            ),
            self.ABOUT: extracted("About SAP", [self._urn_ref("1115")]),
        }
        with (
            self._nav(resolver, side_effect=lambda url, *_, **__: pages[url]) as nav,
            _sleep(),
        ):
            assert await resolver.resolve_company_urn("SAP") == "1115"
            assert [c.args[0] for c in nav.await_args_list] == [
                self.SEARCH,
                self.ABOUT,
            ]

            assert await resolver.resolve_company_urn("sap") == "1115"
            assert nav.await_count == 2

        # The search is captured as a results page, the About page as itself.
        assert nav.await_args_list[0].args[1:] == (
            "search_results",
            CapturePlan(CaptureMode.SEARCH_RESULTS),
        )
        assert nav.await_args_list[1].args[1] == "about"
        assert resolver.navigated is True
        assert resolver._company_urn_cache["sap"] == "1115"
        assert resolver._company_cache is not None
        record = resolver._company_cache.get("SAP")
        assert record is not None
        assert record.company_urn == "1115"
        assert record.linkedin_url == "https://www.linkedin.com/company/sap"
        # Learning an id is not learning firmographics: the record must not
        # read as fresh to the enrichment tools.
        assert not record.has_firmographics()

    async def test_search_card_with_own_urn_skips_about_page(self, mock_page, tmp_path):
        """An id anchor between the top card's link and the next card's is the
        top card's own, and saves the second navigation."""
        resolver = self._resolver(mock_page, tmp_path)
        refs = [
            self._company_ref("sap"),
            self._urn_ref("1115"),
            self._company_ref("sap-labs"),
            self._urn_ref("999"),
        ]
        with self._nav(resolver, return_value=extracted("SAP", refs)) as nav:
            assert await resolver.resolve_company_urn("SAP") == "1115"

        assert nav.await_count == 1

    async def test_urn_after_second_card_is_not_attributed(self, mock_page, tmp_path):
        resolver = self._resolver(mock_page, tmp_path)
        pages = {
            self.SEARCH: extracted(
                "SAP",
                [
                    self._company_ref("sap"),
                    self._company_ref("sap-labs"),
                    self._urn_ref("999"),
                ],
            ),
            self.ABOUT: extracted("About SAP", [self._urn_ref("1115")]),
        }
        with (
            self._nav(resolver, side_effect=lambda url, *_, **__: pages[url]) as nav,
            _sleep(),
        ):
            assert await resolver.resolve_company_urn("SAP") == "1115"

        assert nav.await_count == 2

    async def test_urn_before_any_card_is_not_attributed(self, mock_page, tmp_path):
        resolver = self._resolver(mock_page, tmp_path)
        pages = {
            self.SEARCH: extracted(
                "SAP", [self._urn_ref("999"), self._company_ref("sap")]
            ),
            self.ABOUT: extracted("About SAP", [self._urn_ref("1115")]),
        }
        with (
            self._nav(resolver, side_effect=lambda url, *_, **__: pages[url]) as nav,
            _sleep(),
        ):
            assert await resolver.resolve_company_urn("SAP") == "1115"

        assert nav.await_count == 2

    async def test_company_url_goes_straight_to_about(self, mock_page, tmp_path):
        resolver = self._resolver(mock_page, tmp_path)
        with self._nav(
            resolver, return_value=extracted("About SAP", [self._urn_ref("1115")])
        ) as nav:
            urn = await resolver.resolve_company_urn(
                "https://www.linkedin.com/company/sap/"
            )

        assert urn == "1115"
        nav.assert_awaited_once()
        assert nav.await_args_list[-1].args[0] == self.ABOUT
        assert resolver._company_cache is not None
        record = resolver._company_cache.get("sap")
        assert record is not None
        assert record.company_urn == "1115"

    async def test_throttled_lookup_says_so(self, mock_page, tmp_path):
        resolver = self._resolver(mock_page, tmp_path)
        with self._nav(resolver, return_value=extracted(RATE_LIMITED_SECTION_TEXT)):
            with pytest.raises(FilterValidationError, match="throttled"):
                await resolver.resolve_company_urn("SAP")

    async def test_disk_record_with_url_only_skips_the_search(
        self, mock_page, tmp_path
    ):
        """``enrich_companies`` stores the page URL from a search hit but no
        id; the slug in that URL is enough to go straight to About."""
        resolver = self._resolver(mock_page, tmp_path)
        assert resolver._company_cache is not None
        resolver._company_cache.record_firmographics(
            "SAP",
            datetime.now(),
            source="search",
            linkedin_url="https://www.linkedin.com/company/sap",
        )
        with self._nav(
            resolver, return_value=extracted("About SAP", [self._urn_ref("1115")])
        ) as nav:
            assert await resolver.resolve_company_urn("sap") == "1115"

        nav.assert_awaited_once()
        assert nav.await_args_list[0].args[0] == self.ABOUT
        record = resolver._company_cache.get("SAP")
        assert record is not None
        assert record.company_urn == "1115"

    async def test_promoted_top_card_for_another_company_is_skipped(
        self, mock_page, tmp_path
    ):
        """The first card is often an ad for a different company; the card
        whose name is the query wins, and the first card's own id anchor is
        not attributed to it."""
        resolver = self._resolver(mock_page, tmp_path)
        pages = {
            self.SEARCH: extracted(
                "Page by Deloitte\nSAP",
                [
                    {
                        "kind": "company",
                        "url": "/company/deloitte/",
                        "text": "Page by Deloitte",
                    },
                    self._urn_ref("999"),
                    self._company_ref("sap"),
                ],
            ),
            self.ABOUT: extracted("About SAP", [self._urn_ref("1115")]),
        }
        with (
            self._nav(resolver, side_effect=lambda url, *_, **__: pages[url]) as nav,
            _sleep(),
        ):
            assert await resolver.resolve_company_urn("SAP") == "1115"

        assert [c.args[0] for c in nav.await_args_list] == [self.SEARCH, self.ABOUT]
        assert resolver._company_cache is not None
        assert resolver._company_cache.get("deloitte") is None
        record = resolver._company_cache.get("sap")
        assert record is not None
        assert record.company_urn == "1115"

    async def test_no_card_named_like_the_query_raises_with_candidates(
        self, mock_page, tmp_path
    ):
        resolver = self._resolver(mock_page, tmp_path)
        with self._nav(
            resolver,
            return_value=extracted(
                "Deloitte",
                [self._company_ref("deloitte"), self._company_ref("deloitte-digital")],
            ),
        ) as nav:
            with pytest.raises(FilterValidationError) as excinfo:
                await resolver.resolve_company_urn("Deloitte Consulting")
            with pytest.raises(FilterValidationError):
                await resolver.resolve_company_urn("deloitte consulting")

        message = str(excinfo.value)
        assert "'deloitte'" in message
        assert "'deloitte-digital'" in message
        assert "/company/<slug>/" in message
        # No About page was loaded for a card that was never accepted, and
        # the clean miss is remembered for the batch.
        assert nav.await_count == 1
        assert resolver._company_cache is not None
        assert resolver._company_cache.get("Deloitte Consulting") is None

    async def test_cache_entry_is_written_under_the_hit_name(self, mock_page, tmp_path):
        """The record carries the name LinkedIn shows, not the query's
        spelling; both normalise to the same key."""
        resolver = self._resolver(mock_page, tmp_path)
        refs: list[Reference] = [
            {"kind": "company", "url": "/company/sap/", "text": "SAP, Inc."},
            self._urn_ref("1115"),
        ]
        with self._nav(resolver, return_value=extracted("SAP, Inc.", refs)):
            assert await resolver.resolve_company_urn("sap") == "1115"

        assert resolver._company_cache is not None
        record = resolver._company_cache.get("sap")
        assert record is not None
        assert record.display_name == "SAP, Inc."

    @pytest.mark.parametrize(
        "bad_page",
        [
            extracted(RATE_LIMITED_SECTION_TEXT),
            extracted("", error={"error_type": "NetworkError"}),
        ],
        ids=["throttled", "errored"],
    )
    async def test_a_throttled_or_failed_miss_is_not_cached(
        self, mock_page, tmp_path, bad_page
    ):
        """A retry may succeed, so the second call searches again."""
        resolver = self._resolver(mock_page, tmp_path)
        with self._nav(resolver, return_value=bad_page) as nav, _sleep():
            with pytest.raises(FilterValidationError):
                await resolver.resolve_company_urn("SAP")
            with pytest.raises(FilterValidationError):
                await resolver.resolve_company_urn("SAP")

        assert nav.await_count == 2
        assert "sap" not in resolver._company_urn_cache

    async def test_every_hop_between_two_resolutions_is_paced(
        self, mock_page, tmp_path
    ):
        """Two names resolve as search(A), about(A), search(B), about(B):
        four navigations, so three pauses. The about(A) -> search(B) hop
        is the one a per-resolution pause alone misses."""
        resolver = self._resolver(mock_page, tmp_path)
        pages = {
            self.SEARCH: extracted("SAP", [self._company_ref("sap")]),
            self.ABOUT: extracted("About SAP", [self._urn_ref("1115")]),
            "https://www.linkedin.com/search/results/companies/?keywords=Bosch": (
                extracted("Bosch", [self._company_ref("bosch")])
            ),
            "https://www.linkedin.com/company/bosch/about/": extracted(
                "About Bosch", [self._urn_ref("2222")]
            ),
        }
        with (
            self._nav(resolver, side_effect=lambda url, *_, **__: pages[url]) as nav,
            _sleep() as sleep,
            patch(
                "linkedin_mcp_server.scraping.session.jitter",
                side_effect=lambda base, spread=0.5: base,
            ),
            patch.object(facets_module, "nav_delay", return_value=7.0),
        ):
            assert await resolver.resolve_company_urn("SAP") == "1115"
            assert await resolver.resolve_company_urn("Bosch") == "2222"

        assert nav.await_count == 4
        assert [c.args for c in sleep.await_args_list] == [(7.0,)] * 3

    async def test_the_first_navigation_of_a_batch_is_not_paced(
        self, mock_page, tmp_path
    ):
        resolver = self._resolver(mock_page, tmp_path)
        with (
            self._nav(
                resolver,
                return_value=extracted("About SAP", [self._urn_ref("1115")]),
            ),
            _sleep() as sleep,
        ):
            await resolver.resolve_company_urn("https://www.linkedin.com/company/sap/")

        sleep.assert_not_awaited()
        assert resolver.navigated is True

    async def test_a_failing_write_back_does_not_fail_the_resolution(
        self, mock_page, tmp_path, caplog
    ):
        """``CompanyCache._path`` refuses a name that normalises to nothing;
        the id is the result, the cache write is not."""
        resolver = self._resolver(mock_page, tmp_path)
        refs: list[Reference] = [
            {"kind": "company", "url": "/company/group/", "text": "Group"},
            self._urn_ref("42"),
        ]
        with (
            self._nav(resolver, return_value=extracted("Group", refs)),
            caplog.at_level(logging.WARNING, logger=facets_module.__name__),
        ):
            assert await resolver.resolve_company_urn("Group") == "42"

        assert resolver._company_urn_cache["group"] == "42"
        assert any("Could not cache company urn" in r.message for r in caplog.records)
