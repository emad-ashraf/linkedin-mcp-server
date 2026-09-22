"""What a throttled navigation looks like from the driver's side.

Kept in ``core`` because both the auth probe (``core/auth.py``,
``drivers/browser.py``) and the scraping navigator read them, and the probe
must not import the scraping package to do so. ``scraping/rate_limit.py``
re-exports them next to the backoff parameters.
"""

# A 429 that Chromium *does* commit comes back as an ordinary response, and
# the reason ``page.goto``'s return value is never discarded. Measured against
# a local server answering 429, with and without a body, under both
# ``wait_until="domcontentloaded"`` and ``"commit"``: ``goto`` returns rather
# than raising, ``status`` is 429 and ``Retry-After`` survives on ``headers``.
HTTP_TOO_MANY_REQUESTS = 429

# Throttling's other shape: LinkedIn bounces a request it will not serve
# between routes until Chromium gives up. The loop can pass through an auth
# route, so the URL it stops on says nothing about the session, and no barrier
# is read off a page that ended this way.
REDIRECT_LOOP_NAV_FAILURE = "ERR_TOO_MANY_REDIRECTS"
