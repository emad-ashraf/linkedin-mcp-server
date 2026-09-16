# Messaging read-state preservation

`get_inbox`, `search_conversations`, and `get_conversation` are read-only from
LinkedIn's perspective. They must not select a conversation row, navigate to the
requested thread route, send a read receipt, or use a compensating “mark unread”
write.

## How it works

The messaging page already requests structured conversation data. That response
contains stable conversation URLs, provider URNs, participant profile URLs, and
the `read` boolean. Inbox and search references are built from that response
instead of click-visiting rows.

For `get_conversation`, the reader captures LinkedIn's own authenticated
`messengerMessages` GET request, changes only its exact conversation URN, and
replays it in the same page. It accepts the response only when the HTTP status is
successful and every returned message names one of the target conversation's
exact provider URNs. If LinkedIn has already emitted its anchor-timestamp form
of the same GET operation during the passive page load, the reader can use it
for up to three older batches and deduplicates overlapping anchor messages. It
never opens another thread merely to discover that private query shape; without
a passive history template it returns the verified current batch. Username
resolution matches the requested `/in/` identifier against participant profile
URLs; display names alone are never treated as identity.

This keeps all three tool annotations truthfully read-only and removes the
latency, cancellation window, and provider-side read-receipt effects of
click-and-restore designs.

## Failure behavior and limits

- Missing payloads, an unavailable request template, non-2xx responses, and
  responses that cannot be tied to the exact target fail explicitly. The reader
  does not fall back to opening the thread.
- A direct thread ID must be present in the passively loaded inbox window. The
  reader makes one bounded pagination attempt, then fails instead of deriving or
  guessing a private provider URN from the route ID.
- The current provider batch is normally at most 20 messages. Older batches are
  included only when LinkedIn passively supplies its history query template;
  the reader does not trade read-state safety for deeper history. Message text
  is returned chronologically with sender/date/time headings compatible with
  the existing rendered-conversation response. Stable participant and
  attachment links are returned as references. Person references come only
  from the matched conversation's participant list, never from profile links
  embedded in message content. Provider events or attachments
  without readable text may still have no text block; a thread with no readable
  messages may return an empty `sections` object.
- A simultaneous manual LinkedIn session can change state independently while a
  tool runs. The tools make no read-state writes of their own.
- LinkedIn's private response schema and query decoration can change. Parser and
  identity checks fail closed instead of guessing from localized UI text or
  unstable CSS classes.
- Messaging diagnostics redact thread routes, search terms, and participant
  profile routes. Retained debug traces do not capture private-page screenshots,
  body text, titles, or cookie names.

## Measurement record

Measured against an isolated authenticated test profile on 2026-09-15:

- Passive inbox loads returned 20 stable conversation URLs and participant
  identities with mixed `read=true` and `read=false` state.
- Repeated inbox loads preserved an unselected unread control.
- Loading messaging while the unread control was selected preserved it and
  emitted no observed `markAsRead` request.
- Replaying `messengerMessages` for the unread control returned 20 message
  entities, matched the requested conversation URN, and left `read=false`.
- Functional calls through all four identification paths— inbox, controlled
  search, direct thread ID, and username—left the unread control unread.

These observations are provider behavior, not a public LinkedIn API guarantee;
the fail-closed checks above are the permanent contract.

Measured again in a container-native authenticated session on 2026-09-16:
profile, bounded feed and people search, inbox, and a nonempty conversation
completed through one MCP session. A direct thread lookup initially timed out
when the inbox inventory did not arrive without scrolling; one bounded inbox
scroll resolved that case without opening the thread. A separate Docker-native
read/unread control run preserved both states across inbox, search, direct-ID,
and username reads, with no observed `markAsRead` request.

Cerebro's current adapter accepts the response envelope and fails safely to
partial coverage when a conversation reference has no participant label and a
one-sided conversation does not prove both identities. The provider release
alone therefore does not complete Cerebro issue #247; its pinned-runtime,
adapter, diagnostics, and acceptance work remain a separate follow-up.
