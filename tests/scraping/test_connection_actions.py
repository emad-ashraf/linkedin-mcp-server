"""Tests for the invitation-action owner.

The write gate is what most of these hold: the invite deeplink may only be
opened to submit once LinkedIn has exposed the vanityName invite anchor, and
the one other thing allowed to open it — the note-quota probe — never clicks
a primary button. Every case that reaches a decision drives the real
classifier from structural signals; no case reads a label.

``tests/test_action_signals_dom.py`` covers the other half, where the
programs run against a real DOM in four label sets. Here ``page.evaluate`` is
a mock, so the JS never executes and the signals are supplied directly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.scraping.connection import ActionSignals
from linkedin_mcp_server.scraping.connection_actions import (
    _DIALOG_SELECTOR,
    _MODAL_DIALOG_INDEX_JS,
    _SENT_LIST_HAS_USER_JS,
    ConnectionActions,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

PREMIUM_MESSAGE = (
    "Wysyłaj nieograniczoną liczbę spersonalizowanych zaproszeń dzięki Premium"
)
INVITE_URL = "https://www.linkedin.com/preload/custom-invite/?vanityName=testuser"
DIALOG_TEXT = "Invite Jane to connect Add a note Send without a note"
LIMIT_DIALOG_TEXT = "You've reached the weekly invitation limit Got it"


def _actions(page, read_main_profile: Any = None) -> ConnectionActions:
    """Wire the connection owner the way the facade does.

    The facade hands over one main-profile read and nothing else of the
    person workflow, so the borrow is the whole collaborator surface a test
    has to supply. The default refuses the call: a case that never reads a
    profile should not be able to, and one that does says which texts it
    expects.
    """

    async def unread(_username: str) -> dict[str, Any]:
        raise AssertionError("this case does not read a profile")

    session = ScrapingSession(page)
    return ConnectionActions(
        session,
        PageNavigator(session),
        read_main_profile if read_main_profile is not None else unread,
    )


def _reads(*texts: str) -> AsyncMock:
    """Script the main-profile read: one answer per call, in order.

    A single text answers every call, which is what the states that never
    re-read need. Two or more script the verification re-reads an action
    performs, and a further call raises ``StopIteration`` rather than
    quietly repeating the last page.
    """
    pages = [
        {
            "url": "https://www.linkedin.com/in/testuser/",
            "sections": {"main_profile": text} if text else {},
        }
        for text in texts
    ]
    if len(pages) == 1:
        return AsyncMock(return_value=pages[0])
    return AsyncMock(side_effect=pages)


def _scope_dialog(mock_page, *, buttons: MagicMock, textarea: MagicMock) -> MagicMock:
    """Wire ``page.locator(dialog).nth(i)`` down to the two doubles.

    Every dialog control is reached through ``_invite_dialog``, so a flat
    ``page.locator`` routed by selector never hands out its buttons: the
    scoped chain is ``dialogs -> dialog -> dialog.locator(selector)``, and
    it starts from the modal index the page reports. The index script is
    answered with 0 here so the chain resolves at all; there is no
    ``.first`` on the dialogs double, so a read that fell back to it would
    reach an unawaitable control rather than these. Which index the script
    picks is ``TestInviteDialogScope``'s business.
    """
    dialog = MagicMock()
    dialog.locator = MagicMock(
        side_effect=lambda selector: textarea if "textarea" in selector else buttons
    )
    dialogs = MagicMock(spec=["nth"])
    dialogs.nth = MagicMock(return_value=dialog)
    mock_page.locator = MagicMock(return_value=dialogs)
    _modal_at(mock_page, 0)
    return dialog


def _modal_at(mock_page, index: int) -> None:
    """Have the page report the invite modal at ``index`` (-1: none open).

    Every ``page.evaluate`` answers with it, which is fine for the one other
    script the submit path runs unpatched (the button dump, logged and
    otherwise ignored).
    """
    mock_page.evaluate = AsyncMock(return_value=index)


def _no_poll_sleep():
    return patch(
        "linkedin_mcp_server.scraping.connection_actions.asyncio.sleep",
        new_callable=AsyncMock,
    )


def _signals(
    invite: bool = False,
    compose: bool = False,
    edit: bool = False,
    labeled_action: bool = False,
    labeled_anchor: bool = False,
    incoming_row: bool = False,
) -> ActionSignals:
    return ActionSignals(
        has_invite_anchor=invite,
        has_compose_anchor_in_action_root=compose,
        has_edit_intro_anchor=edit,
        has_labeled_action_button=labeled_action,
        has_labeled_action_anchor=labeled_anchor,
        has_incoming_action_row=incoming_row,
    )


class TestConnectWithPerson:
    async def test_connectable_navigates_deeplink_and_verifies(self, mock_page):
        """Connect via deeplink: dialog opens, submit succeeds, sent list has it.

        The profile is read once. The header after a send is structurally
        identical to a follow-only profile, so nothing here re-reads it; the
        sent-invitations list is the only verification.
        """
        text = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text))
        _modal_at(mock_page, 0)

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(invite=True),
            ),
            patch.object(
                PageNavigator,
                "_navigate_to_page",
                new_callable=AsyncMock,
            ) as mock_nav,
            patch.object(
                actions,
                "_dialog_is_open",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_dialog_is_closed",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_dialog_text",
                new_callable=AsyncMock,
                return_value=DIALOG_TEXT,
            ),
            patch.object(
                actions,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_invitation_in_sent_list",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_sent,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connected"
        assert result["note_sent"] is False
        assert "sent invitations list" in result["message"]
        mock_sent.assert_awaited_once_with("testuser")
        mock_nav.assert_awaited_once_with(INVITE_URL)

    async def test_send_unverified_when_the_sent_list_cannot_be_read(self, mock_page):
        """The write happened; a read that raises must not hide it.

        The exception is the caller's only clue, so its text travels in the
        message, and ``note_sent`` keeps what the submit reported: nothing
        about a failed read says the note went nowhere.
        """
        text = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(invite=True),
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
                return_value=(True, True, None, DIALOG_TEXT),
            ),
            patch.object(
                actions,
                "_invitation_in_sent_list",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ) as mock_sent,
        ):
            result = await actions.connect_with_person("testuser", note="Hello")

        assert result["status"] == "send_unverified"
        assert result["note_sent"] is True
        assert "boom" in result["message"]
        mock_sent.assert_awaited_once_with("testuser")

    async def test_connectable_not_sent_when_absent_from_sent_list(self, mock_page):
        """Dialog submitted but the sent list has no row for the user → not_sent.

        Same flow as the connected case with one answer flipped, so the
        sent-list hit is what ``connected`` rests on. LinkedIn closes a
        weekly-limit dialog on its last button exactly like the invite
        dialog, which is why the dialog text travels in the message.
        """
        text = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text))
        _modal_at(mock_page, 0)

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(invite=True),
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions, "_dialog_is_closed", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_dialog_text",
                new_callable=AsyncMock,
                return_value=LIMIT_DIALOG_TEXT,
            ),
            patch.object(
                actions,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_invitation_in_sent_list",
                new_callable=AsyncMock,
                return_value=False,
            ) as mock_sent,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "not_sent"
        assert result["note_sent"] is False
        assert "does not appear in the sent invitations list" in result["message"]
        assert f"Dialog text: {LIMIT_DIALOG_TEXT}" in result["message"]
        mock_sent.assert_awaited_once_with("testuser")

    async def test_connectable_no_dialog_returns_dialog_not_found(self, mock_page):
        """Deeplink opened nothing → dialog_not_found naming where it landed.

        The message carries the deeplink, the landing URL, the selector and
        whatever live-region notice LinkedIn put on the page, so the caller
        can tell a refusal from a selector miss without a second run.
        """
        text = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text))
        landed = "https://www.linkedin.com/mynetwork/"
        mock_page.url = landed

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(invite=True),
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=False
            ),
            patch.object(
                actions,
                "_page_alerts",
                new_callable=AsyncMock,
                return_value="You have too many pending invitations",
            ),
            patch.object(
                actions, "_invitation_in_sent_list", new_callable=AsyncMock
            ) as mock_sent,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "dialog_not_found"
        assert INVITE_URL in result["message"]
        assert f"landed on {landed}" in result["message"]
        assert repr(_DIALOG_SELECTOR) in result["message"]
        assert "Page notice: You have too many pending invitations" in result["message"]
        mock_sent.assert_not_awaited()

    async def test_no_dialog_message_omits_notice_when_page_has_none(self, mock_page):
        text = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(invite=True),
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=False
            ),
            patch.object(
                actions, "_page_alerts", new_callable=AsyncMock, return_value=""
            ),
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "dialog_not_found"
        assert "Page notice" not in result["message"]

    async def test_unsubmittable_dialog_returns_connect_unavailable(self, mock_page):
        """A dialog opened but its primary button never landed → connect_unavailable.

        Distinct from no dialog at all: the text of what LinkedIn showed is
        the only clue to why, so it rides along in the message.
        """
        text = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(invite=True),
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
                return_value=(False, False, None, LIMIT_DIALOG_TEXT),
            ),
            patch.object(
                actions, "_invitation_in_sent_list", new_callable=AsyncMock
            ) as mock_sent,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connect_unavailable"
        assert f"Dialog text: {LIMIT_DIALOG_TEXT}" in result["message"]
        mock_sent.assert_not_awaited()

    async def test_returns_already_connected_via_anchor(self, mock_page):
        """1st-degree detected via /messaging/compose anchor."""
        text = "Collin\n\n· 1st\n\nEngineer\n\nMessage\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text))

        with patch.object(
            actions,
            "_read_action_signals",
            new_callable=AsyncMock,
            return_value=_signals(compose=True),
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "already_connected"

    async def test_returns_self_profile_via_edit_intro_anchor(self, mock_page):
        """Editing-your-own-profile anchor blocks connect attempts."""
        actions = _actions(mock_page, _reads("Daniel\n\nEdit profile\n"))

        with patch.object(
            actions,
            "_read_action_signals",
            new_callable=AsyncMock,
            return_value=_signals(edit=True),
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connect_unavailable"
        assert "own profile" in result["message"]

    async def test_connect_via_more_menu(self, mock_page):
        """Follow-primary profile with Connect under More: detection sees
        no invite anchor initially, _open_more_menu surfaces it, deeplink
        fires."""
        # Pre-More: Follow primary, Connect hidden under the More dropdown.
        pre = "Christian\n\n· 2nd\n\nFounder\n\nFollow\nMessage\nMore\n"
        actions = _actions(mock_page, _reads(pre))
        _modal_at(mock_page, 0)

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                # 1st: follow_only (compose+labeled, no invite).
                # 2nd: post-More reread reveals invite anchor.
                side_effect=[
                    _signals(compose=True, labeled_action=True),
                    _signals(invite=True, compose=True, labeled_action=True),
                ],
            ),
            patch.object(
                actions,
                "_open_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_open_more,
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                actions,
                "_dialog_is_open",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_dialog_is_closed",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_invitation_in_sent_list",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_sent,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connected"
        mock_open_more.assert_awaited_once()
        mock_sent.assert_awaited_once_with("testuser")
        # Deeplink fired exactly once.
        assert mock_nav.await_count == 1
        await_args = mock_nav.await_args
        assert await_args is not None
        assert "preload/custom-invite" in await_args.args[0]

    async def test_follow_only_after_more_does_not_send(self, mock_page):
        """Pending or genuinely follow-only profile: invite anchor never
        appears even after More-menu open. Critical write-gate guardrail —
        no deeplink fires, no connection request goes out."""
        text = "Public Figure\n\n· 3rd+\n\nCEO\n\nFollow\nMessage\nMore\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                # Both reads (initial + post-More) show no invite anchor.
                side_effect=[
                    _signals(compose=True, labeled_action=True),
                    _signals(compose=True, labeled_action=True),
                ],
            ),
            patch.object(
                actions,
                "_open_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_open_more,
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
                # A successful submit, so a gate that stopped holding reports
                # the deeplink it fired rather than crashing on the mock.
                return_value=(True, False, None, DIALOG_TEXT),
            ) as mock_submit,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connect_unavailable"
        assert result.get("note_sent") is False or "note_sent" not in result
        mock_open_more.assert_awaited_once()
        # Critical: deeplink must NOT fire and dialog must NOT be submitted.
        mock_nav.assert_not_awaited()
        mock_submit.assert_not_awaited()

    async def test_follow_only_with_note_reports_note_limit_from_deeplink_probe(
        self, mock_page
    ):
        """A requested note may reveal Premium quota without submitting."""
        text = "Public Figure\n\n· 3rd+\n\nCEO\n\nFollow\nMessage\nMore\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[
                    _signals(compose=True, labeled_action=True),
                    _signals(compose=True, labeled_action=True),
                ],
            ),
            patch.object(
                actions,
                "_open_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                actions,
                "_probe_invite_note_limit",
                new_callable=AsyncMock,
                return_value=PREMIUM_MESSAGE,
            ) as mock_probe,
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
                # A successful submit, so a gate that stopped holding reports
                # the deeplink it fired rather than crashing on the mock.
                return_value=(True, False, None, DIALOG_TEXT),
            ) as mock_submit,
        ):
            result = await actions.connect_with_person("testuser", note="Hello")

        assert result["status"] == "custom_note_limit_reached"
        assert result["message"] == PREMIUM_MESSAGE
        assert result["note_sent"] is False
        mock_nav.assert_awaited_once()
        mock_probe.assert_awaited_once()
        mock_submit.assert_not_awaited()

    async def test_more_menu_unavailable_does_not_send(self, mock_page):
        """Action root present but no More button (unusual but possible):
        _open_more_menu returns False, no retry, no deeplink fires."""
        text = "Public Figure\n\n· 3rd+\n\nCEO\n\nFollow\nMessage\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(compose=True, labeled_action=True),
            ),
            patch.object(
                actions,
                "_open_more_menu",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
                # A successful submit, so a gate that stopped holding reports
                # the deeplink it fired rather than crashing on the mock.
                return_value=(True, False, None, DIALOG_TEXT),
            ) as mock_submit,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "connect_unavailable"
        mock_nav.assert_not_awaited()
        mock_submit.assert_not_awaited()

    async def test_returns_pending(self, mock_page):
        """Profile with a pending invitation: detected via labeled <a> in
        the action root. Returns status='pending' without firing the
        deeplink (LinkedIn would only show 'already invited' anyway)."""
        text = "Frank\n\n· 3rd\n\nFounder\n\nMessage\nPending\nMore\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(compose=True, labeled_anchor=True),
            ),
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
                # A successful submit, so a gate that stopped holding reports
                # the deeplink it fired rather than crashing on the mock.
                return_value=(True, False, None, DIALOG_TEXT),
            ) as mock_submit,
            patch.object(
                actions, "_open_more_menu", new_callable=AsyncMock
            ) as mock_open_more,
            patch.object(
                actions, "_click_incoming_accept", new_callable=AsyncMock
            ) as mock_accept,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "pending"
        # No write-path side effects, and no action taken on the invitation
        # already out: withdrawing it is what the Pending control does.
        mock_nav.assert_not_awaited()
        mock_submit.assert_not_awaited()
        mock_open_more.assert_not_awaited()
        mock_accept.assert_not_awaited()

    async def test_returns_incoming_request_accepted(self, mock_page):
        """Structural detection + structural accept click, German locale."""
        pre = "Eric\n\n· 2.\n\nAachen\n\nAnnehmen\nIgnorieren\nMehr\nInfo\n"
        post = "Eric\n\n· 1.\n\nAachen\n\nNachricht\nMehr\nInfo\n"
        actions = _actions(mock_page, _reads(pre, post))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[
                    _signals(incoming_row=True),
                    _signals(compose=True),
                ],
            ),
            patch.object(
                actions,
                "_click_incoming_accept",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_accept,
            patch.object(
                PageNavigator,
                "_navigate_to_page",
                new_callable=AsyncMock,
            ) as mock_nav,
            patch.object(
                actions,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
            ) as mock_submit,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "accepted"
        mock_accept.assert_awaited_once()
        mock_nav.assert_not_awaited()
        mock_submit.assert_not_awaited()

    async def test_incoming_request_send_failed_when_click_fails(self, mock_page):
        """Structural accept click did not land; no locale-text guessing —
        report send_failed without navigating or clicking anything else.

        The owner holds no click-by-text helper to patch, so the claim is
        made against the page: a text fallback would have to build a
        locator, and nothing here builds one.
        """
        pre = "Eric\n\n· 2.\n\nAachen\n\nAnnehmen\nIgnorieren\nMehr\nInfo\n"
        actions = _actions(mock_page, _reads(pre))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(incoming_row=True),
            ),
            patch.object(
                actions,
                "_click_incoming_accept",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                PageNavigator,
                "_navigate_to_page",
                new_callable=AsyncMock,
            ) as mock_nav,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "send_failed"
        mock_nav.assert_not_awaited()
        mock_page.locator.assert_not_called()

    async def test_incoming_request_send_failed_when_no_first_degree(self, mock_page):
        """Accept clicked but profile never transitions to 1st-degree."""
        pre = "Eric\n\n· 2.\n\nAachen\n\nAnnehmen\nIgnorieren\nMehr\nInfo\n"
        actions = _actions(mock_page, _reads(pre, pre, pre))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(incoming_row=True),
            ),
            patch.object(
                actions,
                "_click_incoming_accept",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.connection_actions.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "send_failed"

    async def test_incoming_request_accepted_on_settle_retry(self, mock_page):
        """The first post-click read still renders the old top card;
        the settle retry sees the 1st-degree state and reports accepted."""
        pre = "Eric\n\n· 2.\n\nAachen\n\nAnnehmen\nIgnorieren\nMehr\nInfo\n"
        post = "Eric\n\n· 1.\n\nAachen\n\nNachricht\nMehr\nInfo\n"
        actions = _actions(mock_page, _reads(pre, pre, post))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[
                    _signals(incoming_row=True),
                    _signals(incoming_row=True),
                    _signals(compose=True),
                ],
            ),
            patch.object(
                actions,
                "_click_incoming_accept",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.connection_actions.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            result = await actions.connect_with_person("testuser")

        assert result["status"] == "accepted"
        mock_sleep.assert_awaited_once()

    async def test_returns_unavailable_when_no_signals_and_text(self, mock_page):
        """No structural signals, no actionable text → connect_unavailable."""
        text = "Public Figure\n\n· 3rd+\n\nCEO\n\nFollow\nMore\nAbout\n"
        actions = _actions(mock_page, _reads(text))

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=_signals(),
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=False
            ),
            patch.object(actions, "_dismiss_dialog", new_callable=AsyncMock),
        ):
            result = await actions.connect_with_person("testuser")

        # follow_only path goes through deeplink; no dialog opens → unavailable
        assert result["status"] == "connect_unavailable"

    async def test_returns_unavailable_on_empty_page(self, mock_page):
        actions = _actions(mock_page, _reads(""))

        result = await actions.connect_with_person("testuser")

        assert result["status"] == "unavailable"

    async def test_normalizes_before_its_own_downstream_use(self, mock_page):
        """The traversal case cannot see this one.

        The person workflow normalizes too, so removing this workflow's own
        call still raises on "../../feed". A full URL is what separates
        them: the read would succeed while the invite deeplink and the
        action-signal selectors kept receiving the URL where they expect
        the vanity.
        """
        read = _reads("text")
        actions = _actions(mock_page, read)
        seen: list[str] = []

        with (
            patch.object(
                actions,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=lambda username: seen.append(username) or _signals(),
            ),
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
        ):
            await actions.connect_with_person(
                "https://de.linkedin.com/in/williamhgates"
            )

        assert seen == ["williamhgates"]
        read.assert_awaited_once_with("williamhgates")


class TestInviteDialog:
    async def test_premium_upsell_message_reads_linkedin_dialog_text(self, mock_page):
        """Premium upsell detection returns LinkedIn's raw dialog text."""
        actions = _actions(mock_page)
        premium_link = MagicMock()
        premium_link.wait_for = AsyncMock(return_value=None)
        premium_link.is_visible = AsyncMock(return_value=True)
        premium_link.inner_text = AsyncMock(return_value="fallback")
        premium_link.first = premium_link
        mock_page.locator.return_value = premium_link
        mock_page.evaluate = AsyncMock(return_value=PREMIUM_MESSAGE)

        result = await actions._get_premium_upsell_message(timeout=1234)

        assert result == PREMIUM_MESSAGE
        mock_page.locator.assert_called_once_with(
            'dialog[open] a[href*="/premium/"], [role="dialog"] a[href*="/premium/"]'
        )
        premium_link.wait_for.assert_awaited_once_with(state="visible", timeout=1234)

    async def test_reports_premium_after_add_note(self, mock_page):
        """Add-note Premium upsell is a note-limit block, not no-dialog."""
        actions = _actions(mock_page)
        textarea = MagicMock()
        textarea.count = AsyncMock(return_value=0)
        textarea.first = textarea
        textarea.wait_for = AsyncMock(
            side_effect=PlaywrightTimeoutError("textarea timeout")
        )
        add_note_button = MagicMock()
        add_note_button.click = AsyncMock(return_value=None)
        buttons = MagicMock()
        buttons.count = AsyncMock(return_value=3)
        buttons.nth.return_value = add_note_button
        _scope_dialog(mock_page, buttons=buttons, textarea=textarea)

        with (
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_dialog_text",
                new_callable=AsyncMock,
                return_value=DIALOG_TEXT,
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=PREMIUM_MESSAGE,
            ) as mock_message,
            patch.object(
                actions, "_dismiss_dialog", new_callable=AsyncMock
            ) as mock_dismiss,
        ):
            result = await actions._submit_invite_dialog("Hello")

        assert result == (False, False, PREMIUM_MESSAGE, DIALOG_TEXT)
        buttons.nth.assert_called_once_with(1)
        add_note_button.click.assert_awaited_once()
        textarea.wait_for.assert_awaited_once_with(state="visible", timeout=3000)
        mock_message.assert_awaited_once()
        mock_dismiss.assert_awaited_once()

    async def test_reports_premium_after_send_click_failure(self, mock_page):
        """Premium upsell intercepting the Send click is a note-limit block.

        When LinkedIn swaps the invite dialog for the Premium upsell at the
        moment of submit, the original primary button is detached or pointer-
        event covered, so ``_click_dialog_primary_button`` and the keyboard
        fallback both fail. Without the post-click upsell probe the caller
        would dismiss the dialog and report ``connect_unavailable`` even
        though LinkedIn's raw quota message is sitting in the visible modal.
        """
        actions = _actions(mock_page)

        # Textarea already exposed so the reveal/fill branch succeeds and the
        # test focuses on the post-submit failure path.
        textarea = MagicMock()
        textarea.count = AsyncMock(return_value=1)
        textarea.first = textarea
        textarea.fill = AsyncMock()

        buttons = MagicMock()
        buttons.count = AsyncMock(return_value=2)
        primary_button = MagicMock()
        primary_button.focus = AsyncMock()
        buttons.nth.return_value = primary_button
        _scope_dialog(mock_page, buttons=buttons, textarea=textarea)
        mock_page.keyboard = MagicMock()
        mock_page.keyboard.press = AsyncMock()

        message = "You're out of free custom notes. Bypass the limit with Premium..."

        with (
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_dialog_is_closed",
                new_callable=AsyncMock,
                # Still open after the keyboard fallback, so sent stays False.
                return_value=False,
            ) as mock_closed,
            patch.object(
                actions,
                "_dialog_text",
                new_callable=AsyncMock,
                return_value=DIALOG_TEXT,
            ),
            patch.object(
                actions,
                "_fill_dialog_textarea",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=message,
            ) as mock_message,
            patch.object(
                actions, "_dismiss_dialog", new_callable=AsyncMock
            ) as mock_dismiss,
        ):
            result = await actions._submit_invite_dialog("Hello")

        assert result == (False, False, message, DIALOG_TEXT)
        buttons.nth.assert_called_once_with(1)
        primary_button.focus.assert_awaited_once()
        mock_page.keyboard.press.assert_awaited_once_with("Enter")
        mock_closed.assert_awaited_once_with(timeout=2000)
        mock_message.assert_awaited_once()
        mock_dismiss.assert_awaited_once()

    async def test_reports_premium_after_an_accepted_send_click(self, mock_page):
        """A Send click that succeeds is not yet a note that was delivered.

        LinkedIn accepts the click and then swaps the invite dialog for the
        quota upsell, so the only evidence that the note went nowhere is the
        modal standing open afterwards. Reporting this as a send would tell
        the caller a note reached a member who never got one, and the two
        earlier upsell probes cannot see it: both sit on failure paths.
        """
        actions = _actions(mock_page)

        textarea = MagicMock()
        textarea.count = AsyncMock(return_value=1)
        textarea.first = textarea
        textarea.fill = AsyncMock()
        _scope_dialog(mock_page, buttons=MagicMock(), textarea=textarea)

        with (
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions, "_dialog_is_closed", new_callable=AsyncMock
            ) as mock_closed,
            patch.object(
                actions,
                "_dialog_text",
                new_callable=AsyncMock,
                return_value=DIALOG_TEXT,
            ),
            patch.object(
                actions,
                "_fill_dialog_textarea",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=PREMIUM_MESSAGE,
            ) as mock_message,
            patch.object(
                actions, "_dismiss_dialog", new_callable=AsyncMock
            ) as mock_dismiss,
        ):
            result = await actions._submit_invite_dialog("Hello")

        assert result == (False, False, PREMIUM_MESSAGE, DIALOG_TEXT)
        mock_message.assert_awaited_once()
        mock_dismiss.assert_awaited_once()
        # The close wait belongs to the delivered path, which this is not.
        mock_closed.assert_not_awaited()

    async def test_the_quota_probe_never_clicks_the_primary_button(self, mock_page):
        """The probe opens the note editor and touches nothing else.

        It runs only on profiles the write gate already refused, so a click
        on the dialog's *last* button would send the invitation this workflow
        decided not to send. The index ``btn_count - 2`` is the whole of that
        guarantee, and nothing else held it: every case that drives the probe
        through ``connect_with_person`` replaces it wholesale.
        """
        actions = _actions(mock_page)
        clicks: list[int] = []

        def button_at(index: int):
            button = MagicMock()

            async def click(*_args, **_kwargs):
                clicks.append(index)

            button.click = AsyncMock(side_effect=click)
            return button

        # The legacy three-button invite dialog: dismiss, "Add a note", Send.
        buttons = MagicMock()
        buttons.count = AsyncMock(return_value=3)
        buttons.nth = MagicMock(side_effect=button_at)
        textarea = MagicMock()
        textarea.count = AsyncMock(return_value=0)
        textarea.first = textarea
        textarea.wait_for = AsyncMock()
        _scope_dialog(mock_page, buttons=buttons, textarea=textarea)

        with (
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                # Nothing before the note editor opens, the quota block after.
                side_effect=[None, PREMIUM_MESSAGE],
            ),
            patch.object(
                actions, "_dismiss_dialog", new_callable=AsyncMock
            ) as mock_dismiss,
        ):
            message = await actions._probe_invite_note_limit()

        assert message == PREMIUM_MESSAGE
        assert clicks == [1]
        textarea.wait_for.assert_awaited_once_with(state="visible", timeout=3000)
        mock_dismiss.assert_awaited_once()

    async def test_the_quota_probe_touches_nothing_without_a_modal(self, mock_page):
        """Open dialogs that are all chat windows are no dialog to probe.

        ``btn_count - 2`` on a chat window's controls is not "Add a note";
        the probe answers nothing and builds no locator at all.
        """
        actions = _actions(mock_page)
        button = MagicMock()
        button.click = AsyncMock()
        buttons = MagicMock()
        buttons.count = AsyncMock(return_value=3)
        buttons.nth = MagicMock(return_value=button)
        textarea = MagicMock()
        textarea.count = AsyncMock(return_value=0)
        _scope_dialog(mock_page, buttons=buttons, textarea=textarea)
        _modal_at(mock_page, -1)

        with (
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                actions, "_dismiss_dialog", new_callable=AsyncMock
            ) as mock_dismiss,
        ):
            assert await actions._probe_invite_note_limit() is None

        button.click.assert_not_awaited()
        mock_page.locator.assert_not_called()
        mock_dismiss.assert_not_awaited()

    async def test_submit_touches_nothing_without_a_modal(self, mock_page):
        """The open gate passed but no dialog is centred: nothing is clicked.

        The last button across all dialogs was a chat window's "Open send
        options" when this was measured, one index from its Send. A
        submit that fell back to any dialog would press it; this one reports
        no dialog, which the connect path turns into ``dialog_not_found``.
        """
        actions = _actions(mock_page)
        button = MagicMock()
        button.click = AsyncMock()
        button.focus = AsyncMock()
        buttons = MagicMock()
        buttons.count = AsyncMock(return_value=2)
        buttons.nth = MagicMock(return_value=button)
        textarea = MagicMock()
        textarea.count = AsyncMock(return_value=1)
        textarea.first = textarea
        textarea.fill = AsyncMock()
        _scope_dialog(mock_page, buttons=buttons, textarea=textarea)
        _modal_at(mock_page, -1)
        mock_page.keyboard = MagicMock()
        mock_page.keyboard.press = AsyncMock()

        with (
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(actions, "_dismiss_dialog", new_callable=AsyncMock),
        ):
            result = await actions._submit_invite_dialog("Hello")

        assert result == (False, False, None, None)
        button.click.assert_not_awaited()
        button.focus.assert_not_awaited()
        textarea.fill.assert_not_awaited()
        mock_page.keyboard.press.assert_not_awaited()
        mock_page.locator.assert_not_called()

    async def test_handles_two_button_gating_dialog(self, mock_page):
        """Two-button "Add a note to your invitation?" gating dialog (issue
        #455): nth(0) is "Add a note", nth(1) is "Send without a note".

        Asserts the secondary-button click that reveals the textarea fires
        even with btn_count == 2 (legacy guard required >= 3 and skipped
        the click, leaving the textarea unmounted)."""
        actions = _actions(mock_page)

        # Track each button click so we can assert the "Add a note" path
        # was taken to reveal the textarea.
        clicks: list[int] = []

        textarea_visible = {"value": False}

        # Two button locators inside the gating dialog: nth(0) "Add a
        # note" reveals the textarea, nth(1) "Send without a note".
        button_locators = [MagicMock(), MagicMock()]
        for idx, btn in enumerate(button_locators):

            def make_click(i: int):
                async def _click(*args, **kwargs):
                    clicks.append(i)
                    if i == 0:
                        textarea_visible["value"] = True
                    return None

                return _click

            btn.click = AsyncMock(side_effect=make_click(idx))
            btn.focus = AsyncMock()

        button_collection = MagicMock()
        button_collection.count = AsyncMock(return_value=2)
        button_collection.nth = MagicMock(side_effect=lambda i: button_locators[i])

        textarea_locator = MagicMock()
        textarea_locator.count = AsyncMock(
            side_effect=lambda: 1 if textarea_visible["value"] else 0
        )
        textarea_locator.first = textarea_locator
        textarea_locator.fill = AsyncMock()
        textarea_locator.wait_for = AsyncMock()

        _scope_dialog(mock_page, buttons=button_collection, textarea=textarea_locator)
        mock_page.keyboard = MagicMock()
        mock_page.keyboard.press = AsyncMock()

        with (
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                actions, "_dialog_is_closed", new_callable=AsyncMock, return_value=True
            ) as mock_closed,
            patch.object(
                actions,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                actions,
                "_dialog_text",
                new_callable=AsyncMock,
                return_value=DIALOG_TEXT,
            ),
        ):
            (
                submitted,
                note_sent,
                note_limit_message,
                dialog_text,
            ) = await actions._submit_invite_dialog("Hi from a test")

        assert submitted is True
        assert note_sent is True
        assert note_limit_message is None
        assert dialog_text == DIALOG_TEXT
        # Clicked "Add a note" (index 0) to reveal the textarea, then the
        # primary button (index 1) to send.
        assert clicks == [0, 1]
        textarea_locator.wait_for.assert_awaited_once_with(
            state="visible", timeout=3000
        )
        textarea_locator.fill.assert_awaited_once()
        # The delivered path waits for the modal to go, once, after the send.
        mock_closed.assert_awaited_once_with(timeout=5000)

    async def test_returns_no_dialog_text_when_no_dialog_opened(self, mock_page):
        """``None`` for the text, not ``""``: the caller splits on it.

        An empty string is a dialog that opened and said nothing; ``None``
        is no dialog, which the connect path reports as ``dialog_not_found``
        rather than ``connect_unavailable``.
        """
        actions = _actions(mock_page)

        with (
            patch.object(
                actions, "_dialog_is_open", new_callable=AsyncMock, return_value=False
            ),
            patch.object(actions, "_dialog_text", new_callable=AsyncMock) as mock_text,
        ):
            result = await actions._submit_invite_dialog(None)

        assert result == (False, False, None, None)
        mock_text.assert_not_awaited()

    async def test_dialog_text_reads_the_open_dialog_and_collapses_whitespace(
        self, mock_page
    ):
        actions = _actions(mock_page)
        dialog = MagicMock()
        dialog.inner_text = AsyncMock(
            return_value="  Invite Jane\n\nto connect \n Send without a note  "
        )
        dialogs = MagicMock(spec=["nth"])
        dialogs.nth = MagicMock(return_value=dialog)
        mock_page.locator.return_value = dialogs
        _modal_at(mock_page, 0)

        text = await actions._dialog_text()

        assert text == "Invite Jane to connect Send without a note"
        mock_page.locator.assert_called_once_with(_DIALOG_SELECTOR)
        dialogs.nth.assert_called_once_with(0)
        dialog.inner_text.assert_awaited_once_with(timeout=2000)

    async def test_dialog_text_is_empty_when_the_read_fails(self, mock_page):
        actions = _actions(mock_page)
        dialog = MagicMock()
        dialog.inner_text = AsyncMock(side_effect=PlaywrightTimeoutError("gone"))
        mock_page.locator.return_value.nth = MagicMock(return_value=dialog)
        _modal_at(mock_page, 0)

        assert await actions._dialog_text() == ""
        dialog.inner_text.assert_awaited_once()

    async def test_dialog_text_is_empty_when_no_modal_is_open(self, mock_page):
        """A chat window's text is not the invite dialog's."""
        actions = _actions(mock_page)
        _modal_at(mock_page, -1)

        assert await actions._dialog_text() == ""
        mock_page.locator.assert_not_called()


class TestInviteDialogScope:
    """Which dialog the controls are read from: the centred one, by index.

    Open messaging-overlay chat windows are ``role="dialog"`` too and outlive
    a navigation, so "the dialog" is never simply the first one, and with no
    centred one there is no dialog at all: a fallback to the first would
    hand out a chat window. The JS that picks the index runs only in
    ``tests/test_action_signals_dom.py``; here it is a mock and what is held
    is how its answer is spent.
    """

    async def test_picks_the_dialog_at_the_measured_index(self, mock_page):
        actions = _actions(mock_page)
        dialogs = mock_page.locator.return_value
        _modal_at(mock_page, 2)

        dialog = await actions._invite_dialog()

        assert dialog is dialogs.nth.return_value
        dialogs.nth.assert_called_once_with(2)
        mock_page.locator.assert_called_once_with(_DIALOG_SELECTOR)
        mock_page.evaluate.assert_awaited_once_with(_MODAL_DIALOG_INDEX_JS)

    async def test_is_none_when_no_dialog_is_centred(self, mock_page):
        actions = _actions(mock_page)
        _modal_at(mock_page, -1)

        assert await actions._invite_dialog() is None
        mock_page.locator.assert_not_called()

    async def test_is_none_when_the_script_raises(self, mock_page):
        actions = _actions(mock_page)
        mock_page.evaluate = AsyncMock(side_effect=RuntimeError("context destroyed"))

        assert await actions._invite_dialog() is None
        mock_page.locator.assert_not_called()

    async def test_the_controls_answer_nothing_without_a_modal(self, mock_page):
        """Click, fill and read all fail closed rather than reach a chat window."""
        actions = _actions(mock_page)
        _modal_at(mock_page, -1)

        assert await actions._click_dialog_primary_button() is False
        assert await actions._fill_dialog_textarea("Hello") is False
        mock_page.locator.assert_not_called()


class TestModalPolling:
    """The open/closed waits poll the modal index; nothing else is consulted.

    ``asyncio.sleep`` is a mock throughout, so a case that reaches it has
    polled once and would have waited; the count of them is the number of
    misses each case allows.
    """

    async def test_open_answers_at_once_when_the_modal_is_present(self, mock_page):
        actions = _actions(mock_page)

        with (
            patch.object(
                actions, "_modal_index", new_callable=AsyncMock, return_value=0
            ) as mock_index,
            _no_poll_sleep() as mock_sleep,
        ):
            assert await actions._dialog_is_open(timeout=60000) is True

        mock_index.assert_awaited_once()
        mock_sleep.assert_not_awaited()

    async def test_open_waits_through_misses_until_the_modal_appears(self, mock_page):
        actions = _actions(mock_page)

        with (
            patch.object(
                actions,
                "_modal_index",
                new_callable=AsyncMock,
                side_effect=[-1, -1, 0],
            ) as mock_index,
            _no_poll_sleep() as mock_sleep,
        ):
            assert await actions._dialog_is_open(timeout=60000) is True

        assert mock_index.await_count == 3
        assert mock_sleep.await_args_list == [((0.25,), {})] * 2

    async def test_open_gives_up_at_the_deadline(self, mock_page):
        """A zero budget is one read; a loop that ignored the deadline would
        reach the sleep, which here refuses."""
        actions = _actions(mock_page)

        with (
            patch.object(
                actions, "_modal_index", new_callable=AsyncMock, return_value=-1
            ) as mock_index,
            _no_poll_sleep() as mock_sleep,
        ):
            mock_sleep.side_effect = AssertionError("polled past the deadline")
            assert await actions._dialog_is_open(timeout=0) is False

        mock_index.assert_awaited_once()

    async def test_closed_waits_through_a_lingering_modal(self, mock_page):
        actions = _actions(mock_page)

        with (
            patch.object(
                actions,
                "_modal_index",
                new_callable=AsyncMock,
                side_effect=[0, 0, -1],
            ) as mock_index,
            _no_poll_sleep() as mock_sleep,
        ):
            assert await actions._dialog_is_closed(timeout=60000) is True

        assert mock_index.await_count == 3
        assert mock_sleep.await_count == 2

    async def test_closed_gives_up_at_the_deadline(self, mock_page):
        actions = _actions(mock_page)

        with (
            patch.object(
                actions, "_modal_index", new_callable=AsyncMock, return_value=0
            ) as mock_index,
            _no_poll_sleep() as mock_sleep,
        ):
            mock_sleep.side_effect = AssertionError("polled past the deadline")
            assert await actions._dialog_is_closed(timeout=0) is False

        mock_index.assert_awaited_once()

    async def test_index_is_minus_one_when_the_script_answers_nothing_usable(
        self, mock_page
    ):
        """The page double's default answer is a dict, which is no index."""
        actions = _actions(mock_page)

        assert await actions._modal_index() == -1

    async def test_dismiss_presses_escape_then_waits_for_the_modal_to_go(
        self, mock_page
    ):
        actions = _actions(mock_page)
        mock_page.keyboard = MagicMock()
        mock_page.keyboard.press = AsyncMock()

        with patch.object(
            actions, "_dialog_is_closed", new_callable=AsyncMock, return_value=False
        ) as mock_closed:
            await actions._dismiss_dialog()

        mock_page.keyboard.press.assert_awaited_once_with("Escape")
        mock_closed.assert_awaited_once_with(timeout=3000)


class TestInvitationInSentList:
    """The sent-invitations check: one navigation, a polled script, no text.

    ``asyncio.sleep`` is replaced throughout: the settle before navigating
    and the pause between polls are cadence, and the count of them is what
    each case pins.
    """

    SENT_URL = "https://www.linkedin.com/mynetwork/invitation-manager/sent/"
    ROWS_SELECTOR = 'main a[href*="/in/"]'

    def _rows(self, mock_page) -> MagicMock:
        rows = MagicMock()
        rows.first = rows
        rows.wait_for = AsyncMock()
        mock_page.locator.return_value = rows
        return rows

    def _no_sleep(self):
        return patch(
            "linkedin_mcp_server.scraping.connection_actions.asyncio.sleep",
            new_callable=AsyncMock,
        )

    async def test_navigates_to_the_sent_list_and_returns_the_script_answer(
        self, mock_page
    ):
        actions = _actions(mock_page)
        rows = self._rows(mock_page)
        mock_page.evaluate = AsyncMock(return_value=True)

        with (
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            self._no_sleep() as mock_sleep,
        ):
            found = await actions._invitation_in_sent_list("testuser")

        assert found is True
        mock_nav.assert_awaited_once_with(self.SENT_URL)
        mock_page.locator.assert_called_once_with(self.ROWS_SELECTOR)
        rows.wait_for.assert_awaited_once_with(timeout=10000)
        mock_page.evaluate.assert_awaited_once_with(_SENT_LIST_HAS_USER_JS, "testuser")
        # One settle before leaving the profile page, none after a hit.
        assert mock_sleep.await_count == 1

    async def test_returns_true_on_a_later_poll(self, mock_page):
        """The list renders late; the third read is the one that lands."""
        actions = _actions(mock_page)
        self._rows(mock_page)
        mock_page.evaluate = AsyncMock(side_effect=[False, False, True])

        with (
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            self._no_sleep() as mock_sleep,
        ):
            assert await actions._invitation_in_sent_list("testuser") is True

        assert mock_page.evaluate.await_count == 3
        # The settle, then one pause after each of the two misses.
        assert mock_sleep.await_count == 3

    async def test_returns_false_when_the_user_is_not_listed(self, mock_page):
        """Four misses are the whole budget; a fifth read never happens."""
        actions = _actions(mock_page)
        self._rows(mock_page)
        mock_page.evaluate = AsyncMock(return_value=False)

        with (
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            self._no_sleep() as mock_sleep,
        ):
            assert await actions._invitation_in_sent_list("testuser") is False

        assert mock_page.evaluate.await_count == 4
        assert mock_sleep.await_count == 5

    async def test_returns_false_when_the_script_raises(self, mock_page):
        """An unreadable list is not evidence the invite went out, and is
        not retried either."""
        actions = _actions(mock_page)
        self._rows(mock_page)
        mock_page.evaluate = AsyncMock(side_effect=RuntimeError("context destroyed"))

        with (
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            self._no_sleep(),
        ):
            assert await actions._invitation_in_sent_list("testuser") is False

        mock_page.evaluate.assert_awaited_once()

    async def test_still_evaluates_when_no_row_renders_in_time(self, mock_page):
        """An empty sent list is a real answer, so the wait is not a gate."""
        actions = _actions(mock_page)
        rows = self._rows(mock_page)
        rows.wait_for = AsyncMock(side_effect=PlaywrightTimeoutError("no rows"))
        mock_page.evaluate = AsyncMock(return_value=True)

        with (
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            self._no_sleep(),
        ):
            assert await actions._invitation_in_sent_list("testuser") is True

        rows.wait_for.assert_awaited_once_with(timeout=10000)
        mock_page.evaluate.assert_awaited_once()
