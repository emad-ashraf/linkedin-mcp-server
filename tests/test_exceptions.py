from datetime import datetime, timezone

from linkedin_mcp_server.exceptions import (
    AccountCooldownError,
    CredentialsNotFoundError,
    LinkedInMCPError,
    SessionExpiredError,
)


def test_base_exception():
    err = LinkedInMCPError("test")
    assert str(err) == "test"


def test_session_expired_default_message():
    err = SessionExpiredError()
    assert "expired" in str(err).lower()


def test_session_expired_custom_message():
    err = SessionExpiredError("custom")
    assert str(err) == "custom"


def test_inheritance():
    assert issubclass(SessionExpiredError, LinkedInMCPError)
    assert issubclass(CredentialsNotFoundError, LinkedInMCPError)


def test_account_cooldown_names_the_type_and_the_resume_time():
    resume_at = datetime(2026, 9, 21, 10, 30, tzinfo=timezone.utc)
    err = AccountCooldownError(resume_at, reason="the hourly action cap is reached")

    assert err.error_type == "account_cooldown"
    assert err.resume_at == "2026-09-21T10:30:00+00:00"
    # Both fields are in the text too: a ToolError carries nothing else.
    assert "error_type=account_cooldown" in str(err)
    assert "resume_at=2026-09-21T10:30:00+00:00" in str(err)
    assert "hourly action cap" in str(err)
    assert "Do not call any LinkedIn-touching tool before that time" in str(err)
    assert issubclass(AccountCooldownError, LinkedInMCPError)
