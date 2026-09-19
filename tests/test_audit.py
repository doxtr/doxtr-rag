"""P4 tests for the audit logger (CONTRACTS §C7)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from doxtr_rag.audit import AuditLog
from doxtr_rag.schema import AccessControl


@pytest.fixture
def audit_path(tmp_path: Path) -> Path:
    return tmp_path / ".doxtr" / "security_audit.jsonl"


def test_log_created_with_0600(audit_path: Path) -> None:
    log = AuditLog(audit_path)
    mode = stat.S_IMODE(os.stat(log.path).st_mode)
    assert mode == 0o600


def test_record_appends_jsonl(audit_path: Path) -> None:
    log = AuditLog(audit_path)
    log.record(
        resource_id="JIRA-1",
        url="https://jira/i",
        reason="jira_ils_restricted",
        resolved_scope=AccessControl.LOCAL_PRIVATE,
    )
    log.record(
        resource_id="CONF-9",
        url="https://conf/i",
        reason="confluence_read_restricted",
        resolved_scope=AccessControl.LOCAL_PRIVATE,
    )
    events = log.read_events()
    assert len(events) == 2
    assert events[0].resource_id == "JIRA-1"
    assert events[1].resolved_scope is AccessControl.LOCAL_PRIVATE


def test_record_scrubs_credentials(audit_path: Path) -> None:
    log = AuditLog(audit_path)
    ev = log.record(
        resource_id="X",
        url="https://user:pw@jira/i?token=ghp_" + "a" * 20,
        reason="http_403",
        resolved_scope=AccessControl.LOCAL_PRIVATE,
    )
    assert "pw" not in ev.scrubbed_url
    assert "ghp_" not in ev.scrubbed_url
    # And nothing unscrubbed hits disk.
    raw = log.path.read_text()
    assert "ghp_" not in raw and "pw@" not in raw


def test_info_emitted_for_private(audit_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    log = AuditLog(audit_path)
    with caplog.at_level("INFO"):
        log.record(
            resource_id="JIRA-1",
            url="https://jira/i",
            reason="jira_ils_restricted",
            resolved_scope=AccessControl.LOCAL_PRIVATE,
        )
    assert any("LOCAL_PRIVATE" in r.message for r in caplog.records)


def test_no_info_for_shared(audit_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    log = AuditLog(audit_path)
    with caplog.at_level("INFO"):
        log.record(
            resource_id="OK-1",
            url="https://jira/i",
            reason="jira_unrestricted",
            resolved_scope=AccessControl.SHARED_ORG,
            emit_info=True,
        )
    assert not any("LOCAL_PRIVATE" in r.message for r in caplog.records)


def test_reopen_preserves_0600_and_appends(audit_path: Path) -> None:
    AuditLog(audit_path).record(
        resource_id="A",
        url="u",
        reason="r",
        resolved_scope=AccessControl.LOCAL_PRIVATE,
    )
    log2 = AuditLog(audit_path)
    assert stat.S_IMODE(os.stat(log2.path).st_mode) == 0o600
    log2.record(
        resource_id="B", url="u", reason="r", resolved_scope=AccessControl.LOCAL_PRIVATE
    )
    assert len(log2.read_events()) == 2


def test_read_events_empty_when_absent(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / ".doxtr" / "audit.jsonl")
    log.path.unlink()
    assert log.read_events() == []
