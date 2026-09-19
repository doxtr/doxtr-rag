"""doxtr-rag Test Harness — reusable assertion helpers.

Mirrors the role of the reference project's ``assertions.py`` but targets RAG
build artifacts (scope routing, audit log, cache DB) rather than LaTeX markers.
These helpers are imported by ``test_runner.py`` and by scenario tests.
"""

from __future__ import annotations

import json
from pathlib import Path


def assert_no_private_in_shared(shared_scopes: list[str]) -> None:
    """Assert that no ``LOCAL_PRIVATE`` chunk was routed to the shared store."""
    leaked = [s for s in shared_scopes if s != "SHARED_ORG"]
    assert not leaked, f"LOCAL_PRIVATE leaked into SHARED_ORG target: {leaked}"


def assert_audit_scrubbed(audit_path: Path) -> None:
    """Assert the audit JSONL contains no obvious credential material."""
    if not audit_path.exists():
        return
    text = audit_path.read_text(encoding="utf-8")
    for needle in ("Bearer ", "ghp_", "AKIA", "xoxb-"):
        assert needle not in text, f"unscrubbed credential marker in audit: {needle}"


def read_audit_events(audit_path: Path) -> list[dict[str, object]]:
    """Return parsed JSON Lines audit events (empty list if absent)."""
    if not audit_path.exists():
        return []
    events: list[dict[str, object]] = []
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events
