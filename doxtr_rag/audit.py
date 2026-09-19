"""Audit logger for restricted-xlink routing (P4 — CONTRACTS §C7).

Appends :class:`~doxtr_rag.schema.SecurityAuditEvent` records to
``.doxtr/security_audit.jsonl`` (JSON Lines, append-safe, ``0600`` perms) and
emits a Sphinx ``[INFO]`` per redirected restricted resource. All URLs are
scrubbed via the shared :func:`~doxtr_rag.scrub.scrub` before writing/logging.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from doxtr_rag.schema import AccessControl, SecurityAuditEvent
from doxtr_rag.scrub import scrub

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLog:
    """Append-only, ``0600`` JSON Lines audit log for restricted xlinks."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Create with 0600 up front so no window exists with looser perms.
        if not self._path.exists():
            fd = os.open(str(self._path), os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        else:
            try:
                os.chmod(self._path, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:  # pragma: no cover - platform dependent
                pass

    def record(
        self,
        *,
        resource_id: str,
        url: str,
        reason: str,
        resolved_scope: AccessControl,
        emit_info: bool = True,
        timestamp: Optional[str] = None,
    ) -> SecurityAuditEvent:
        """Append one audit event and (optionally) emit a Sphinx ``[INFO]``.

        The URL is scrubbed both by this method and again inside
        :class:`SecurityAuditEvent` (defense-in-depth).
        """
        event = SecurityAuditEvent(
            timestamp=timestamp or _utc_now_iso(),
            resource_id=resource_id,
            scrubbed_url=scrub(url),
            reason=reason,
            resolved_scope=resolved_scope,
        )
        line = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
        # Atomic append: single write() call under append mode.
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if emit_info and resolved_scope is AccessControl.LOCAL_PRIVATE:
            logger.info(
                "[doxtr-rag] restricted xlink %s routed to LOCAL_PRIVATE (%s): %s",
                resource_id,
                reason,
                event.scrubbed_url,
            )
        return event

    @property
    def path(self) -> Path:
        return self._path

    def read_events(self) -> list[SecurityAuditEvent]:
        """Read back all recorded events (for tests / inspection)."""
        if not self._path.exists():
            return []
        events: list[SecurityAuditEvent] = []
        for raw in self._path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                events.append(SecurityAuditEvent.model_validate_json(raw))
        return events
