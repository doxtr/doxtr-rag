"""Domain schemas for doxtr-rag (CONTRACTS §C1, §C2, §C7).

Pydantic v2 models plus the ``AccessControl.most_restrictive`` resolver. These
models are pure data: they must not import Sphinx execution logic or database
clients (keeps producers decoupled from consumers).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from doxtr_rag.scrub import scrub, scrub_mapping_values


class AccessControl(str, Enum):
    """Scope classification for a chunk, node, mount, or xlink.

    ``LOCAL_PRIVATE`` is the fail-closed default: any unset, ``None``, or
    ambiguous scope resolves to it. ``SHARED_ORG`` is only ever the result of
    an affirmative, unambiguous decision (CONVENTIONS Cross-Cutting #1/#2).
    """

    SHARED_ORG = "SHARED_ORG"
    LOCAL_PRIVATE = "LOCAL_PRIVATE"

    @classmethod
    def coerce(cls, value: "AccessControl | str | None") -> "AccessControl":
        """Coerce an arbitrary value to a scope, failing closed.

        Anything that is not affirmatively ``SHARED_ORG`` becomes
        ``LOCAL_PRIVATE``.
        """
        if value is None:
            return cls.LOCAL_PRIVATE
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            return cls.LOCAL_PRIVATE

    @classmethod
    def most_restrictive(cls, *scopes: "AccessControl | str | None") -> "AccessControl":
        """Return the most restrictive scope across all inputs (fail-closed).

        Returns ``LOCAL_PRIVATE`` if ANY input is ``LOCAL_PRIVATE``, ``None``,
        or ambiguous. Only returns ``SHARED_ORG`` when EVERY input is
        affirmatively ``SHARED_ORG``. An empty input set fails closed to
        ``LOCAL_PRIVATE``.

        This is the ONLY place final scope is decided (CONVENTIONS §2); it is
        called exclusively by the ingestion engine (P5). Producers emit
        provisional scopes.
        """
        if not scopes:
            return cls.LOCAL_PRIVATE
        for scope in scopes:
            if cls.coerce(scope) is cls.LOCAL_PRIVATE:
                return cls.LOCAL_PRIVATE
        return cls.SHARED_ORG


# Metadata keys that must never appear in DocumentChunk.metadata. Matched
# case-insensitively as whole words / prefixes; this is a defense-in-depth
# guard on top of value-level scrubbing.
_CREDENTIAL_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(^|[_\-])(authorization|auth)($|[_\-])"),
    re.compile(r"(?i)token"),
    re.compile(r"(?i)secret"),
    re.compile(r"(?i)password|passwd|pwd"),
    re.compile(r"(?i)api[_\-]?key"),
    re.compile(r"(?i)cookie"),
    re.compile(r"(?i)credential"),
    re.compile(r"(?i)bearer"),
    re.compile(r"(?i)private[_\-]?key"),
    re.compile(r"(?i)session[_\-]?id"),
)

_REQUIRED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "source_file",
        "section_title",
        "breadcrumb",
        "access_scope",
        "content_hash",
        "provenance",
    }
)


def _is_credential_key(key: str) -> bool:
    return any(p.search(key) for p in _CREDENTIAL_KEY_PATTERNS)


class DocumentChunk(BaseModel):
    """The universal unit passed from producers (P2/P3/P4) to ingestion (P5).

    Attributes:
        id: Stable chunk identifier.
        text: The chunk body. Scrubbed at construction (value-level).
        embedding: The dense vector, populated by P5 (not producers).
        metadata: See required keys below; scrubbed and credential-guarded.

    Required ``metadata`` keys (producers MUST populate):
        - ``source_file`` (str)
        - ``section_title`` (str)
        - ``breadcrumb`` (list[str])
        - ``access_scope`` (:class:`AccessControl`, provisional; finalized by P5)
        - ``content_hash`` (str, sha256)
        - ``provenance`` (dict — one of ``{page}``/``{slide}``/``{cell}`` OR
          ``{docname, anchor}``)
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    text: str
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, meta: dict[str, Any]) -> dict[str, Any]:
        # 1. Reject credential-like metadata KEYS (defense-in-depth).
        offending = [k for k in meta if _is_credential_key(k)]
        if offending:
            raise ValueError(
                f"credential-like metadata keys are forbidden: {sorted(offending)}"
            )
        # 2. Require the contract keys.
        missing = _REQUIRED_METADATA_KEYS - set(meta)
        if missing:
            raise ValueError(f"missing required metadata keys: {sorted(missing)}")
        # 3. Normalize access_scope to the enum (fail-closed).
        meta = dict(meta)
        meta["access_scope"] = AccessControl.coerce(meta.get("access_scope"))
        # 4. Value-level scrub over ALL metadata values.
        scrubbed = scrub_mapping_values(meta)
        # scrub_mapping_values may stringify the enum; restore it.
        scrubbed["access_scope"] = meta["access_scope"]
        return scrubbed

    @model_validator(mode="after")
    def _scrub_text(self) -> "DocumentChunk":
        object.__setattr__(self, "text", scrub(self.text))
        return self

    @property
    def access_scope(self) -> AccessControl:
        """The (provisional until P5) access scope from metadata."""
        return AccessControl.coerce(self.metadata.get("access_scope"))

    # ------------------------------------------------------------------
    # Canonical store (de)serialization (CONTRACTS §C2)
    # ------------------------------------------------------------------
    # ChromaDB metadata accepts only flat scalar values (str/int/float/bool).
    # These two methods are the SINGLE authoritative mapping between the rich
    # DocumentChunk.metadata and the flat form persisted to Chroma. P5 (Python
    # writer) and P6 (TS reader) both mirror THIS shape — they must not invent
    # their own flatten/unflatten, which would drift across the language
    # boundary and break the P7 round-trip.
    #
    # Flat shape:
    #   source_file    -> str
    #   section_title  -> str
    #   breadcrumb     -> JSON-encoded list[str]  (key "breadcrumb")
    #   access_scope   -> str enum value          (key "access_scope")
    #   content_hash   -> str
    #   provenance     -> JSON-encoded dict        (key "provenance")
    def to_store_metadata(self) -> dict[str, str]:
        """Flatten metadata to Chroma-native scalars (scrubbed, deterministic)."""
        import json

        meta = self.metadata
        flat: dict[str, str] = {
            "source_file": scrub(str(meta.get("source_file", ""))),
            "section_title": scrub(str(meta.get("section_title", ""))),
            "breadcrumb": json.dumps(
                [scrub(str(b)) for b in meta.get("breadcrumb", [])],
                separators=(",", ":"),
            ),
            "access_scope": self.access_scope.value,
            "content_hash": str(meta.get("content_hash", "")),
            "provenance": json.dumps(
                scrub_mapping_values(dict(meta.get("provenance", {}))),
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
        return flat

    @classmethod
    def from_store_metadata(cls, flat: dict[str, Any]) -> dict[str, Any]:
        """Inverse of :meth:`to_store_metadata`: rebuild rich metadata."""
        import json

        def _load(value: Any, default: Any) -> Any:
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (ValueError, TypeError):
                    return default
            return value if value is not None else default

        return {
            "source_file": flat.get("source_file", ""),
            "section_title": flat.get("section_title", ""),
            "breadcrumb": _load(flat.get("breadcrumb"), []),
            "access_scope": AccessControl.coerce(flat.get("access_scope")),
            "content_hash": flat.get("content_hash", ""),
            "provenance": _load(flat.get("provenance"), {}),
        }


class SecurityAuditEvent(BaseModel):
    """An append-only audit record for restricted-xlink routing (written by P4).

    ``scrubbed_url`` MUST already have passed through :func:`scrub` (C3) before
    the event is constructed; the model re-scrubs defensively.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    resource_id: str
    scrubbed_url: str
    reason: str
    resolved_scope: AccessControl

    @field_validator("scrubbed_url", "reason", "resource_id")
    @classmethod
    def _scrub_str(cls, value: str) -> str:
        return scrub(value)
