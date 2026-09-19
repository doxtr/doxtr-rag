"""F1 tests for schema models (CONTRACTS §C1, §C2, §C7)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from doxtr_rag.schema import AccessControl, DocumentChunk, SecurityAuditEvent


# -- AccessControl -----------------------------------------------------------
def test_most_restrictive_all_shared() -> None:
    assert (
        AccessControl.most_restrictive("SHARED_ORG", AccessControl.SHARED_ORG)
        is AccessControl.SHARED_ORG
    )


def test_most_restrictive_any_private_wins() -> None:
    assert (
        AccessControl.most_restrictive("SHARED_ORG", "LOCAL_PRIVATE")
        is AccessControl.LOCAL_PRIVATE
    )


def test_most_restrictive_none_fails_closed() -> None:
    assert AccessControl.most_restrictive("SHARED_ORG", None) is AccessControl.LOCAL_PRIVATE


def test_most_restrictive_ambiguous_fails_closed() -> None:
    assert AccessControl.most_restrictive("BOGUS") is AccessControl.LOCAL_PRIVATE


def test_most_restrictive_empty_fails_closed() -> None:
    assert AccessControl.most_restrictive() is AccessControl.LOCAL_PRIVATE


def test_coerce_variants() -> None:
    assert AccessControl.coerce(None) is AccessControl.LOCAL_PRIVATE
    assert AccessControl.coerce("SHARED_ORG") is AccessControl.SHARED_ORG
    assert AccessControl.coerce(AccessControl.SHARED_ORG) is AccessControl.SHARED_ORG
    assert AccessControl.coerce("nope") is AccessControl.LOCAL_PRIVATE


# -- DocumentChunk -----------------------------------------------------------
def _meta(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "source_file": "source/index.rst",
        "section_title": "Intro",
        "breadcrumb": ["Docs", "Intro"],
        "access_scope": "SHARED_ORG",
        "content_hash": "a" * 64,
        "provenance": {"docname": "index", "anchor": "intro"},
    }
    base.update(over)
    return base


def test_documentchunk_valid() -> None:
    chunk = DocumentChunk(id="c1", text="hello world", metadata=_meta())
    assert chunk.access_scope is AccessControl.SHARED_ORG
    assert chunk.embedding is None


def test_documentchunk_missing_required_key() -> None:
    meta = _meta()
    del meta["breadcrumb"]
    with pytest.raises(ValidationError):
        DocumentChunk(id="c1", text="x", metadata=meta)


def test_documentchunk_rejects_credential_keys() -> None:
    for bad in ("authorization", "api_key", "session_id", "password", "secret_token"):
        with pytest.raises(ValidationError):
            DocumentChunk(id="c1", text="x", metadata=_meta(**{bad: "v"}))


def test_documentchunk_scrubs_text() -> None:
    chunk = DocumentChunk(
        id="c1", text="see Bearer abcdef123456 for auth", metadata=_meta()
    )
    assert "abcdef123456" not in chunk.text


def test_documentchunk_scrubs_metadata_values() -> None:
    chunk = DocumentChunk(
        id="c1",
        text="ok",
        metadata=_meta(section_title="url https://u:pw@h/x"),
    )
    assert "pw" not in str(chunk.metadata["section_title"])


def test_documentchunk_ambiguous_scope_fails_closed() -> None:
    chunk = DocumentChunk(id="c1", text="x", metadata=_meta(access_scope="weird"))
    assert chunk.access_scope is AccessControl.LOCAL_PRIVATE


def test_documentchunk_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(id="c1", text="x", metadata=_meta(), unexpected=1)  # type: ignore[call-arg]


# -- SecurityAuditEvent ------------------------------------------------------
def test_audit_event_scrubs_url() -> None:
    ev = SecurityAuditEvent(
        timestamp="2026-01-01T00:00:00Z",
        resource_id="JIRA-1",
        scrubbed_url="https://u:pw@jira/rest",
        reason="restricted",
        resolved_scope=AccessControl.LOCAL_PRIVATE,
    )
    assert "pw" not in ev.scrubbed_url
    assert ev.resolved_scope is AccessControl.LOCAL_PRIVATE


# -- validate_assignment (guards re-run on reassignment) ---------------------
def test_documentchunk_reassignment_rescrubs_text() -> None:
    chunk = DocumentChunk(id="c1", text="ok", metadata=_meta())
    chunk.text = "leak Bearer abcdef123456"
    assert "abcdef123456" not in chunk.text


def test_documentchunk_reassignment_reguards_metadata() -> None:
    chunk = DocumentChunk(id="c1", text="ok", metadata=_meta())
    with pytest.raises(ValidationError):
        chunk.metadata = _meta(api_key="v")


# -- canonical store (de)serialization ---------------------------------------
def test_to_from_store_metadata_roundtrip() -> None:
    chunk = DocumentChunk(
        id="c1",
        text="body",
        metadata=_meta(
            breadcrumb=["Docs", "Intro"],
            provenance={"page": 3},
            access_scope="SHARED_ORG",
        ),
    )
    flat = chunk.to_store_metadata()
    # All values are Chroma-native scalars (strings here).
    assert all(isinstance(v, str) for v in flat.values())
    assert flat["access_scope"] == "SHARED_ORG"
    rich = DocumentChunk.from_store_metadata(flat)
    assert rich["breadcrumb"] == ["Docs", "Intro"]
    assert rich["provenance"] == {"page": 3}
    assert rich["access_scope"] is AccessControl.SHARED_ORG


def test_to_store_metadata_scrubs_values() -> None:
    chunk = DocumentChunk(
        id="c1",
        text="x",
        metadata=_meta(section_title="tok Bearer abcdef123456"),
    )
    flat = chunk.to_store_metadata()
    assert "abcdef123456" not in flat["section_title"]


def test_from_store_metadata_tolerates_bad_json() -> None:
    rich = DocumentChunk.from_store_metadata(
        {"breadcrumb": "not-json", "provenance": "{bad", "access_scope": "bogus"}
    )
    assert rich["breadcrumb"] == []
    assert rich["provenance"] == {}
    assert rich["access_scope"] is AccessControl.LOCAL_PRIVATE
