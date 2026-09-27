"""Tests for ad-hoc content ingestion (``doxtr_rag.ingest``).

Covers URL-extension detection, the provenance chunk, URL fetching (httpx
mocked), logical-source rewriting, and the end-to-end ``ingest()`` flow into
both a canonical (private) store and an independent custom store. Uses an
on-disk embedded Chroma store with a fake 1024-dim provider so the tests run
without the pinned bge-m3 weights (mirroring ``tests/test_storage.py``).
"""

from __future__ import annotations

import os

import pytest

import doxtr_rag.embedding_provider as ep_mod
from doxtr_rag import ingest as ing
from doxtr_rag.config import SphinxRAGConfig
from doxtr_rag.schema import AccessControl


class FakeProvider:
    dimension = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[(len(t) % 7) / 10.0] * 1024 for t in texts]


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # ingest lazily imports EmbeddingProvider (custom-store path), while the
    # IngestionEngine (private/shared path) imports it at storage module top.
    # Patch both bindings so no test depends on the pinned bge-m3 weights.
    import doxtr_rag.storage as storage_mod

    monkeypatch.setattr(ep_mod, "EmbeddingProvider", FakeProvider)
    monkeypatch.setattr(storage_mod, "EmbeddingProvider", FakeProvider)


def _config(tmp_path: str) -> SphinxRAGConfig:
    return SphinxRAGConfig.from_mapping(
        {
            "rag_local_store_mode": "embedded",
            "rag_local_store_path": os.path.join(tmp_path, "local_chroma"),
            "rag_shared_store_mode": "embedded",
            "rag_shared_store_uri": os.path.join(tmp_path, "shared_chroma"),
        }
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_url_extension_detection() -> None:
    assert ing._url_extension("https://x/rfc8693.pdf") == ".pdf"
    assert ing._url_extension("https://x/page.md?ref=1#frag") == ".md"
    assert ing._url_extension("https://x/some/page/") == ".html"


def test_provenance_chunk_records_url_and_tags() -> None:
    c = ing._provenance_chunk(
        source_id="https://example.com/x",
        scope=AccessControl.LOCAL_PRIVATE,
        source_url="https://example.com/x",
        tags="a, b",
        title="Example",
    )
    assert c is not None
    assert "Title: Example" in c.text
    assert "Source URL: https://example.com/x" in c.text
    assert "Tags: a, b" in c.text


def test_provenance_chunk_none_without_url_or_tags() -> None:
    assert (
        ing._provenance_chunk(
            source_id="x",
            scope=AccessControl.LOCAL_PRIVATE,
            source_url=None,
            tags=None,
            title=None,
        )
        is None
    )


def test_custom_store_addressing(tmp_path) -> None:
    cfg = _config(str(tmp_path))
    vs = ing._custom_store(cfg, "my_store")
    assert vs._address.database == "my_store"
    assert vs._address.collection.endswith("_1024")


def test_extract_path_rewrites_logical_source(tmp_path) -> None:
    p = os.path.join(str(tmp_path), "doc.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("# Title\n\nBody text.\n")
    cfg = _config(str(tmp_path))
    chunks = ing._extract_path(
        p, AccessControl.LOCAL_PRIVATE, cfg, logical_source="https://x/doc.md"
    )
    assert chunks
    assert all(c.id.startswith("https://x/doc.md") for c in chunks)
    assert all(c.metadata["source_file"] == "https://x/doc.md" for c in chunks)


# ---------------------------------------------------------------------------
# URL fetch (httpx mocked)
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, content: bytes, ctype: str) -> None:
        self.content = content
        self.headers = {"content-type": ctype}
        self.is_redirect = False
        self.next_request = None

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, *a, **k) -> None:
        self._verify = k.get("verify")

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *a) -> None:
        return None

    def get(self, url: str) -> _FakeResp:
        return _FakeResp(b"# Fetched\n\nRemote body.\n", "text/markdown")


def test_fetch_url_refines_extension_from_content_type(monkeypatch) -> None:
    import httpx

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    # Neutralise the SSRF guard: this test exercises extension refinement, not
    # networking, and the fake host does not resolve.
    monkeypatch.setattr(ing, "_assert_public_url", lambda _u: None)
    body, ext = ing._fetch_url("https://x/no-hint")
    assert body.startswith(b"# Fetched")
    # URL had no extension hint; content-type "markdown" refines to .md.
    assert ext == ".md"


def test_fetch_url_rejects_loopback_ssrf(monkeypatch) -> None:
    # A URL resolving to loopback must fail closed (SSRF-via-redirect guard),
    # unless the discouraged insecure override is set.
    monkeypatch.delenv("DOXTR_RAG_INGEST_INSECURE", raising=False)
    with pytest.raises(ValueError, match="non-public|resolve"):
        ing._fetch_url("http://127.0.0.1:8787/x")
    with pytest.raises(ValueError, match="non-public|resolve"):
        ing._fetch_url("http://169.254.169.254/latest/meta-data/")


def test_fetch_url_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError, match="non-http"):
        ing._assert_public_url("file:///etc/passwd")


def test_fetch_url_insecure_override_bypasses_ssrf_guard(monkeypatch) -> None:
    # The insecure escape hatch disables the guard (documented last resort).
    monkeypatch.setenv("DOXTR_RAG_INGEST_INSECURE", "1")
    # Should not raise even for a loopback URL.
    ing._assert_public_url("http://127.0.0.1:8787/x")


# ---------------------------------------------------------------------------
# End-to-end ingest()
# ---------------------------------------------------------------------------
def test_ingest_requires_exactly_one_source(tmp_path) -> None:
    cfg = _config(str(tmp_path))
    with pytest.raises(ValueError):
        ing.ingest(config=cfg)
    with pytest.raises(ValueError):
        ing.ingest(config=cfg, file="a", url="b")


def test_ingest_missing_file_raises(tmp_path) -> None:
    cfg = _config(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        ing.ingest(config=cfg, file=str(tmp_path / "nope.md"))


def test_ingest_file_into_private_store(tmp_path) -> None:
    p = os.path.join(str(tmp_path), "doc.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("# Title\n\nBody text.\n")
    cfg = _config(str(tmp_path))
    counts = ing.ingest(config=cfg, file=p, scope="private")
    assert counts["ingested"] >= 1


def test_ingest_file_into_custom_store_with_provenance(tmp_path) -> None:
    p = os.path.join(str(tmp_path), "doc.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("# Title\n\nBody text.\n")
    cfg = _config(str(tmp_path))
    counts = ing.ingest(
        config=cfg,
        file=p,
        store="custom_kb",
        source_url="https://x/doc",
        tags="k1",
        title="Doc",
    )
    assert counts["store"] == "custom_kb"
    # +1 for the prepended provenance chunk.
    assert counts["ingested"] >= 2


def test_ingest_url_into_private_store(tmp_path, monkeypatch) -> None:
    import httpx

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    monkeypatch.setattr(ing, "_assert_public_url", lambda _u: None)
    cfg = _config(str(tmp_path))
    counts = ing.ingest(config=cfg, url="https://x/page.md", scope="private")
    assert counts["ingested"] >= 1
