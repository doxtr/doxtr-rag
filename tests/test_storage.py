"""P5 tests for storage & ingestion (isolation, parity, writability, routing).

Uses ChromaDB EphemeralClient-backed stores with a fake 1024-dim embedding
provider so tests run without the pre-baked bge-m3 weights.
"""

from __future__ import annotations

import chromadb
import pytest

import uuid

from doxtr_rag.base_store import StoreAddress
from doxtr_rag.config import SphinxRAGConfig, StoreTarget
from doxtr_rag.schema import AccessControl, DocumentChunk
from doxtr_rag.storage import (
    ChromaVectorStore,
    IngestionEngine,
    IsolationViolationError,
)


class FakeProvider:
    dimension = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Deterministic tiny vectors; length must be 1024 for parity.
        return [[(len(t) % 7) / 10.0] * 1024 for t in texts]


def _chunk(cid: str, scope: str) -> DocumentChunk:
    return DocumentChunk(
        id=cid,
        text=f"body {cid}",
        metadata={
            "source_file": "source/a.rst",
            "section_title": "S",
            "breadcrumb": ["a"],
            "access_scope": scope,
            "content_hash": "h" * 64,
            "provenance": {"docname": "a", "anchor": "s"},
        },
    )


def _store(client: object, scope: AccessControl, database: str, collection: str,
           writable: bool = True) -> ChromaVectorStore:
    # Unique collection name per instance (EphemeralClient shares global state).
    unique = f"{collection}_{uuid.uuid4().hex[:8]}"
    target = StoreTarget(mode="embedded", path="/x", database=database, writable=writable)
    return ChromaVectorStore(
        target=target,
        address=StoreAddress("default_tenant", database, unique),
        provider=FakeProvider(),
        scope=scope,
        client=client,
    )


@pytest.fixture
def engine() -> IngestionEngine:
    client = chromadb.EphemeralClient()
    shared = _store(client, AccessControl.SHARED_ORG, "doxtr_shared", "shared_col")
    private = _store(client, AccessControl.LOCAL_PRIVATE, "doxtr_private", "priv_col")
    return IngestionEngine(
        SphinxRAGConfig.from_mapping({}),
        provider=FakeProvider(),
        shared_store=shared,
        private_store=private,
    )


# -- routing + reconciliation -------------------------------------------------
def test_routes_by_scope(engine: IngestionEngine) -> None:
    counts = engine.ingest([_chunk("c1", "SHARED_ORG"), _chunk("c2", "LOCAL_PRIVATE")])
    assert counts["SHARED_ORG"] == 1
    assert counts["LOCAL_PRIVATE"] == 1
    assert engine.shared_store.count() == 1  # type: ignore[union-attr]
    assert engine.private_store.count() == 1  # type: ignore[union-attr]


def test_extra_scope_forces_private(engine: IngestionEngine) -> None:
    # A SHARED_ORG chunk with an extra LOCAL_PRIVATE api scope -> private.
    counts = engine.ingest(
        [_chunk("c1", "SHARED_ORG")],
        extra_scopes={"c1": [AccessControl.LOCAL_PRIVATE]},
    )
    assert counts["LOCAL_PRIVATE"] == 1
    assert counts["SHARED_ORG"] == 0


def test_reconcile_scope_is_most_restrictive() -> None:
    assert IngestionEngine.reconcile_scope("SHARED_ORG", "SHARED_ORG") is AccessControl.SHARED_ORG
    assert IngestionEngine.reconcile_scope("SHARED_ORG", None) is AccessControl.LOCAL_PRIVATE


# -- isolation guard (defense-in-depth) --------------------------------------
def test_isolation_guard_blocks_private_into_shared(engine: IngestionEngine) -> None:
    with pytest.raises(IsolationViolationError):
        engine.shared_store.upsert([_chunk("evil", "LOCAL_PRIVATE")])  # type: ignore[union-attr]


def test_single_server_two_databases_isolation() -> None:
    # One client hosting both scopes as distinct databases/collections; the
    # guard still blocks cross-scope writes at the resolved boundary.
    client = chromadb.EphemeralClient()
    shared = _store(client, AccessControl.SHARED_ORG, "doxtr_shared", "col_shared")
    private = _store(client, AccessControl.LOCAL_PRIVATE, "doxtr_private", "col_private")
    eng = IngestionEngine(
        SphinxRAGConfig.from_mapping({}),
        provider=FakeProvider(),
        shared_store=shared,
        private_store=private,
    )
    eng.ingest([_chunk("s", "SHARED_ORG"), _chunk("p", "LOCAL_PRIVATE")])
    with pytest.raises(IsolationViolationError):
        shared.upsert([_chunk("leak", "LOCAL_PRIVATE")])


# -- writability probe (requirement #22) -------------------------------------
def test_readonly_shared_skips_gracefully() -> None:
    client = chromadb.EphemeralClient()
    shared = _store(client, AccessControl.SHARED_ORG, "doxtr_shared", "shared_col", writable=False)
    private = _store(client, AccessControl.LOCAL_PRIVATE, "doxtr_private", "priv_col")
    eng = IngestionEngine(
        SphinxRAGConfig.from_mapping({}),
        provider=FakeProvider(),
        shared_store=shared,
        private_store=private,
    )
    counts = eng.ingest([_chunk("c1", "SHARED_ORG")])
    assert counts["shared_skipped"] == 1
    assert counts["SHARED_ORG"] == 0
    assert shared.count() == 0
    # Still queryable (read path unaffected).
    assert shared.query([0.0] * 1024, 1) == []


# -- batching ----------------------------------------------------------------
def test_configurable_batch_size(engine: IngestionEngine) -> None:
    chunks = [_chunk(f"c{i}", "LOCAL_PRIVATE") for i in range(10)]
    engine._batch_size = 3  # noqa: SLF001
    counts = engine.ingest(chunks)
    assert counts["LOCAL_PRIVATE"] == 10
    assert engine.private_store.count() == 10  # type: ignore[union-attr]


# -- pruning -----------------------------------------------------------------
def test_prune_deletes_from_both(engine: IngestionEngine) -> None:
    engine.ingest([_chunk("c1", "SHARED_ORG"), _chunk("c2", "LOCAL_PRIVATE")])
    engine.prune(["c1", "c2"])
    assert engine.shared_store.count() == 0  # type: ignore[union-attr]
    assert engine.private_store.count() == 0  # type: ignore[union-attr]


def test_prune_empty_noop(engine: IngestionEngine) -> None:
    engine.prune([])  # must not raise


# -- embedding parity --------------------------------------------------------
def test_parity_dimension_mismatch_is_fatal() -> None:
    from doxtr_rag.embedding import assert_dimension

    with pytest.raises(ValueError):
        assert_dimension(768)


def test_query_roundtrip_inflates_metadata(engine: IngestionEngine) -> None:
    engine.ingest([_chunk("c2", "LOCAL_PRIVATE")])
    res = engine.private_store.query([0.0] * 1024, 5)  # type: ignore[union-attr]
    assert any(c.id == "c2" for c in res)
    c = next(c for c in res if c.id == "c2")
    assert c.access_scope is AccessControl.LOCAL_PRIVATE
    assert c.metadata["breadcrumb"] == ["a"]
    assert c.metadata["provenance"] == {"docname": "a", "anchor": "s"}


# -- client construction / open_stores ---------------------------------------
def test_split_host_port() -> None:
    from doxtr_rag.storage import _split_host_port

    assert _split_host_port("https://chroma.example.com:9000", True) == (
        "chroma.example.com",
        9000,
    )
    assert _split_host_port("http://localhost", False) == ("localhost", 8000)
    assert _split_host_port("https://h", True) == ("h", 443)


def test_build_embedded_client_and_open_stores(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = SphinxRAGConfig.from_mapping(
        {
            "rag_local_store_mode": "embedded",
            "rag_local_store_path": str(tmp_path / "priv"),
            "rag_shared_store_mode": "embedded",
            "rag_shared_store_uri": str(tmp_path / "shared"),
        }
    )
    eng = IngestionEngine(config, provider=FakeProvider())
    eng.open_stores()
    assert eng.shared_store is not None
    assert eng.private_store is not None
    # A real embedded round-trip through the resolved stores.
    counts = eng.ingest([_chunk("c1", "LOCAL_PRIVATE")])
    assert counts["LOCAL_PRIVATE"] == 1
    assert eng.private_store.count() == 1  # type: ignore[union-attr]


def test_build_http_client(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import doxtr_rag.storage as storage

    captured: dict[str, object] = {}

    class _Col:
        def count(self) -> int:
            return 0

    class _Client:
        def get_or_create_collection(self, name: str, metadata: object = None) -> _Col:
            return _Col()

    def fake_httpclient(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return _Client()

    import chromadb

    monkeypatch.setattr(chromadb, "HttpClient", fake_httpclient)
    target = StoreTarget(
        mode="http",
        uri="https://chroma.example.com:8443",
        database="doxtr_shared",
        tenant="default_tenant",
        ssl=True,
        headers={"Authorization": "Bearer realtoken"},
        writable=True,
    )
    store = ChromaVectorStore(
        target=target,
        address=StoreAddress("default_tenant", "doxtr_shared", "bge_m3_1024"),
        provider=FakeProvider(),
        scope=AccessControl.SHARED_ORG,
    )
    assert captured["host"] == "chroma.example.com"
    assert captured["port"] == 8443
    assert captured["ssl"] is True
    # The real auth header is passed through to the client (not scrubbed).
    assert captured["headers"]["Authorization"] == "Bearer realtoken"  # type: ignore[index]
    assert store.collection_name == "bge_m3_1024"
