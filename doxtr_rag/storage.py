"""Scoped ChromaDB store & ingestion engine (P5 — Wave C integration).

Dual-backend ChromaDB manager. Each store (SHARED_ORG / LOCAL_PRIVATE) is
resolved from its :class:`~doxtr_rag.config.StoreTarget` and may run
``embedded`` (on-disk ``PersistentClient``) or ``http`` (external server),
addressed by the canonical ``(tenant, database, collection)`` from
:mod:`doxtr_rag.base_store` (C4.1).

Key guarantees:
- **Final scope reconciliation** via ``AccessControl.most_restrictive`` is the
  ONLY place final scope is decided (CONVENTIONS §2).
- **Physical store isolation** (CONVENTIONS §4): a defense-in-depth guard inside
  the shared-store adapter raises on any non-``SHARED_ORG`` chunk, checked
  against the resolved target database — so it holds even when one external
  server hosts both scopes as separate databases.
- **Embedding parity** (CONVENTIONS §5): both stores use the same bge-m3/1024
  provider; a dimension mismatch is fatal.
- **Writability probe** (requirement #22): a read-only shared store skips
  ingestion gracefully (``[INFO]``) but stays queryable.
- **Cache-aware** upserts + deletion pruning against P2's cache (C8), in
  configurable batches.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from doxtr_rag.base_store import (
    BaseVectorStore,
    StoreAddress,
    address_from_config,
)
from doxtr_rag.config import SphinxRAGConfig, StoreTarget
from doxtr_rag.embedding_provider import EmbeddingProvider
from doxtr_rag.schema import AccessControl, DocumentChunk

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100


class IsolationViolationError(RuntimeError):
    """Raised when a non-SHARED_ORG chunk is routed to the shared store."""


class _EmptyCollection:
    """A no-op, read-only collection handle for an absent read-only store.

    Never writes; queries/count return empty. Lets a read-only shared store
    stay queryable without ever creating a collection (requirement #22).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def count(self) -> int:
        return 0

    def query(self, **_kwargs: Any) -> dict[str, Any]:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "embeddings": [[]]}

    def upsert(self, **_kwargs: Any) -> None:  # pragma: no cover - never called (writable=False gates upsert)
        raise IsolationViolationError("cannot write a read-only collection")

    def delete(self, **_kwargs: Any) -> None:  # pragma: no cover
        return None


class ChromaVectorStore(BaseVectorStore):
    """A concrete ChromaDB-backed store for one scope.

    Args:
        target: The resolved store target (mode/uri/path/tenant/database/...).
        address: The canonical ``(tenant, database, collection)`` for the scope.
        provider: The shared embedding provider (parity across stores).
        scope: The scope this store serves; the shared store enforces the
            isolation guard.
        client: An optional pre-built Chroma client (injected in tests).
    """

    def __init__(
        self,
        *,
        target: StoreTarget,
        address: StoreAddress,
        provider: EmbeddingProvider,
        scope: AccessControl,
        client: Optional[Any] = None,
    ) -> None:
        self._target = target
        self._address = address
        self._provider = provider
        self._scope = scope
        self._client = client if client is not None else _build_client(target, address)
        self._collection = self._open_collection(scope)

    def _open_collection(self, scope: AccessControl) -> Any:
        """Open the scope's collection.

        A read-only store must NOT be written to at construction: try
        ``get_collection`` first and only ``get_or_create_collection`` when the
        store is writable (requirement #22 — never write a read-only shared
        store). If a read-only store's collection is absent, expose an empty
        queryable collection view by creating a transient handle without
        writing (best-effort get_or_create is skipped).
        """
        meta = {"hnsw:space": "cosine", "doxtr_rag_scope": scope.value}
        if self._target.writable:
            return self._client.get_or_create_collection(
                name=self._address.collection, metadata=meta
            )
        try:
            return self._client.get_collection(name=self._address.collection)
        except Exception:  # noqa: BLE001 - read-only + absent collection
            # Do not write; return a lazy handle. Queries yield nothing.
            logger.info(
                "[doxtr-rag] read-only store %s has no collection %s yet; "
                "queries will return empty until it is populated elsewhere",
                self._address.database,
                self._address.collection,
            )
            return _EmptyCollection(self._address.collection)

    @property
    def collection_name(self) -> str:
        return self._address.collection

    @property
    def embedding_dimension(self) -> int:
        return self._provider.dimension

    @property
    def writable(self) -> bool:
        return self._target.writable

    def _guard(self, chunks: list[DocumentChunk]) -> None:
        """Defense-in-depth: the SHARED_ORG store rejects any non-shared chunk.

        Checked against the resolved target database so it holds even when one
        external server hosts both scopes.
        """
        if self._scope is not AccessControl.SHARED_ORG:
            return
        for c in chunks:
            if c.access_scope is not AccessControl.SHARED_ORG:
                raise IsolationViolationError(
                    f"refusing to write non-SHARED_ORG chunk {c.id!r} "
                    f"(scope={c.access_scope.value}) to shared database "
                    f"{self._address.database!r}"
                )

    def upsert(self, chunks: list[DocumentChunk], batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        if not chunks:
            return
        self._guard(chunks)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.text for c in batch]
            embeddings = [
                c.embedding if c.embedding is not None else emb
                for c, emb in zip(batch, self._provider.embed(texts))
            ]
            self._collection.upsert(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c.to_store_metadata() for c in batch],
            )

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)

    def query(self, embedding: list[float], top_k: int) -> list[DocumentChunk]:
        res = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        return _results_to_chunks(res)

    def count(self) -> int:
        return int(self._collection.count())


def _via_loopback(target: StoreTarget, loopback_uri: Optional[str]) -> StoreTarget:
    """Return an http StoreTarget pointing at ``loopback_uri`` for an embedded
    store, preserving tenant/database/writable; otherwise return ``target``.
    """
    if not loopback_uri or target.mode != "embedded":
        return target
    return StoreTarget(
        mode="http",
        uri=loopback_uri,
        path=None,
        tenant=target.tenant,
        database=target.database,
        ssl=False,
        headers={},
        writable=target.writable,
    )


def _build_client(target: StoreTarget, address: StoreAddress) -> Any:
    """Build a ChromaDB client from a resolved target (embedded or http)."""
    import chromadb

    if target.mode == "http":
        # Parse host/port from the URI; secrets are in target.headers (env).
        host, port = _split_host_port(target.uri or "", target.ssl)
        _ensure_http_tenant_database(
            host, port, target.ssl, target.headers, target.tenant, target.database
        )
        return chromadb.HttpClient(
            host=host,
            port=port,
            ssl=target.ssl,
            headers=target.headers,
            tenant=target.tenant,
            database=target.database,
        )
    # embedded
    _ensure_tenant_database(target.path or ".doxtr/chroma", target.tenant, target.database)
    return chromadb.PersistentClient(
        path=target.path or ".doxtr/chroma",
        tenant=target.tenant,
        database=target.database,
    )


def _ensure_http_tenant_database(
    host: str, port: int, ssl: bool, headers: dict[str, str], tenant: str, database: str
) -> None:
    """Best-effort create of (tenant, database) on an HTTP Chroma server.

    Needed for the loopback server (starts empty) and for external servers that
    have not pre-created the doxtr databases. Non-fatal on error.
    """
    import chromadb

    try:
        admin = chromadb.AdminClient(
            chromadb.config.Settings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
                chroma_server_host=host,
                chroma_server_http_port=port,
                chroma_server_ssl_enabled=ssl,
            )
        )
        try:
            admin.get_tenant(tenant)
        except Exception:  # noqa: BLE001
            admin.create_tenant(tenant)
        try:
            admin.get_database(database, tenant=tenant)
        except Exception:  # noqa: BLE001
            admin.create_database(database, tenant=tenant)
    except Exception as exc:  # noqa: BLE001 - non-fatal
        logger.debug("doxtr-rag: http tenant/database ensure skipped: %s", exc)


def _ensure_tenant_database(path: str, tenant: str, database: str) -> None:
    """Best-effort create of the (tenant, database) on an embedded store."""
    import chromadb
    from chromadb.config import Settings

    try:
        admin = chromadb.AdminClient(
            Settings(is_persistent=True, persist_directory=path)
        )
        try:
            admin.get_tenant(tenant)
        except Exception:  # noqa: BLE001
            admin.create_tenant(tenant)
        try:
            admin.get_database(database, tenant=tenant)
        except Exception:  # noqa: BLE001
            admin.create_database(database, tenant=tenant)
    except Exception as exc:  # noqa: BLE001 - non-fatal; client may still work
        logger.debug("doxtr-rag: tenant/database ensure skipped: %s", exc)


def _split_host_port(uri: str, ssl: bool) -> tuple[str, int]:
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if ssl else 8000)
    return host, port


def _results_to_chunks(res: dict[str, Any]) -> list[DocumentChunk]:
    """Convert a Chroma query/get result into DocumentChunks (metadata inflated)."""
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    embs = (res.get("embeddings") or [[None] * len(ids)])
    embs0 = embs[0] if embs else [None] * len(ids)
    chunks: list[DocumentChunk] = []
    for i, cid in enumerate(ids):
        flat = metas[i] if i < len(metas) and metas[i] else {}
        rich = DocumentChunk.from_store_metadata(flat)
        emb = embs0[i] if i < len(embs0) else None
        chunks.append(
            DocumentChunk(
                id=cid,
                text=docs[i] if i < len(docs) else "",
                embedding=list(map(float, emb)) if emb is not None else None,
                metadata=rich,
            )
        )
    return chunks


class IngestionEngine:
    """Coordinates dual-store ingestion with final scope reconciliation.

    Buffers provisional chunks from producers (P2/P3/P4), finalizes their scope
    via ``most_restrictive``, and upserts each into the correct store
    addressed by C4.1. The shared store is skipped gracefully when read-only.
    """

    def __init__(
        self,
        config: SphinxRAGConfig,
        *,
        provider: Optional[EmbeddingProvider] = None,
        shared_store: Optional[ChromaVectorStore] = None,
        private_store: Optional[ChromaVectorStore] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._config = config
        self._provider = provider or EmbeddingProvider()
        self._batch_size = batch_size
        self._shared = shared_store
        self._private = private_store

    # -- store lifecycle ------------------------------------------------
    def open_stores(
        self,
        *,
        shared_loopback_uri: Optional[str] = None,
        private_loopback_uri: Optional[str] = None,
    ) -> None:
        """Resolve targets and open both stores (idempotent).

        When a loopback URI is supplied for an embedded store, the store is
        opened as an ``http`` client against that loopback endpoint so a single
        process (the loopback server) owns the on-disk store — avoiding
        PersistentClient/server contention. The ``writable`` flag from the
        original probe is preserved.
        """
        if self._shared is None:
            target = _via_loopback(self._config.shared_target(), shared_loopback_uri)
            self._shared = ChromaVectorStore(
                target=target,
                address=address_from_config(AccessControl.SHARED_ORG, self._config),
                provider=self._provider,
                scope=AccessControl.SHARED_ORG,
            )
        if self._private is None:
            target = _via_loopback(self._config.local_target(), private_loopback_uri)
            self._private = ChromaVectorStore(
                target=target,
                address=address_from_config(AccessControl.LOCAL_PRIVATE, self._config),
                provider=self._provider,
                scope=AccessControl.LOCAL_PRIVATE,
            )

    # -- scope reconciliation -------------------------------------------
    @staticmethod
    def reconcile_scope(*provisional: AccessControl | str | None) -> AccessControl:
        """Finalize scope (the sole caller of ``most_restrictive``)."""
        return AccessControl.most_restrictive(*provisional)

    def _finalize(self, chunk: DocumentChunk, *extra_scopes: AccessControl | str | None) -> DocumentChunk:
        final = self.reconcile_scope(chunk.access_scope, *extra_scopes)
        meta = dict(chunk.metadata)
        meta["access_scope"] = final
        return DocumentChunk(
            id=chunk.id, text=chunk.text, embedding=chunk.embedding, metadata=meta
        )

    # -- ingestion ------------------------------------------------------
    def ingest(
        self,
        chunks: list[DocumentChunk],
        *,
        extra_scopes: Optional[dict[str, list[AccessControl | str | None]]] = None,
    ) -> dict[str, int]:
        """Route finalized chunks to the correct store; return per-scope counts.

        ``extra_scopes`` optionally maps chunk id -> additional provisional
        scopes (e.g. an xlink API scope) to fold into ``most_restrictive``.
        """
        if self._shared is None or self._private is None:
            self.open_stores()
        assert self._shared is not None and self._private is not None

        extra = extra_scopes or {}
        shared_batch: list[DocumentChunk] = []
        private_batch: list[DocumentChunk] = []
        for chunk in chunks:
            final = self._finalize(chunk, *extra.get(chunk.id, []))
            if final.access_scope is AccessControl.SHARED_ORG:
                shared_batch.append(final)
            else:
                private_batch.append(final)

        counts = {"SHARED_ORG": 0, "LOCAL_PRIVATE": 0, "shared_skipped": 0}
        # Private store: normally writable.
        if private_batch:
            self._private.upsert(private_batch, self._batch_size)
            counts["LOCAL_PRIVATE"] = len(private_batch)
        # Shared store: skip gracefully if read-only (requirement #22).
        if shared_batch:
            if self._shared.writable:
                self._shared.upsert(shared_batch, self._batch_size)
                counts["SHARED_ORG"] = len(shared_batch)
            else:
                counts["shared_skipped"] = len(shared_batch)
                logger.info(
                    "[doxtr-rag] shared store is read-only; skipping ingestion of "
                    "%d SHARED_ORG chunks (still queryable)",
                    len(shared_batch),
                )
        return counts

    def prune(self, chunk_ids: list[str]) -> None:
        """Delete pruned chunk ids from BOTH stores (id-scoped, safe)."""
        if self._shared is None or self._private is None:
            self.open_stores()
        assert self._shared is not None and self._private is not None
        if not chunk_ids:
            return
        if self._private is not None:
            self._private.delete(chunk_ids)
        if self._shared is not None and self._shared.writable:
            self._shared.delete(chunk_ids)

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider

    @property
    def shared_store(self) -> Optional[ChromaVectorStore]:
        return self._shared

    @property
    def private_store(self) -> Optional[ChromaVectorStore]:
        return self._private


def resolve_endpoints(
    config: SphinxRAGConfig,
    *,
    shared_query_uri: Optional[str] = None,
    private_query_uri: Optional[str] = None,
    embed_uri: Optional[str] = None,
) -> dict[str, Any]:
    """Build the resolved per-scope endpoint descriptor for P6 (endpoints.json).

    P6 runs as a separate process at agent time and must learn, per scope, the
    HTTP ``{host, port, ssl, tenant, database, collection}`` to connect to.
    ``*_query_uri`` overrides carry the loopback URI when a store is embedded
    (embedded stores are served over a loopback HTTP server for the JS client).
    Secrets are NEVER written here — only header KEY names are referenced.
    """
    from urllib.parse import urlparse

    def _one(scope: AccessControl, target: StoreTarget, query_uri: Optional[str]) -> dict[str, Any]:
        addr = address_from_config(scope, config)
        uri = query_uri or target.uri or ""
        parsed = urlparse(uri) if uri else None
        ssl = bool(parsed and parsed.scheme == "https")
        host = (parsed.hostname if parsed else None) or "127.0.0.1"
        port = (parsed.port if parsed else None) or (443 if ssl else 8000)
        return {
            "mode": target.mode,
            "host": host,
            "port": port,
            "ssl": ssl,
            "tenant": addr.tenant,
            "database": addr.database,
            "collection": addr.collection,
            "header_keys": sorted(target.headers.keys()),
        }

    return {
        "embedding": {
            "model": config.embedding_model,
            "dimension": config.embedding_dimension,
            # Auto-started bge-m3 embed server URL, when available (Option E):
            # the query side reads this so no manual DOXTR_RAG_EMBED_URL is
            # needed for the common embedded deployment.
            "embed_url": embed_uri,
        },
        "shared": _one(AccessControl.SHARED_ORG, config.shared_target(), shared_query_uri),
        "private": _one(AccessControl.LOCAL_PRIVATE, config.local_target(), private_query_uri),
    }
