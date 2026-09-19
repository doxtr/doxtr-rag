"""Abstract vector-store contract and canonical addressing (CONTRACTS §C4/§C4.1).

This module is the single source of truth for *where each scope's vectors
live*: the ``(tenant, database, collection)`` triple per :class:`AccessControl`
scope. Both the Python ingestion engine (P5) and the TypeScript query extension
(P6) MUST derive addressing from here — never hard-code collection strings
independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from doxtr_rag.schema import AccessControl, DocumentChunk

#: The pinned dense embedding dimension for bge-m3 (CONVENTIONS §Embedding).
EMBEDDING_DIMENSION: int = 1024

#: Default databases per scope (may be overridden via config, C6).
DEFAULT_SHARED_DATABASE: str = "doxtr_shared"
DEFAULT_PRIVATE_DATABASE: str = "doxtr_private"
DEFAULT_TENANT: str = "default_tenant"


def collection_name(model_revision: str, dimension: int = EMBEDDING_DIMENSION) -> str:
    """Return the model-revision + dimension tagged collection name.

    Collections are versioned by ``<model_rev>_<dim>`` so a model or dimension
    change produces a distinct collection (drop & rebuild — CONVENTIONS
    §Collection versioning). The revision is sanitized to a safe token.
    """
    safe_rev = "".join(c if c.isalnum() else "_" for c in model_revision).strip("_")
    return f"{safe_rev}_{dimension}"


@dataclass(frozen=True)
class StoreAddress:
    """Resolved ``(tenant, database, collection)`` address for one scope."""

    tenant: str
    database: str
    collection: str


def address_for(
    scope: AccessControl,
    *,
    tenant: str = DEFAULT_TENANT,
    shared_database: str = DEFAULT_SHARED_DATABASE,
    private_database: str = DEFAULT_PRIVATE_DATABASE,
    model_revision: str,
    dimension: int = EMBEDDING_DIMENSION,
) -> StoreAddress:
    """Return the canonical address for a scope (C4.1 addressing table).

    ``SHARED_ORG`` → ``shared_database``; anything else (fail-closed) →
    ``private_database``. A single external server MAY host both scopes as
    separate databases under one tenant; isolation is enforced at the resolved
    database/collection boundary, not merely by process.

    NOTE: collection names are intentionally **scope-independent** (both scopes
    use ``<model_rev>_<dim>``). Cross-scope isolation is therefore enforced by
    the differing ``database``, NOT by collection name. A downstream isolation
    guard (P5) MUST key on the resolved ``database``, never on the collection.
    """
    database = shared_database if scope is AccessControl.SHARED_ORG else private_database
    return StoreAddress(
        tenant=tenant,
        database=database,
        collection=collection_name(model_revision, dimension),
    )


def address_from_config(scope: AccessControl, config: object) -> StoreAddress:
    """Resolve a :class:`StoreAddress` for ``scope`` from a config object.

    Convenience wrapper so P5 (and, mirrored, P6) do not re-thread
    tenant/database/revision into :func:`address_for` at every call site. The
    ``config`` is duck-typed against :class:`doxtr_rag.config.SphinxRAGConfig`
    (avoids a circular import): it must expose ``rag_store_tenant``,
    ``rag_shared_store_database``, ``rag_local_store_database``,
    ``model_revision`` and ``embedding_dimension``.
    """
    return address_for(
        scope,
        tenant=getattr(config, "rag_store_tenant", DEFAULT_TENANT),
        shared_database=getattr(config, "rag_shared_store_database", DEFAULT_SHARED_DATABASE),
        private_database=getattr(config, "rag_local_store_database", DEFAULT_PRIVATE_DATABASE),
        model_revision=getattr(config, "model_revision", "bge-m3"),
        dimension=getattr(config, "embedding_dimension", EMBEDDING_DIMENSION),
    )


class BaseVectorStore(ABC):
    """Abstract contract every concrete vector store backend implements."""

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """The resolved collection name (model-revision + dimension tagged)."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """The embedding dimension; MUST be 1024 (bge-m3 dense)."""

    @abstractmethod
    def upsert(self, chunks: list[DocumentChunk], batch_size: int = 100) -> None:
        """Insert or update chunks in configurable batches."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete chunks by id."""

    @abstractmethod
    def query(self, embedding: list[float], top_k: int) -> list[DocumentChunk]:
        """Return the top-``k`` nearest chunks for a query embedding."""
