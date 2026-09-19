"""F1 tests for base_store contract + canonical addressing (CONTRACTS §C4/§C4.1)."""

from __future__ import annotations

import pytest

from doxtr_rag.base_store import (
    DEFAULT_PRIVATE_DATABASE,
    DEFAULT_SHARED_DATABASE,
    EMBEDDING_DIMENSION,
    BaseVectorStore,
    address_for,
    address_from_config,
    collection_name,
)
from doxtr_rag.schema import AccessControl, DocumentChunk


def test_embedding_dimension_is_1024() -> None:
    assert EMBEDDING_DIMENSION == 1024


def test_collection_name_tags_revision_and_dim() -> None:
    assert collection_name("bge-m3") == "bge_m3_1024"
    assert collection_name("bge-m3", 512) == "bge_m3_512"


def test_address_for_shared() -> None:
    addr = address_for(AccessControl.SHARED_ORG, model_revision="bge-m3")
    assert addr.database == DEFAULT_SHARED_DATABASE
    assert addr.collection == "bge_m3_1024"
    assert addr.tenant == "default_tenant"


def test_address_for_private_and_failclosed() -> None:
    addr = address_for(AccessControl.LOCAL_PRIVATE, model_revision="bge-m3")
    assert addr.database == DEFAULT_PRIVATE_DATABASE
    # A non-SHARED value must map to the private database (fail-closed).
    addr2 = address_for(AccessControl.coerce("bogus"), model_revision="bge-m3")
    assert addr2.database == DEFAULT_PRIVATE_DATABASE


def test_address_for_custom_tenant_databases() -> None:
    addr = address_for(
        AccessControl.SHARED_ORG,
        tenant="acme",
        shared_database="sh",
        private_database="pr",
        model_revision="rev1",
    )
    assert (addr.tenant, addr.database) == ("acme", "sh")


def test_address_from_config() -> None:
    from doxtr_rag.config import SphinxRAGConfig

    c = SphinxRAGConfig.from_mapping({"rag_store_tenant": "acme"})
    shared = address_from_config(AccessControl.SHARED_ORG, c)
    private = address_from_config(AccessControl.LOCAL_PRIVATE, c)
    assert shared.tenant == "acme"
    assert shared.database == DEFAULT_SHARED_DATABASE
    assert private.database == DEFAULT_PRIVATE_DATABASE
    # Collection names are scope-INDEPENDENT (isolation is by database).
    assert shared.collection == private.collection == "bge_m3_1024"


def test_base_store_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseVectorStore()  # type: ignore[abstract]


def test_base_store_concrete_subclass() -> None:
    class Mem(BaseVectorStore):
        def __init__(self) -> None:
            self.store: dict[str, DocumentChunk] = {}

        @property
        def collection_name(self) -> str:
            return "bge_m3_1024"

        @property
        def embedding_dimension(self) -> int:
            return EMBEDDING_DIMENSION

        def upsert(self, chunks: list[DocumentChunk], batch_size: int = 100) -> None:
            for c in chunks:
                self.store[c.id] = c

        def delete(self, ids: list[str]) -> None:
            for i in ids:
                self.store.pop(i, None)

        def query(self, embedding: list[float], top_k: int) -> list[DocumentChunk]:
            return list(self.store.values())[:top_k]

    m = Mem()
    assert m.embedding_dimension == 1024
    assert m.collection_name == "bge_m3_1024"
    assert m.query([0.0], 3) == []
