"""P2 tests for the incremental build cache (CONTRACTS §C8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from doxtr_rag.cache import SCHEMA_VERSION, BuildCache


@pytest.fixture
def cache() -> BuildCache:
    return BuildCache(":memory:")


def test_new_resource_is_dirty(cache: BuildCache) -> None:
    assert cache.is_dirty("node:index#a", "hash1") is True


def test_unchanged_resource_not_dirty(cache: BuildCache) -> None:
    cache.update_cache("node:index#a", "node", "hash1", ["c1"])
    assert cache.is_dirty("node:index#a", "hash1") is False


def test_changed_hash_is_dirty(cache: BuildCache) -> None:
    cache.update_cache("node:index#a", "node", "hash1", ["c1"])
    assert cache.is_dirty("node:index#a", "hash2") is True


def test_three_hash_domains(cache: BuildCache) -> None:
    cache.update_cache("node:index#a", "node", "h1", ["c1"])
    cache.update_cache("file:/mnt/a.pdf", "external_file", "h2", ["c2", "c3"])
    cache.update_cache("xlink:JIRA-1", "xlink", "h3", ["c4"])
    assert cache.is_dirty("node:index#a", "h1") is False
    assert cache.is_dirty("file:/mnt/a.pdf", "h2") is False
    assert cache.is_dirty("xlink:JIRA-1", "h3") is False
    assert cache.chunk_ids_for("file:/mnt/a.pdf") == ["c2", "c3"]


def test_update_replaces_chunk_mapping(cache: BuildCache) -> None:
    cache.update_cache("node:x", "node", "h1", ["c1", "c2"])
    cache.update_cache("node:x", "node", "h2", ["c3"])
    assert cache.chunk_ids_for("node:x") == ["c3"]


def test_prune_deleted_returns_chunk_ids(cache: BuildCache) -> None:
    cache.update_cache("node:keep", "node", "h1", ["k1"])
    cache.update_cache("node:gone", "node", "h2", ["g1", "g2"])
    pruned = cache.prune_deleted(seen_ids={"node:keep"})
    assert sorted(pruned) == ["g1", "g2"]
    # keep survives; gone is removed.
    assert cache.is_dirty("node:keep", "h1") is False
    assert cache.is_dirty("node:gone", "h2") is True


def test_prune_nothing_when_all_seen(cache: BuildCache) -> None:
    cache.update_cache("node:a", "node", "h", ["c"])
    assert cache.prune_deleted({"node:a"}) == []


def test_schema_version_recorded(cache: BuildCache) -> None:
    assert cache.schema_version == SCHEMA_VERSION


def test_persistence_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "sub" / "cache.db"
    with BuildCache(db) as c:
        c.update_cache("node:a", "node", "h1", ["c1"])
    assert db.exists()
    with BuildCache(db) as c2:
        assert c2.is_dirty("node:a", "h1") is False
        assert c2.schema_version == SCHEMA_VERSION
