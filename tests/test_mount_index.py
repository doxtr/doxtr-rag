"""Tests for the portable per-mount sidecar index (``doxtr_rag.mount_index``).

Exercises the sidecar layout helpers, deterministic port/database derivation,
incremental (SHA-gated) indexing with pruning, mount resolution, the
best-effort single-writer lock, and the loopback-serving helpers. Uses an
on-disk embedded Chroma store with a fake 1024-dim provider so the tests run
without the pinned bge-m3 weights (mirroring ``tests/test_storage.py``).
"""

from __future__ import annotations

import os

import pytest

import doxtr_rag.embedding_provider as ep_mod
import doxtr_rag.storage as storage_mod
from doxtr_rag import mount_index as mi
from doxtr_rag.config import SphinxRAGConfig
from doxtr_rag.schema import AccessControl


class FakeProvider:
    """Deterministic 1024-dim provider so no bge-m3 weights are required."""

    dimension = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[(len(t) % 7) / 10.0] * 1024 for t in texts]


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # Store construction now goes through storage.open_named_store, which
    # references the EmbeddingProvider imported at storage module scope; patch
    # both the origin module and storage so every call site gets the fake.
    monkeypatch.setattr(ep_mod, "EmbeddingProvider", FakeProvider)
    monkeypatch.setattr(storage_mod, "EmbeddingProvider", FakeProvider)


def _config(tmp_path: str) -> SphinxRAGConfig:
    return SphinxRAGConfig.from_mapping(
        {
            "rag_local_store_mode": "embedded",
            "rag_local_store_path": os.path.join(tmp_path, "local_chroma"),
        }
    )


def _write(mount: str, name: str, content: str) -> str:
    p = os.path.join(mount, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(content)
    return p


# ---------------------------------------------------------------------------
# Layout / derivation helpers
# ---------------------------------------------------------------------------
def test_sidecar_layout_paths(tmp_path) -> None:
    m = str(tmp_path / "vault")
    assert mi.sidecar_dir(m).endswith(os.path.join("vault", ".doxtr"))
    assert mi.sidecar_cache_path(m).endswith(mi.SIDECAR_CACHE)
    assert mi.sidecar_chroma_path(m).endswith(mi.SIDECAR_CHROMA)


def test_resource_id_uses_walk_path() -> None:
    assert mi.resource_id_for_file("vault/a.md") == "external_file:vault/a.md"


def test_sidecar_database_name_stable_and_safe(tmp_path) -> None:
    m = str(tmp_path / "my vault!")
    name = mi._sidecar_database_name(m)
    assert name.startswith("mount_")
    # Only alnum / - / _ survive (Chroma-safe).
    assert all(ch.isalnum() or ch in "-_" for ch in name)
    assert mi._sidecar_database_name(m) == name  # stable


def test_mount_port_deterministic_and_in_window(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DOXTR_RAG_MOUNT_PORT_BASE", raising=False)
    m = str(tmp_path / "vault")
    p1 = mi.mount_port(m)
    p2 = mi.mount_port(m)
    assert p1 == p2
    assert mi.DEFAULT_MOUNT_PORT_BASE <= p1 < mi.DEFAULT_MOUNT_PORT_BASE + mi.MOUNT_PORT_WINDOW


def test_mount_port_base_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DOXTR_RAG_MOUNT_PORT_BASE", "9000")
    p = mi.mount_port(str(tmp_path / "vault"))
    assert 9000 <= p < 9000 + mi.MOUNT_PORT_WINDOW


def test_mount_port_base_env_invalid_falls_back(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DOXTR_RAG_MOUNT_PORT_BASE", "not-an-int")
    p = mi.mount_port(str(tmp_path / "vault"))
    assert mi.DEFAULT_MOUNT_PORT_BASE <= p < mi.DEFAULT_MOUNT_PORT_BASE + mi.MOUNT_PORT_WINDOW


# ---------------------------------------------------------------------------
# Mount resolution
# ---------------------------------------------------------------------------
def test_resolve_mount_bare_path_is_private() -> None:
    path, scope = mi._resolve_mount(SphinxRAGConfig.from_mapping({}), "/some/dir")
    assert path == "/some/dir" and scope is AccessControl.LOCAL_PRIVATE


def test_resolve_mount_dict_honours_declared_scope() -> None:
    cfg = SphinxRAGConfig.from_mapping({})
    path, scope = mi._resolve_mount(cfg, {"path": "/d", "scope": "SHARED_ORG"})
    assert path == "/d" and scope is AccessControl.SHARED_ORG


def test_find_mount_matches_realpath(tmp_path) -> None:
    m = str(tmp_path / "vault")
    os.makedirs(m)
    cfg = SphinxRAGConfig.from_mapping({"rag_external_mounts": [{"path": m}]})
    assert mi.find_mount(cfg, m) == {"path": m}
    assert mi.find_mount(cfg, str(tmp_path / "other")) is None


# ---------------------------------------------------------------------------
# Incremental indexing + pruning
# ---------------------------------------------------------------------------
def test_index_mount_builds_incremental_and_prunes(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    _write(mount, "a.md", "# A\n\nAlpha body.\n")
    _write(mount, "b.txt", "Beta paragraph one.\n\nBeta paragraph two.\n")
    cfg = _config(str(tmp_path))

    # First pass: everything is new + ingested.
    stats = mi.index_mount(cfg, mount)
    assert stats["scanned"] == 2
    assert stats["ingested_files"] == 2
    assert stats["chunks"] >= 2
    assert stats["skipped"] == 0
    assert os.path.isfile(mi.sidecar_cache_path(mount))
    assert os.path.isdir(mi.sidecar_chroma_path(mount))

    # Second pass, no changes: both files skipped (no re-embed).
    stats2 = mi.index_mount(cfg, mount)
    assert stats2["skipped"] == 2 and stats2["ingested_files"] == 0

    # Modify one file: only it is re-ingested.
    _write(mount, "a.md", "# A\n\nAlpha body CHANGED.\n")
    stats3 = mi.index_mount(cfg, mount)
    assert stats3["ingested_files"] == 1 and stats3["skipped"] == 1

    # Delete a file: it is pruned from the sidecar.
    os.remove(os.path.join(mount, "b.txt"))
    stats4 = mi.index_mount(cfg, mount)
    assert stats4["deleted"] >= 1


def test_index_mount_missing_directory(tmp_path) -> None:
    cfg = _config(str(tmp_path))
    stats = mi.index_mount(cfg, str(tmp_path / "nope"))
    assert stats == {
        "scanned": 0,
        "ingested_files": 0,
        "skipped": 0,
        "deleted": 0,
        "chunks": 0,
    }


def test_index_mount_skips_sidecar_dir(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    _write(mount, "a.md", "# A\n\nBody.\n")
    cfg = _config(str(tmp_path))
    mi.index_mount(cfg, mount)
    # A file living under .doxtr must never be scanned/ingested.
    files = list(mi._iter_supported_files(mount))
    assert all(".doxtr" not in f for f in files)


# ---------------------------------------------------------------------------
# watch_mount(once=True)
# ---------------------------------------------------------------------------
def test_watch_once_runs_single_pass(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    _write(mount, "a.md", "# A\n\nBody.\n")
    cfg = _config(str(tmp_path))
    mi.watch_mount(cfg, mount, once=True)
    assert os.path.isdir(mi.sidecar_chroma_path(mount))


def test_watch_missing_directory_returns(tmp_path) -> None:
    cfg = _config(str(tmp_path))
    # Should not raise.
    mi.watch_mount(cfg, str(tmp_path / "nope"), once=True)


# ---------------------------------------------------------------------------
# Single-writer lock
# ---------------------------------------------------------------------------
def test_mount_lock_prevents_second_writer(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mi.sidecar_dir(mount))
    lock_a = mi._MountLock(mount)
    lock_b = mi._MountLock(mount)
    assert lock_a.acquire() is True
    assert lock_b.acquire() is False  # live holder -> refused
    lock_a.release()
    assert lock_b.acquire() is True
    lock_b.release()


def test_mount_lock_reclaims_stale(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mi.sidecar_dir(mount))
    lock = mi._MountLock(mount)
    # Write a bogus dead PID.
    with open(lock.path, "w", encoding="utf-8") as fh:
        fh.write("999999999")
    assert lock.acquire() is True  # stale -> reclaimed
    lock.release()


def test_mount_lock_stale_on_bad_content(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mi.sidecar_dir(mount))
    lock = mi._MountLock(mount)
    # Non-numeric and non-positive PIDs are treated as stale.
    with open(lock.path, "w", encoding="utf-8") as fh:
        fh.write("garbage")
    assert lock._stale() is True
    with open(lock.path, "w", encoding="utf-8") as fh:
        fh.write("0")
    assert lock._stale() is True


def test_watch_mount_refuses_second_writer(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    cfg = _config(str(tmp_path))
    held = mi._MountLock(mount)
    assert held.acquire() is True
    try:
        # A live lock holder makes watch_mount refuse (single-writer invariant).
        mi.watch_mount(cfg, mount, once=True)
        assert not os.path.isdir(mi.sidecar_chroma_path(mount))
    finally:
        held.release()


# ---------------------------------------------------------------------------
# Serving sidecars
# ---------------------------------------------------------------------------
def test_serve_mount_without_sidecar_returns_none(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    cfg = _config(str(tmp_path))
    # No sidecar built yet -> nothing to serve.
    assert mi.serve_mount(cfg, mount) is None


def test_serve_all_mounts_skips_unbuilt(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    cfg = SphinxRAGConfig.from_mapping(
        {
            "rag_local_store_mode": "embedded",
            "rag_local_store_path": os.path.join(str(tmp_path), "local_chroma"),
            "rag_external_mounts": [{"path": mount}],
        }
    )
    uris, servers = mi.serve_all_mounts(cfg)
    # No sidecar -> no server, empty maps (never raises).
    assert uris == {} and servers == []


def test_serve_built_sidecar_over_loopback(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    _write(mount, "a.md", "# A\n\nBody.\n")
    cfg = _config(str(tmp_path))
    mi.index_mount(cfg, mount)
    server = mi.serve_mount(cfg, mount)
    try:
        assert server is not None
        assert server.uri.startswith("http://127.0.0.1:")
        # Deterministic port for stable endpoints.json.
        assert str(mi.mount_port(mount)) in server.uri
    finally:
        if server is not None:
            server.stop()


def test_serve_all_mounts_serves_built_sidecar(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    _write(mount, "a.md", "# A\n\nBody.\n")
    cfg = SphinxRAGConfig.from_mapping(
        {
            "rag_local_store_mode": "embedded",
            "rag_local_store_path": os.path.join(str(tmp_path), "local_chroma"),
            "rag_external_mounts": [{"path": mount}],
        }
    )
    mi.index_mount(cfg, mount)
    uris, servers = mi.serve_all_mounts(cfg)
    try:
        assert mount in uris
        assert len(servers) == 1
    finally:
        for s in servers:
            s.stop()


# ---------------------------------------------------------------------------
# Symlink confinement + empty-extraction handling
# ---------------------------------------------------------------------------
def test_iter_supported_files_skips_escaping_symlink(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    outside = str(tmp_path / "outside")
    os.makedirs(mount)
    os.makedirs(outside)
    with open(os.path.join(outside, "secret.md"), "w", encoding="utf-8") as fh:
        fh.write("# Secret\n\nLeaked.\n")
    link = os.path.join(mount, "escape.md")
    try:
        os.symlink(os.path.join(outside, "secret.md"), link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported on this platform")
    files = list(mi._iter_supported_files(mount))
    # The escaping symlink is skipped fail-closed.
    assert all(os.path.realpath(f) != os.path.realpath(link) for f in files)


def test_index_mount_forgets_empty_file(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    # A registered extension but with no extractable content.
    _write(mount, "empty.md", "   \n\n   \n")
    cfg = _config(str(tmp_path))
    stats = mi.index_mount(cfg, mount)
    assert stats["scanned"] == 1
    assert stats["ingested_files"] == 0


# ---------------------------------------------------------------------------
# Polling watcher (single incremental cycle then stop)
# ---------------------------------------------------------------------------
def test_polling_watcher_runs_incrementally(tmp_path, monkeypatch) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    _write(mount, "a.md", "# A\n\nBody.\n")
    cfg = _config(str(tmp_path))

    calls = {"n": 0}
    real_index = mi.index_mount

    def _counting_index(c, m, **kw):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise KeyboardInterrupt  # break the poll loop deterministically
        return real_index(c, m, **kw)

    monkeypatch.setattr(mi, "index_mount", _counting_index)
    # Force the polling path (pretend watchdog is absent).
    monkeypatch.setattr(
        mi, "_watch_with_watchdog", lambda *a, **k: (_ for _ in ()).throw(ImportError())
    )
    mi.watch_mount(cfg, mount, poll_interval=0.01)
    assert calls["n"] >= 2


def test_watchdog_watcher_indexes_on_event(tmp_path, monkeypatch) -> None:
    """Exercise the event-driven watcher with a stub watchdog module."""
    import sys
    import types

    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    _write(mount, "a.md", "# A\n\nBody.\n")
    cfg = _config(str(tmp_path))

    fired: dict[str, object] = {}

    events_mod = types.ModuleType("watchdog.events")

    class _FSEH:  # minimal FileSystemEventHandler stand-in
        pass

    events_mod.FileSystemEventHandler = _FSEH

    observers_mod = types.ModuleType("watchdog.observers")

    class _Observer:
        def schedule(self, handler, path, recursive=False):
            fired["handler"] = handler

        def start(self):
            # Fire one non-directory event so the handler marks work pending.
            evt = types.SimpleNamespace(
                is_directory=False, src_path=os.path.join(mount, "a.md"), dest_path=""
            )
            fired["handler"].on_any_event(evt)

        def stop(self):
            pass

        def join(self):
            pass

    observers_mod.Observer = _Observer

    watchdog_pkg = types.ModuleType("watchdog")
    monkeypatch.setitem(sys.modules, "watchdog", watchdog_pkg)
    monkeypatch.setitem(sys.modules, "watchdog.events", events_mod)
    monkeypatch.setitem(sys.modules, "watchdog.observers", observers_mod)

    calls = {"n": 0}
    real_index = mi.index_mount

    class _StopLoop(Exception):
        pass

    def _counting_index(c, m, **kw):
        calls["n"] += 1
        result = real_index(c, m, **kw)
        if calls["n"] >= 2:
            # Break out of the (otherwise infinite) watch loop deterministically
            # once the event-driven re-index has run.
            raise _StopLoop
        return result

    monkeypatch.setattr(mi, "index_mount", _counting_index)

    # Advance a fake monotonic clock so the debounce elapses on the 2nd sleep,
    # triggering the event-driven index_mount which then raises _StopLoop.
    ticks = iter([0.0, 100.0, 200.0, 300.0, 400.0])
    monkeypatch.setattr(mi.time, "monotonic", lambda: next(ticks, 999.0))
    monkeypatch.setattr(mi.time, "sleep", lambda _s: None)

    with pytest.raises(_StopLoop):
        mi.watch_mount(cfg, mount, poll_interval=1.0)
    # Initial catch-up pass + at least one event-driven re-index.
    assert calls["n"] >= 2


# ---------------------------------------------------------------------------
# endpoints.json "mounts" array (resolve_endpoints integration)
# ---------------------------------------------------------------------------
def test_resolve_endpoints_publishes_mounts(tmp_path) -> None:
    from doxtr_rag.storage import resolve_endpoints

    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    cfg = SphinxRAGConfig.from_mapping(
        {
            "rag_local_store_mode": "embedded",
            "rag_local_store_path": os.path.join(str(tmp_path), "local_chroma"),
            "rag_shared_store_mode": "embedded",
            "rag_shared_store_uri": os.path.join(str(tmp_path), "shared_chroma"),
            "rag_external_mounts": [{"path": mount}],
        }
    )
    # With a served-mount URI, the mount is published under "mounts".
    desc = resolve_endpoints(
        cfg,
        mount_descriptors=mi.build_mount_endpoints(
            cfg, {mount: "http://127.0.0.1:8801"}
        ),
    )
    assert len(desc["mounts"]) == 1
    entry = desc["mounts"][0]
    assert entry["path"] == mount
    assert entry["scope"] == "LOCAL_PRIVATE"  # fail-closed default
    assert entry["port"] == 8801
    assert entry["database"] == mi._sidecar_database_name(mount)


def test_resolve_endpoints_omits_unserved_mount(tmp_path) -> None:
    from doxtr_rag.storage import resolve_endpoints

    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    cfg = SphinxRAGConfig.from_mapping(
        {
            "rag_local_store_mode": "embedded",
            "rag_local_store_path": os.path.join(str(tmp_path), "local_chroma"),
            "rag_shared_store_mode": "embedded",
            "rag_shared_store_uri": os.path.join(str(tmp_path), "shared_chroma"),
            "rag_external_mounts": [{"path": mount}],
        }
    )
    # No URI for this mount -> omitted, and backward-compatible empty array.
    assert mi.build_mount_endpoints(cfg, {}) == []
    desc = resolve_endpoints(cfg)
    assert desc["mounts"] == []
