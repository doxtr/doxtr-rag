"""Portable per-mount sidecar index + background watcher (doxtr-rag).

External mounts (e.g. a ``vault/`` folder of DevOps role descriptions, HAR
captures, meeting exports) rarely change but can hold hundreds of documents.
Re-scanning and re-embedding them on every ``make html`` is wasteful, and a
fresh checkout cannot query them at all until a build has run once.

This module adds a **portable, per-mount sidecar** that lives *next to the
files* under ``<mount>/.doxtr/``:

- ``<mount>/.doxtr/mount-cache.db`` — a SQLite hash cache (same shape as the
  build :class:`~doxtr_rag.cache.BuildCache`) tracking each file's SHA-256 and
  the chunk ids it produced. This is what lets an incremental scan *skip*
  unchanged files without re-reading/parsing them.
- ``<mount>/.doxtr/chroma`` — an embedded ChromaDB holding the mount's vectors,
  co-located with the files so a person who checks out the folder can query it
  immediately, with no ``make html``-triggered ingest first.

Two entry points drive it (both standalone; no Sphinx needed):

- :func:`index_mount` — build/refresh the sidecar once (``doxtr-rag mount-index``).
- :func:`watch_mount` — a single-consumer background loop that detects
  create/modify/delete via SHA comparison and incrementally
  ingests/prunes the sidecar (``doxtr-rag mount-watch``). Uses ``watchdog`` for
  event-driven change detection when installed, else polls.

Design invariants (mirroring the build pipeline):

- **Single writer per sidecar DB.** ``index_mount`` runs once; ``watch_mount``
  is a single consumer loop. Neither runs concurrently with the other against
  the same mount by design (a lock file guards this best-effort).
- **Fail-closed scope.** The mount's scope comes from its declaration in
  ``conf.py`` (``SHARED_ORG`` only if affirmatively declared), defaulting to
  ``LOCAL_PRIVATE`` — identical to :meth:`SphinxRAGConfig.mount_scope`.
- **Never crash.** Every stage degrades gracefully with a log line.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from doxtr_rag.base_store import collection_name
from doxtr_rag.cache import BuildCache
from doxtr_rag.config import SphinxRAGConfig
from doxtr_rag.schema import AccessControl, DocumentChunk

logger = logging.getLogger(__name__)

#: Sidecar directory name placed under each mount. Already skipped by
#: :func:`doxtr_rag.extractors.scan_mount` (it never descends into ``.doxtr``),
#: so the sidecar never ingests itself.
SIDECAR_DIRNAME = ".doxtr"
SIDECAR_CACHE = "mount-cache.db"
SIDECAR_CHROMA = "chroma"
SIDECAR_LOCK = "mount.lock"

#: Poll interval (seconds) for the fallback watcher when ``watchdog`` is absent.
DEFAULT_POLL_INTERVAL = 5.0

#: Base loopback port for per-mount Chroma query servers. Each configured mount
#: gets a deterministic port derived from this base + a stable hash of its
#: realpath, kept BELOW the Linux ephemeral range so transient ``chroma run``
#: subprocesses cannot collide. Override the base with DOXTR_RAG_MOUNT_PORT_BASE.
DEFAULT_MOUNT_PORT_BASE = 8790
#: Size of the port window mount servers are hashed into (base .. base+window-1).
MOUNT_PORT_WINDOW = 200


def mount_port(mount_path: str) -> int:
    """Deterministic loopback port for a mount's sidecar Chroma server.

    Stable across restarts (derived from the mount realpath) so the published
    ``endpoints.json`` never points at a drifting port. Collisions within a
    project are unlikely given the ``MOUNT_PORT_WINDOW`` spread. NOTE: if two
    mounts do hash to the same port, :func:`serve_mount` adopts any live
    listener on that port (``reuse_if_running=True``) WITHOUT verifying it
    serves the same store path — so the second mount would silently share the
    first's server and its ``endpoints.json`` entry would point at the first
    mount's database/collection. Override ``DOXTR_RAG_MOUNT_PORT_BASE`` (or
    space the mounts) if a collision is observed.
    """
    import hashlib

    base = DEFAULT_MOUNT_PORT_BASE
    raw = os.environ.get("DOXTR_RAG_MOUNT_PORT_BASE")
    if raw and raw.strip():
        try:
            base = int(raw)
        except ValueError:
            base = DEFAULT_MOUNT_PORT_BASE
    digest = hashlib.sha256(os.path.realpath(mount_path).encode("utf-8")).digest()
    return base + (digest[0] << 8 | digest[1]) % MOUNT_PORT_WINDOW


# ---------------------------------------------------------------------------
# Sidecar layout helpers
# ---------------------------------------------------------------------------
def sidecar_dir(mount_path: str) -> str:
    return os.path.join(mount_path, SIDECAR_DIRNAME)


def sidecar_cache_path(mount_path: str) -> str:
    return os.path.join(sidecar_dir(mount_path), SIDECAR_CACHE)


def sidecar_chroma_path(mount_path: str) -> str:
    return os.path.join(sidecar_dir(mount_path), SIDECAR_CHROMA)


def _sidecar_database_name(mount_path: str) -> str:
    """A stable, filesystem-independent database name for the mount.

    The vectors live in the sidecar's own embedded Chroma directory, so the
    database name only needs to be stable and readable; derive it from the
    mount's basename with a short hash of the realpath for uniqueness.
    """
    import hashlib

    base = os.path.basename(os.path.normpath(mount_path)) or "mount"
    digest = hashlib.sha256(os.path.realpath(mount_path).encode("utf-8")).hexdigest()[:8]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in base)
    return f"mount_{safe}_{digest}"


def resource_id_for_file(path: str) -> str:
    """Canonical sidecar resource id for a mounted file (mirrors the build).

    IMPORTANT: this must use the SAME ``path`` string the extractor sees so
    chunk ids and the ``external_file:<path>`` resource id are byte-identical
    between a sidecar-built store and a Sphinx-build-built store. The build
    derives the id from the extractor's ``source_file`` metadata, which is the
    ``os.walk`` join path (NOT the realpath). We therefore key on that same
    path and never substitute the realpath here.
    """
    return f"external_file:{path}"


# ---------------------------------------------------------------------------
# Sidecar Chroma store (embedded, co-located with the files)
# ---------------------------------------------------------------------------
def _open_sidecar_store(
    config: SphinxRAGConfig, mount_path: str, *, writable: bool
) -> Any:
    """Open (creating if writable) the embedded Chroma store under the mount.

    Delegates construction to the single public
    :func:`doxtr_rag.storage.open_named_store` factory (the sidecar is an
    ad-hoc, independently-addressed embedded store co-located with the mount at
    ``<mount>/.doxtr/chroma``). Opened as ``LOCAL_PRIVATE`` — a sidecar is a
    self-contained store and the SHARED_ORG isolation guard only fires for a
    store opened as shared.
    """
    from doxtr_rag.storage import open_named_store

    return open_named_store(
        config,
        database=_sidecar_database_name(mount_path),
        path=sidecar_chroma_path(mount_path),
        scope=AccessControl.LOCAL_PRIVATE,
        writable=writable,
    )


def build_mount_endpoints(
    config: SphinxRAGConfig, mount_query_uris: dict[str, str]
) -> list[dict[str, Any]]:
    """Build the fully-formed ``endpoints.json`` ``"mounts"`` entries.

    ``mount_query_uris`` maps a configured mount path to the loopback URI of the
    Chroma server serving that mount's portable sidecar store
    (``<mount>/.doxtr/chroma``). Returns one descriptor per served mount — each
    ``{path, scope, mode, host, port, ssl, tenant, database, collection,
    header_keys}`` — which :func:`doxtr_rag.storage.resolve_endpoints` publishes
    verbatim. A mount without a URI (no sidecar built, or not served) is
    omitted. Owning this here (rather than in ``storage``) keeps the endpoints
    writer decoupled from the sidecar addressing scheme: a child theme can swap
    mount addressing by overriding this one function.
    """
    from urllib.parse import urlparse

    entries: list[dict[str, Any]] = []
    coll = collection_name(config.model_revision, config.embedding_dimension)
    for mount in config.rag_external_mounts:
        if not isinstance(mount, dict):
            continue
        mpath = mount.get("path")
        if not mpath:
            continue
        uri = mount_query_uris.get(mpath) or mount_query_uris.get(
            os.path.realpath(mpath)
        )
        if not uri:
            continue  # no sidecar server for this mount -> not queryable via P6
        parsed = urlparse(uri)
        ssl = parsed.scheme == "https"
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if ssl else 8000)
        scope = config.mount_scope(mount)
        entries.append(
            {
                "path": mpath,
                "scope": scope.value,
                "mode": "embedded",
                "host": host,
                "port": port,
                "ssl": ssl,
                "tenant": config.rag_store_tenant,
                "database": _sidecar_database_name(mpath),
                "collection": coll,
                "header_keys": [],
            }
        )
    return entries


# ---------------------------------------------------------------------------
# File enumeration + change detection
# ---------------------------------------------------------------------------
def iter_supported_files(mount_path: str) -> Iterator[str]:
    """Yield walk paths of supported files under ``mount_path`` (skips .doxtr).

    Yields the ``os.walk`` join path (NOT the realpath) on purpose: it is the
    same path string the extractor sees, and :func:`resource_id_for_file`
    depends on it for build/sidecar ID parity — never substitute the realpath
    here. Supported extensions are read from the public
    :func:`doxtr_rag.extractors.supported_extensions` so a child-registered
    extractor is honoured on the scan path too.
    """
    from doxtr_rag.extractors import supported_extensions
    from doxtr_rag.pathscope import path_is_confined

    supported = supported_extensions()
    roots = [mount_path]
    for root, _dirs, files in os.walk(mount_path):
        if SIDECAR_DIRNAME in Path(root).parts:
            continue
        for name in sorted(files):
            fpath = os.path.join(root, name)
            if Path(fpath).suffix.lower() not in supported:
                continue
            if not path_is_confined(fpath, roots):
                logger.warning(
                    "doxtr-rag: symlink/path escapes mount %s -> skipped "
                    "(fail-closed): %s",
                    mount_path,
                    fpath,
                )
                continue
            yield fpath


#: Backward-compatible private alias (kept for existing callers/tests). Prefer
#: the public :func:`iter_supported_files`.
_iter_supported_files = iter_supported_files


def _extract_file_chunks(
    path: str, scope: AccessControl, config: SphinxRAGConfig
) -> list[DocumentChunk]:
    from doxtr_rag.extractors import extract_file

    return list(
        extract_file(
            path,
            scope,
            max_chunk_chars=config.rag_max_chunk_chars,
            chunk_overlap_chars=config.rag_chunk_overlap_chars,
        )
    )


# ---------------------------------------------------------------------------
# One-shot / incremental indexing
# ---------------------------------------------------------------------------
def index_mount(
    config: SphinxRAGConfig,
    mount: dict[str, Any] | str,
    *,
    batch_size: int = 100,
    prune: bool = True,
) -> dict[str, int]:
    """Build or refresh the sidecar index for one mount (incremental).

    Only files whose SHA-256 differs from the sidecar cache are re-extracted
    and re-embedded; unchanged files are skipped entirely. Deleted files are
    pruned from both the sidecar cache and its Chroma store when ``prune``.

    Returns a small stats dict: ``{"scanned", "ingested_files", "skipped",
    "deleted", "chunks"}``.
    """
    mount_path, scope = _resolve_mount(config, mount)
    if not os.path.isdir(mount_path):
        logger.warning("doxtr-rag: mount directory not found: %s", mount_path)
        return {"scanned": 0, "ingested_files": 0, "skipped": 0, "deleted": 0, "chunks": 0}

    os.makedirs(sidecar_dir(mount_path), exist_ok=True)
    stats = {"scanned": 0, "ingested_files": 0, "skipped": 0, "deleted": 0, "chunks": 0}

    from doxtr_rag.extractors import _sha256_file  # noqa: SLF001

    cache = BuildCache(sidecar_cache_path(mount_path))
    store = _open_sidecar_store(config, mount_path, writable=True)
    if not store.writable:
        logger.warning(
            "doxtr-rag: sidecar store under %s is not writable; aborting index",
            mount_path,
        )
        cache.close()
        return stats

    seen_ids: set[str] = set()
    try:
        for fpath in iter_supported_files(mount_path):
            stats["scanned"] += 1
            rid = resource_id_for_file(fpath)
            seen_ids.add(rid)
            try:
                sha = _sha256_file(fpath)
            except OSError:
                sha = ""
            if not cache.is_dirty(rid, sha):
                stats["skipped"] += 1
                continue
            chunks = _extract_file_chunks(fpath, scope, config)
            if not chunks:
                # Nothing extractable now: drop any stale chunks + forget it.
                stale = cache.chunk_ids_for(rid)
                if stale:
                    store.delete(stale)
                cache.update_cache(rid, "external_file", sha, [])
                continue
            new_ids = {c.id for c in chunks}
            stale = [cid for cid in cache.chunk_ids_for(rid) if cid not in new_ids]
            if stale:
                store.delete(stale)
            store.upsert(chunks, batch_size)
            cache.update_cache(rid, "external_file", sha, [c.id for c in chunks])
            stats["ingested_files"] += 1
            stats["chunks"] += len(chunks)

        if prune:
            stale_chunk_ids = cache.prune_deleted(seen_ids)
            if stale_chunk_ids:
                store.delete(stale_chunk_ids)
                stats["deleted"] = len(stale_chunk_ids)
    finally:
        cache.close()
    logger.info("[doxtr-rag] mount-index %s: %s", mount_path, stats)
    return stats


# ---------------------------------------------------------------------------
# Background watcher (single consumer)
# ---------------------------------------------------------------------------
def watch_mount(
    config: SphinxRAGConfig,
    mount: dict[str, Any] | str,
    *,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    batch_size: int = 100,
    once: bool = False,
) -> None:
    """Watch a mount and keep its sidecar index up to date (single consumer).

    Detects created/modified/deleted files and re-indexes incrementally. Uses
    ``watchdog`` when installed for event-driven detection (debounced), else a
    periodic poll. An initial full :func:`index_mount` pass runs before
    watching so the sidecar is current on start. ``once=True`` performs that
    single pass and returns (useful for tests / cron-style refresh).
    """
    mount_path, _scope = _resolve_mount(config, mount)
    if not os.path.isdir(mount_path):
        logger.warning("doxtr-rag: mount directory not found: %s", mount_path)
        return

    lock = _MountLock(mount_path)
    if not lock.acquire():
        logger.warning(
            "doxtr-rag: another indexer holds %s; refusing to start a second "
            "writer (single-writer invariant)",
            lock.path,
        )
        return
    try:
        # Initial catch-up pass so a fresh start reflects current state.
        index_mount(config, mount, batch_size=batch_size)
        if once:
            return
        try:
            _watch_with_watchdog(config, mount, mount_path, poll_interval, batch_size)
        except ImportError:
            logger.info(
                "[doxtr-rag] 'watchdog' not installed; falling back to polling "
                "every %.1fs for %s",
                poll_interval,
                mount_path,
            )
            _watch_by_polling(config, mount, mount_path, poll_interval, batch_size)
    finally:
        lock.release()


def _watch_with_watchdog(
    config: SphinxRAGConfig,
    mount: dict[str, Any] | str,
    mount_path: str,
    poll_interval: float,
    batch_size: int,
) -> None:
    """Event-driven watcher (raises ImportError if watchdog is absent)."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    debounce = max(1.0, poll_interval)
    pending: dict[str, float] = {}

    # watchdog is an optional dependency: when it is not installed, mypy sees
    # `FileSystemEventHandler` as `Any` and (under strict) rejects subclassing
    # it. The import is guarded at runtime (this function raises ImportError if
    # watchdog is absent), so silence the subclass-of-Any check here.
    class _Handler(FileSystemEventHandler):  # type: ignore[misc]  # pragma: no cover - thin adapter
        def on_any_event(self, event: Any) -> None:
            if getattr(event, "is_directory", False):
                return
            path = getattr(event, "dest_path", "") or getattr(event, "src_path", "")
            if not path or SIDECAR_DIRNAME in Path(path).parts:
                return
            pending[os.path.realpath(mount_path)] = time.monotonic()

    observer = Observer()
    observer.schedule(_Handler(), mount_path, recursive=True)
    observer.start()
    logger.info("[doxtr-rag] watching %s (watchdog)", mount_path)
    try:
        while True:
            time.sleep(debounce)
            if not pending:
                continue
            last = max(pending.values())
            if time.monotonic() - last >= debounce:
                pending.clear()
                index_mount(config, mount, batch_size=batch_size)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        observer.stop()
        observer.join()


def _watch_by_polling(
    config: SphinxRAGConfig,
    mount: dict[str, Any] | str,
    mount_path: str,
    poll_interval: float,
    batch_size: int,
) -> None:
    """Polling watcher: re-run the incremental index on a fixed interval.

    :func:`index_mount` is already incremental (SHA-gated), so a poll that
    finds nothing changed is cheap: it stats + hashes files but re-embeds
    none.
    """
    logger.info("[doxtr-rag] watching %s (poll %.1fs)", mount_path, poll_interval)
    try:
        while True:
            time.sleep(poll_interval)
            index_mount(config, mount, batch_size=batch_size)
    except KeyboardInterrupt:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _resolve_mount(
    config: SphinxRAGConfig, mount: dict[str, Any] | str
) -> tuple[str, AccessControl]:
    """Normalize a mount (dict from conf.py, or a bare path) to (path, scope)."""
    if isinstance(mount, str):
        return mount, AccessControl.LOCAL_PRIVATE
    path = mount.get("path", "")
    return path, config.mount_scope(mount)


def find_mount(config: SphinxRAGConfig, path: str) -> Optional[dict[str, Any]]:
    """Return the configured mount dict whose path matches ``path`` (realpath).

    Lets the CLI honour a mount's declared ``scope`` even when the user passes
    just the directory path. Falls back to None (caller treats it as a bare,
    private path) when no configured mount matches.
    """
    target = os.path.realpath(path)
    for mount in config.rag_external_mounts:
        if not isinstance(mount, dict):
            continue
        mpath = mount.get("path")
        if mpath and os.path.realpath(mpath) == target:
            return mount
    return None


class _MountLock:
    """Best-effort single-writer lock file under the mount's sidecar dir.

    Not a hard mutex (no fcntl dependency); it prevents the common footgun of
    two ``mount-watch``/``mount-index`` processes writing the same sidecar. A
    stale lock (holding PID no longer alive) is reclaimed.
    """

    def __init__(self, mount_path: str) -> None:
        self.path = os.path.join(sidecar_dir(mount_path), SIDECAR_LOCK)

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path) and not self._stale():
            return False
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.write(str(os.getpid()))
            return True
        except OSError:
            return False

    def _stale(self) -> bool:
        try:
            with open(self.path, encoding="utf-8") as fh:
                pid = int((fh.read() or "0").strip() or "0")
        except (OSError, ValueError):
            return True
        if pid <= 0:
            return True
        try:
            os.kill(pid, 0)
        except OSError:
            return True  # no such process -> stale
        return False

    def release(self) -> None:
        try:
            if os.path.exists(self.path):
                os.unlink(self.path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Serving sidecars over loopback (for the HTTP-only JS query client)
# ---------------------------------------------------------------------------
def serve_mount(config: SphinxRAGConfig, mount_path: str) -> Optional[Any]:
    """Start (or reuse) a loopback Chroma server over a mount's sidecar store.

    Returns the running server (with ``.uri``) or ``None`` when the mount has
    no built sidecar (nothing to serve) or the server could not be started.
    The port is deterministic (:func:`mount_port`) so ``endpoints.json`` stays
    stable across restarts. Best-effort and non-raising.
    """
    chroma_path = sidecar_chroma_path(mount_path)
    if not os.path.isdir(chroma_path):
        logger.info(
            "[doxtr-rag] no sidecar to serve for %s (run 'doxtr-rag mount-index' "
            "first); skipping",
            mount_path,
        )
        return None
    try:
        from doxtr_rag.server import LoopbackChromaServer

        server = LoopbackChromaServer(chroma_path, port=mount_port(mount_path))
        server.start(reuse_if_running=True)
        logger.info("[doxtr-rag] serving mount sidecar %s -> %s", mount_path, server.uri)
        return server
    except Exception as exc:  # noqa: BLE001 - never raise from serving
        logger.warning(
            "doxtr-rag: could not serve sidecar for %s: %s", mount_path, exc
        )
        return None


def serve_all_mounts(config: SphinxRAGConfig) -> tuple[dict[str, str], list[Any]]:
    """Serve every configured mount that has a built sidecar.

    Returns ``(uris, servers)`` where ``uris`` maps mount path -> loopback URI
    (pass to :func:`build_mount_endpoints` to produce the ``endpoints.json``
    ``"mounts"`` entries) and ``servers`` is the list of started servers the
    caller must keep alive / shut down.
    """
    uris: dict[str, str] = {}
    servers: list[Any] = []
    for mount in config.rag_external_mounts:
        if not isinstance(mount, dict):
            continue
        mpath = mount.get("path")
        if not mpath or not os.path.isdir(mpath):
            continue
        server = serve_mount(config, mpath)
        if server is not None:
            uris[mpath] = server.uri
            servers.append(server)
    return uris, servers
