"""Build pipeline wiring (P5 — the ``doxtr_rag.pipeline.Pipeline`` seam).

Implements the internal ``Pipeline`` interface that :mod:`doxtr_rag._lifecycle`
drives across the three Sphinx events (documented in F0's lifecycle module):

    Pipeline.from_app(app) -> Pipeline
    pipeline.builder_inited()        # load config, resolve mounts, start loopback
                                     #   servers for embedded stores, open stores,
                                     #   scan external mounts (P3)
    pipeline.doctree_resolved(doctree, docname)  # harvest (P2), cache-check, buffer
    pipeline.build_finished(exception)           # batch upsert + prune + endpoints
                                                 #   + flush + close

Serving topology (P6 forward-compat): an ``embedded`` store is served over a
loopback-bound Chroma HTTP server, and BOTH P5 ingestion and P6 queries go
through that HTTP endpoint (one process writes the on-disk store — the server —
so there is no PersistentClient/server contention). An ``http`` store is used
directly. The resolved per-scope endpoints are written to
``.doxtr/endpoints.json`` so the separate P6 process can connect without
re-deriving addressing or guessing the ephemeral loopback port.

All stages are defensive: failures are logged and never crash ``sphinx-build``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from doxtr_rag.cache import BuildCache
from doxtr_rag.config import SphinxRAGConfig, StoreTarget
from doxtr_rag.schema import AccessControl, DocumentChunk
from doxtr_rag.storage import IngestionEngine, resolve_endpoints

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from sphinx.application import Sphinx


@dataclass
class _Resource:
    """A group of chunks belonging to one cache resource (node/file/xlink)."""

    resource_id: str
    kind: str
    sha256: str
    chunks: list[DocumentChunk] = field(default_factory=list)
    extra_scopes: list[AccessControl | str | None] = field(default_factory=list)


class Pipeline:
    """Owns the per-build ingestion lifecycle."""

    def __init__(
        self,
        config: SphinxRAGConfig,
        *,
        srcdir: str = ".",
        engine: Optional[IngestionEngine] = None,
        cache: Optional[BuildCache] = None,
        app: Optional[Any] = None,
        start_servers: bool = True,
        serve_embed: bool = True,
    ) -> None:
        self._config = config
        self._srcdir = srcdir
        self._app = app
        self._engine = engine
        self._cache = cache
        self._resources: dict[str, _Resource] = {}
        self._seen_resource_ids: set[str] = set()
        self._servers: list[Any] = []
        self._start_servers = start_servers
        self._serve_embed = serve_embed
        self._loopback_uris: dict[str, Optional[str]] = {"shared": None, "private": None}
        self._embed_uri: Optional[str] = None
        self._mount_uris: dict[str, str] = {}
        self._xlink_scopes: dict[str, AccessControl] = {}

    @classmethod
    def from_app(cls, app: "Sphinx") -> "Pipeline":
        mapping = {
            k: getattr(app.config, k)
            for k in SphinxRAGConfig.model_fields
            if hasattr(app.config, k)
        }
        config = SphinxRAGConfig.from_mapping(mapping)
        srcdir = str(getattr(app, "srcdir", "."))
        return cls(config, srcdir=srcdir, app=app)

    # -- path confinement roots -----------------------------------------
    def _allowed_roots(self) -> list[str]:
        roots = [self._srcdir]
        for mount in self._config.rag_external_mounts:
            path = mount.get("path") if isinstance(mount, dict) else None
            if path:
                roots.append(path)
        return [r for r in roots if r]

    # -- event: builder-inited ------------------------------------------
    def builder_inited(self) -> None:
        self._cache = self._cache or BuildCache(self._config.rag_cache_db)
        # Start loopback servers for embedded stores and rewrite the engine's
        # targets to point at the loopback HTTP endpoint (single writer: the
        # server process owns the on-disk store).
        shared_uri = self._maybe_serve(self._config.shared_target(), "shared")
        private_uri = self._maybe_serve(self._config.local_target(), "private")
        if self._engine is None:
            self._engine = IngestionEngine(self._config)
        self._loopback_uris = {"shared": shared_uri, "private": private_uri}
        self._engine.open_stores(
            shared_loopback_uri=shared_uri, private_loopback_uri=private_uri
        )
        self._maybe_serve_embed()
        self._serve_mount_sidecars()
        self._scan_mounts()

    def _serve_mount_sidecars(self) -> None:
        """Start loopback servers over each mount's portable sidecar store.

        Makes newly-configured vaults queryable by the P6 JS client (which is
        HTTP-only) the same way shared/private are. Mounts without a built
        sidecar are skipped (they still get ingested into the canonical stores
        by ``_scan_mounts``; the sidecar is the fast/portable path). The
        servers intentionally OUTLIVE the build (see :meth:`_close`).
        """
        if not self._start_servers:
            return
        try:
            from doxtr_rag.mount_index import serve_all_mounts

            uris, servers = serve_all_mounts(self._config)
            self._mount_uris = uris
            self._servers.extend(servers)
        except Exception as exc:  # noqa: BLE001 - never crash the build
            logger.warning("doxtr-rag: could not serve mount sidecars: %s", exc)

    def _maybe_serve_embed(self) -> None:
        """Auto-start the bge-m3 embed server so the query side needs no manual
        endpoint (Option E). It serves the SAME provider the engine uses for
        ingestion (embedding parity). Records its URL for endpoints.json. Skips
        gracefully if disabled or the provider can't embed (e.g. weights
        absent) — the query side can still use an external DOXTR_RAG_EMBED_URL.
        """
        if not self._start_servers or not self._serve_embed or self._engine is None:
            return
        provider = self._engine.provider
        port = self._config.embed_port
        # Reuse an embed server already listening on the stable port (e.g. from
        # the standalone launcher or a prior build) rather than binding again.
        from doxtr_rag.server import port_is_serving

        if port_is_serving(port):
            self._embed_uri = f"http://127.0.0.1:{port}/embed"
            logger.info(
                "[doxtr-rag] reusing running embed server on %s", self._embed_uri
            )
            return
        try:
            # Probe once so a weights-absent provider skips cleanly rather than
            # starting a server that errors on every request.
            provider.embed(["probe"])
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "[doxtr-rag] embed server not started (%s); run "
                "'doxtr-rag fetch-weights' or set DOXTR_RAG_EMBED_URL.",
                type(exc).__name__,
            )
            return
        try:
            from doxtr_rag.embed_server import EmbedServer

            server = EmbedServer(host="127.0.0.1", port=port, provider=provider)
            server.start()
            self._servers.append(server)
            self._embed_uri = server.uri
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("doxtr-rag: could not start embed server: %s", exc)

    def _maybe_serve(self, target: StoreTarget, scope_key: str) -> Optional[str]:
        """Start (or reuse) a loopback Chroma server for an embedded store.

        Binds the stable configured port for the scope and reuses an
        already-running server on that port rather than spawning a duplicate,
        so endpoints.json stays stable across builds.
        """
        if target.mode != "embedded" or not self._start_servers:
            return None
        port = (
            self._config.shared_port
            if scope_key == "shared"
            else self._config.private_port
        )
        try:
            from doxtr_rag.server import LoopbackChromaServer

            server = LoopbackChromaServer(target.path or ".doxtr/chroma", port=port)
            server.start(reuse_if_running=True)
            self._servers.append(server)
            if server.reused:
                logger.info(
                    "[doxtr-rag] reusing running %s Chroma server on %s",
                    scope_key,
                    server.uri,
                )
            return server.uri
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "doxtr-rag: could not start loopback server for %s store: %s",
                scope_key,
                exc,
            )
            return None

    def _scan_mounts(self) -> None:
        for mount in self._config.rag_external_mounts:
            if not isinstance(mount, dict):
                continue
            path = mount.get("path")
            if not path or not os.path.isdir(path):
                continue
            scope = self._config.mount_scope(mount)
            self._scan_one_mount(path, scope)

    def _scan_one_mount(self, path: str, scope: AccessControl) -> None:
        """Scan one external mount, reusing a per-mount sidecar cache.

        The sidecar cache (``<mount>/.doxtr/mount-cache.db``, written by
        ``doxtr-rag mount-index`` / ``mount-watch``) records each file's
        SHA-256. When present, a file whose hash is unchanged is NOT re-read or
        re-parsed here: its previously-recorded chunk ids are looked up in the
        build cache and the resource is registered as unchanged so
        ``_ingest_dirty_resources`` skips re-embedding it. This avoids
        re-scanning hundreds of rarely-changing mount files on every build.

        Only files that are new or whose hash changed are extracted. If the
        sidecar is absent, the behaviour degrades to a full scan (the original
        semantics), so nothing breaks without a sidecar.
        """
        from doxtr_rag.extractors import _sha256_file, extract_file
        from doxtr_rag.mount_index import (
            iter_supported_files,
            resource_id_for_file,
            sidecar_cache_path,
        )

        side_cache = None
        side_path = sidecar_cache_path(path)
        if os.path.isfile(side_path):
            try:
                side_cache = BuildCache(side_path)
            except Exception as exc:  # noqa: BLE001 - degrade to full scan
                logger.info(
                    "[doxtr-rag] sidecar cache unreadable for %s (%s); full scan",
                    path,
                    exc,
                )
                side_cache = None

        try:
            for fpath in iter_supported_files(path):
                rid = resource_id_for_file(fpath)
                try:
                    sha = _sha256_file(fpath)
                except OSError:
                    sha = ""
                # Fast path: sidecar says this file is unchanged AND the build
                # cache already has it at the same hash -> register as-is (no
                # re-parse, no re-embed) so the vector store keeps its chunks.
                if (
                    side_cache is not None
                    and not side_cache.is_dirty(rid, sha)
                    and self._cache is not None
                    and not self._cache.is_dirty(rid, sha)
                ):
                    self._seen_resource_ids.add(rid)
                    continue
                chunks = list(
                    extract_file(
                        fpath,
                        scope,
                        max_chunk_chars=self._config.rag_max_chunk_chars,
                        chunk_overlap_chars=self._config.rag_chunk_overlap_chars,
                    )
                )
                if chunks:
                    self._add_resource(rid, "external_file", sha, chunks)
        finally:
            if side_cache is not None:
                try:
                    side_cache.close()
                except Exception:  # noqa: BLE001
                    pass

    # -- event: doctree-resolved ----------------------------------------
    def doctree_resolved(self, doctree: Any, docname: str) -> None:
        from doxtr_rag.harvester import harvest_document, make_sphinx_resolver

        resolver = None
        if self._app is not None and getattr(self._app, "env", None) is not None:
            resolver = make_sphinx_resolver(self._app, self._app.env)
        source_path = self._docname_to_path(docname)
        rid, sha, chunks = harvest_document(
            doctree,
            docname=docname,
            source_path=source_path,
            allowed_roots=self._allowed_roots(),
            resolver=resolver,
        )
        self._add_resource(rid, "node", sha, chunks)

    def _add_resource(
        self, rid: str, kind: str, sha: str, chunks: list[DocumentChunk]
    ) -> None:
        if not chunks:
            # A resource that now produces NO chunks (e.g. emptied doc) is left
            # OUT of the seen set so prune_deleted removes its stale vectors.
            return
        self._seen_resource_ids.add(rid)
        self._resources[rid] = _Resource(rid, kind, sha, chunks)

    def _docname_to_path(self, docname: str) -> str:
        return os.path.join(self._srcdir, f"{docname}.rst")

    def classify_xlinks(self, urls: dict[str, str]) -> None:
        """Classify resolved xlink targets (P4) and buffer them as xlink resources.

        ``urls`` maps xlink unique-id -> resolved URL. Each is classified via the
        unified :func:`doxtr_rag.security.classify_xlink`; the resulting
        provisional API scope is recorded so it is folded into
        ``most_restrictive`` for any chunk that references the xlink, and a
        restricted routing is audited (C7). Fail-closed and non-crashing.
        """
        if not urls:
            return
        from doxtr_rag.audit import AuditLog
        from doxtr_rag.security import NetworkBudget, classify_xlink

        budget = NetworkBudget()
        audit = AuditLog(self._config.rag_audit_log_path)
        for xid, url in urls.items():
            try:
                result = classify_xlink(url, resource_id=xid, budget=budget)
            except Exception as exc:  # noqa: BLE001 - fail closed
                logger.warning("doxtr-rag: xlink classify error for %s: %s", xid, exc)
                continue
            self._xlink_scopes[xid] = result.scope
            if result.scope is AccessControl.LOCAL_PRIVATE:
                audit.record(
                    resource_id=xid,
                    url=url,
                    reason=result.reason,
                    resolved_scope=result.scope,
                )

    # -- event: build-finished ------------------------------------------
    def build_finished(self, exception: Exception | None) -> None:
        try:
            if exception is not None:
                return  # failed build: do not mutate the store
            if self._engine is None:
                return
            self._ingest_dirty_resources()
            self._prune_stale()
            self._write_endpoints()
        finally:
            self._close()

    def _ingest_dirty_resources(self) -> None:
        """Cache-aware upsert: skip unchanged resources; record after upsert.

        For a changed resource, the delta of previously-cached chunk ids that
        are no longer produced is deleted from the stores first, so a resource
        whose chunk-id set shrinks does not leave orphan vectors.
        """
        assert self._engine is not None
        for res in self._resources.values():
            if self._cache is not None and not self._cache.is_dirty(
                res.resource_id, res.sha256
            ):
                continue  # unchanged -> skip re-embedding/upsert
            new_ids = {c.id for c in res.chunks}
            if self._cache is not None:
                stale = [
                    cid
                    for cid in self._cache.chunk_ids_for(res.resource_id)
                    if cid not in new_ids
                ]
                if stale:
                    self._engine.prune(stale)
            extra = {c.id: res.extra_scopes for c in res.chunks} if res.extra_scopes else None
            counts = self._engine.ingest(res.chunks, extra_scopes=extra)
            logger.info("[doxtr-rag] %s: %s", res.resource_id, counts)
            if self._cache is not None:
                self._cache.update_cache(
                    res.resource_id,
                    res.kind,  # type: ignore[arg-type]
                    res.sha256,
                    [c.id for c in res.chunks],
                )

    def _prune_stale(self) -> None:
        """Prune resources (by resource_id) no longer seen this build."""
        assert self._engine is not None
        if self._cache is None:
            return
        stale_chunk_ids = self._cache.prune_deleted(self._seen_resource_ids)
        self._engine.prune(stale_chunk_ids)

    def _write_endpoints(self) -> None:
        """Persist the resolved per-scope endpoints for the P6 TS process."""
        try:
            from doxtr_rag.mount_index import build_mount_endpoints

            mount_descriptors = build_mount_endpoints(self._config, self._mount_uris)
            descriptor = resolve_endpoints(
                self._config,
                shared_query_uri=self._loopback_uris.get("shared"),
                private_query_uri=self._loopback_uris.get("private"),
                embed_uri=self._embed_uri,
                mount_descriptors=mount_descriptors,
            )
            path = os.path.join(
                os.path.dirname(self._config.rag_cache_db) or ".", "endpoints.json"
            )
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(descriptor, fh, indent=2, sort_keys=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("doxtr-rag: could not write endpoints.json: %s", exc)

    def _close(self) -> None:
        """Close per-build resources. Loopback servers intentionally OUTLIVE the
        build: P6 queries the knowledge base at agent time (a separate process)
        against the loopback endpoints recorded in endpoints.json, so the
        servers must keep running after ``build-finished``. Call
        :meth:`shutdown` explicitly to stop them (e.g. at session end / tests).
        """
        if self._cache is not None:
            try:
                self._cache.close()
            except Exception:  # noqa: BLE001
                pass

    def shutdown(self) -> None:
        """Stop any loopback servers started for embedded stores."""
        for server in self._servers:
            try:
                server.stop()
            except Exception:  # noqa: BLE001
                pass
        self._servers = []
