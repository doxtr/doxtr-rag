"""Configuration parsing and store resolution (CONTRACTS §C5, §C6).

Parses the ``conf.py`` keys registered by F0 into a typed :class:`SphinxRAGConfig`
and resolves each store (shared / private) into a normalized
:class:`StoreTarget`. The ``writable`` flag on a target is derived from a
**runtime capability probe** of the actual mount/endpoint — not from the mode
(CONTRACTS §C5 invariant, requirement #22).

Connection secrets for ``http`` stores come from the environment, never from
the URI; resolved URIs are scrubbed before any logging.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from doxtr_rag.base_store import (
    DEFAULT_PRIVATE_DATABASE,
    DEFAULT_SHARED_DATABASE,
    DEFAULT_TENANT,
    EMBEDDING_DIMENSION,
)
from doxtr_rag.schema import AccessControl
from doxtr_rag.scrub import scrub

StoreMode = Literal["embedded", "http"]

#: Pinned embedding model + revision (CONVENTIONS §Embedding Model Lifecycle).
DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-m3"
DEFAULT_EMBEDDING_REVISION: str = "bge-m3"

#: Environment variable holding HTTP store auth headers as ``k=v;k=v`` pairs.
#: Split per store so shared/private can carry different credentials.
_SHARED_HEADERS_ENV = "DOXTR_RAG_SHARED_STORE_HEADERS"
_PRIVATE_HEADERS_ENV = "DOXTR_RAG_LOCAL_STORE_HEADERS"


def _parse_headers_env(var: str) -> dict[str, str]:
    """Parse ``k=v;k=v`` header pairs from an env var (empty if unset)."""
    raw = os.environ.get(var, "")
    headers: dict[str, str] = {}
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        if key:
            headers[key] = value
    return headers


def _probe_embedded_writable(path: str) -> bool:
    """Return True if ``path`` (its parent if absent) is writable on disk."""
    target = path
    while target and not os.path.exists(target):
        parent = os.path.dirname(os.path.abspath(target))
        if parent == target:
            break
        target = parent
    check = target or os.getcwd()
    return os.access(check, os.W_OK)


class StoreTarget(BaseModel):
    """Normalized resolution of one store (CONTRACTS §C5).

    Consumed by P5 (Python ingestion) and P6 (TS query). ``writable`` is
    derived from a runtime probe, NOT from ``mode``.
    """

    model_config = ConfigDict(extra="forbid")

    mode: StoreMode
    uri: Optional[str] = None
    path: Optional[str] = None
    tenant: str = DEFAULT_TENANT
    database: str
    ssl: bool = False
    headers: dict[str, str] = Field(default_factory=dict)
    writable: bool = False

    @property
    def scrubbed_uri(self) -> Optional[str]:
        """The URI with any embedded credentials scrubbed (for logging)."""
        return scrub(self.uri) if self.uri is not None else None

    @property
    def scrubbed_headers(self) -> dict[str, str]:
        """Header values scrubbed for logging. The stored ``headers`` retain
        the real env-sourced secrets P5/P6 need to authenticate; only this
        view is safe to log (mirrors :attr:`scrubbed_uri`)."""
        return {k: scrub(v) for k, v in self.headers.items()}


class SphinxRAGConfig(BaseModel):
    """Typed view of the ``conf.py`` RAG settings + derived embedding parity."""

    model_config = ConfigDict(extra="forbid")

    # Original six keys.
    rag_shared_store_uri: Optional[str] = None
    rag_local_store_path: str = ".doxtr/chroma"
    rag_external_mounts: list[dict[str, Any]] = Field(default_factory=list)
    rag_cache_db: str = ".doxtr/cache.db"
    rag_audit_log_path: str = ".doxtr/security_audit.jsonl"
    rag_embedding_provider: str = DEFAULT_EMBEDDING_REVISION

    # ChromaDB deployment keys.
    rag_shared_store_mode: StoreMode = "http"
    rag_local_store_mode: StoreMode = "embedded"
    rag_shared_store_database: str = DEFAULT_SHARED_DATABASE
    rag_local_store_database: str = DEFAULT_PRIVATE_DATABASE
    rag_store_tenant: str = DEFAULT_TENANT
    rag_local_server_uri: Optional[str] = None

    # Chunk-capping keys. Bound the size of emitted chunks (in characters, a
    # dependency-free proxy for tokens) so a boundary-sparse document cannot
    # collapse into one oversized chunk. ``rag_max_chunk_chars <= 0`` disables
    # capping. The env vars DOXTR_RAG_MAX_CHUNK_CHARS /
    # DOXTR_RAG_CHUNK_OVERLAP_CHARS override these when set.
    rag_max_chunk_chars: int = 2000
    rag_chunk_overlap_chars: int = 300

    # Stable loopback ports for the query servers. Both the Sphinx build
    # (``builder-inited``) and the standalone ``doxtr-rag-serve`` launcher bind
    # these SAME ports, and the build reuses an already-running server on a
    # port rather than spawning a new one on an OS-assigned random port. This
    # keeps endpoints.json stable across builds/restarts so the query side
    # (P6) never talks to a dead port. They sit BELOW the Linux ephemeral
    # range (32768-60999) so transient ``chroma run`` subprocesses cannot
    # collide. Env vars DOXTR_RAG_SHARED_PORT / DOXTR_RAG_PRIVATE_PORT /
    # DOXTR_RAG_EMBED_PORT override these when set.
    rag_shared_port: int = 8787
    rag_private_port: int = 8788
    rag_embed_port: int = 8789

    # Derived, exposed for parity enforcement.
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int = EMBEDDING_DIMENSION

    @field_validator("embedding_dimension")
    @classmethod
    def _dimension_is_bge_m3(cls, dim: int) -> int:
        if dim != EMBEDDING_DIMENSION:
            raise ValueError(
                f"embedding_dimension must be {EMBEDDING_DIMENSION} for bge-m3, got {dim}"
            )
        return dim

    @field_validator("rag_shared_store_mode", "rag_local_store_mode")
    @classmethod
    def _valid_mode(cls, mode: str) -> str:
        if mode not in ("embedded", "http"):
            raise ValueError(f"store mode must be 'embedded' or 'http', got {mode!r}")
        return mode

    @property
    def model_revision(self) -> str:
        """The canonical model-revision token for collection naming (C4.1).

        This is the SINGLE source P5 (Python) and P6 (TS) must both feed into
        ``collection_name``/``address_for`` so they never pick different
        tokens. It is the pinned bge-m3 revision, not the raw provider string.
        """
        return DEFAULT_EMBEDDING_REVISION

    # ------------------------------------------------------------------
    # Stable server ports (env override > conf.py value)
    # ------------------------------------------------------------------
    @staticmethod
    def _port_env(var: str, fallback: int) -> int:
        raw = os.environ.get(var)
        if raw is None or raw.strip() == "":
            return fallback
        try:
            return int(raw)
        except ValueError:
            return fallback

    @property
    def shared_port(self) -> int:
        """Effective stable port for the shared loopback Chroma server."""
        return self._port_env("DOXTR_RAG_SHARED_PORT", self.rag_shared_port)

    @property
    def private_port(self) -> int:
        """Effective stable port for the private loopback Chroma server."""
        return self._port_env("DOXTR_RAG_PRIVATE_PORT", self.rag_private_port)

    @property
    def embed_port(self) -> int:
        """Effective stable port for the bge-m3 embed server."""
        return self._port_env("DOXTR_RAG_EMBED_PORT", self.rag_embed_port)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "SphinxRAGConfig":
        """Build a config from a plain mapping (e.g. Sphinx ``app.config``).

        Only known keys are read; unknown keys are ignored. Missing keys fall
        back to typed defaults. Raises ``pydantic.ValidationError`` on wrong
        types or an invalid embedding dimension.
        """
        known = set(cls.model_fields)
        data = {k: v for k, v in mapping.items() if k in known}
        return cls(**data)

    def mount_scope(self, mount: dict[str, Any]) -> AccessControl:
        """Return the declared scope for a mount, fail-closed to LOCAL_PRIVATE.

        A mount is ``SHARED_ORG`` only when it *affirmatively* declares
        ``{"scope": "SHARED_ORG"}`` (or ``access_control``). Anything else —
        missing, malformed, or ``LOCAL_PRIVATE`` — is private (CONVENTIONS
        §External mount default).
        """
        declared = mount.get("scope", mount.get("access_control"))
        return AccessControl.coerce(declared)

    # ------------------------------------------------------------------
    # Store resolution
    # ------------------------------------------------------------------
    def _resolve(
        self,
        *,
        mode: StoreMode,
        uri: Optional[str],
        path: Optional[str],
        database: str,
        headers_env: str,
    ) -> StoreTarget:
        if mode == "http":
            if not uri or "://" not in uri:
                raise ValueError(
                    f"http store requires a scheme-qualified URI, got {uri!r}"
                )
            ssl = uri.lower().startswith("https")
            headers = _parse_headers_env(headers_env)
            return StoreTarget(
                mode="http",
                uri=uri,
                path=None,
                tenant=self.rag_store_tenant,
                database=database,
                ssl=ssl,
                headers=headers,
                writable=_probe_http_writable(
                    uri, ssl, headers, database, self.rag_store_tenant
                ),
            )
        # embedded
        if not path:
            raise ValueError("embedded store requires a filesystem path")
        return StoreTarget(
            mode="embedded",
            uri=None,
            path=path,
            tenant=self.rag_store_tenant,
            database=database,
            ssl=False,
            headers={},
            writable=_probe_embedded_writable(path),
        )

    def shared_target(self) -> StoreTarget:
        """Resolve the SHARED_ORG store target (writability from runtime probe)."""
        return self._resolve(
            mode=self.rag_shared_store_mode,
            uri=self.rag_shared_store_uri,
            path=self.rag_shared_store_uri
            if self.rag_shared_store_mode == "embedded"
            else None,
            database=self.rag_shared_store_database,
            headers_env=_SHARED_HEADERS_ENV,
        )

    def local_target(self) -> StoreTarget:
        """Resolve the LOCAL_PRIVATE store target."""
        if self.rag_local_store_mode == "http":
            uri = self.rag_local_server_uri
        else:
            uri = None
        return self._resolve(
            mode=self.rag_local_store_mode,
            uri=uri,
            path=self.rag_local_store_path
            if self.rag_local_store_mode == "embedded"
            else None,
            database=self.rag_local_store_database,
            headers_env=_PRIVATE_HEADERS_ENV,
        )


def _probe_http_writable(
    uri: str, ssl: bool, headers: dict[str, str], database: str, tenant: str
) -> bool:
    """Probe an HTTP Chroma endpoint for genuine WRITE capability (fail-safe).

    Liveness alone is NOT treated as writability: a reachable but read-only
    server (or a read-only mount behind it) must resolve to ``writable=False``
    so a shared store is never written to when it should not be. The probe
    therefore attempts a real, non-destructive write — create a throwaway probe
    collection in the target database and immediately delete it. Success (both
    create and delete accepted) → writable; any error (unreachable, timeout,
    401/403, read-only, 4xx/5xx) fails **closed** to non-writable and never
    raises.

    An operator may also force the answer without a live probe by setting
    ``DOXTR_RAG_ASSUME_WRITABLE`` to ``0``/``1`` (useful for CI and for servers
    that reject collection churn).
    """
    forced = os.environ.get("DOXTR_RAG_ASSUME_WRITABLE")
    if forced in ("0", "false", "False"):
        return False
    if forced in ("1", "true", "True"):
        return True
    try:
        base = uri.rstrip("/")
        timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=10.0)
        with httpx.Client(timeout=timeout, headers=headers, verify=ssl) as client:
            # 1. Liveness: a live server is a precondition (not sufficient).
            hb = client.get(f"{base}/api/v2/heartbeat")
            if hb.status_code != 200:
                return False
            # 2. Genuine write probe: create + delete a throwaway collection
            #    in the resolved database. Uses the v2 tenant/database path.
            probe_name = "doxtr_rag_write_probe"
            create = client.post(
                f"{base}/api/v2/tenants/{tenant}/databases/{database}/collections",
                json={"name": probe_name, "get_or_create": True},
            )
            if create.status_code not in (200, 201):
                return False
            client.delete(
                f"{base}/api/v2/tenants/{tenant}/databases/{database}/collections/{probe_name}"
            )
            return True
    except Exception:  # noqa: BLE001 - fail-closed, never raise from a probe
        return False
