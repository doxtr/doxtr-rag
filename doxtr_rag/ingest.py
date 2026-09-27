"""Direct content ingestion for doxtr-rag (``doxtr-rag ingest``).

The Sphinx build is the primary ingestion path (harvested doctree + external
mounts). This module adds an *ad-hoc* ingestion path for a single file or a
single URL, routed to a chosen store:

- ``--ingest-scope shared`` / ``--ingest-scope private`` -> the two canonical
  stores, via the :class:`~doxtr_rag.storage.IngestionEngine` (so scope
  reconciliation and the SHARED_ORG isolation guard apply).
- ``--store <name>`` -> an *independent* named database, co-located with the
  private embedded store on disk and always writable. Created on demand. Kept
  separate from the shared/private isolation model on purpose: a custom store
  is neither the org-shared nor the local-private knowledge base, so it is not
  subject to (or protected by) their guards.

An xlink has no file, only a URL. ``--url`` fetches the URL, extracts it with
the extension-appropriate extractor (defaulting to HTML), and prepends a small
**provenance chunk** carrying the source URL + tags so a later search both
matches the content and surfaces the originating link. ``--source-url`` /
``--tags`` attach the same provenance to a file ingest.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import socket
import tempfile
from typing import Any, Optional

from doxtr_rag.config import SphinxRAGConfig
from doxtr_rag.schema import AccessControl, DocumentChunk

logger = logging.getLogger(__name__)

#: URL suffixes that do NOT map cleanly to an extractor by extension alone but
#: we still want to recognise; everything else defaults to HTML. The concrete
#: extension set is read from the extractor registry at call time (see
#: :func:`_url_extension`) so a child-registered extractor is honoured on the
#: URL-ingest path too.
_URL_EXT_FALLBACK = ".html"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _url_extension(url: str) -> str:
    """Best-effort file extension for a URL; default to ``.html``.

    Candidate extensions come from the public
    :func:`doxtr_rag.extractors.supported_extensions` (longest suffix first, so
    e.g. ``.markdown`` wins over ``.md``), so any extractor a child theme
    registers is selectable here without editing this module.
    """
    from doxtr_rag.extractors import supported_extensions

    path = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    lower = path.lower()
    for ext in sorted(supported_extensions(), key=len, reverse=True):
        if lower.endswith(ext):
            return ext
    return _URL_EXT_FALLBACK


def _resolved_ips(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve a hostname to the IPs it would connect to (best-effort)."""
    # A bracketed/plain IP literal resolves to itself.
    try:
        return [ipaddress.ip_address(host.strip("[]"))]
    except ValueError:
        pass
    infos = socket.getaddrinfo(host, None)
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        sockaddr = info[4]
        try:
            ips.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    return ips


def _assert_public_url(url: str) -> None:
    """Fail closed on a URL that resolves to a non-public address (SSRF guard).

    ``ingest --url`` fetches a caller-supplied URL and follows redirects, so a
    public URL could 30x-redirect to a loopback/link-local/private endpoint
    (e.g. the cloud metadata service ``169.254.169.254`` or the local Chroma
    servers). Every hop is validated: the scheme must be ``http``/``https`` and
    every resolved IP must be global (public). Set
    ``DOXTR_RAG_INGEST_INSECURE=1`` to bypass (last resort, discouraged — same
    toggle that disables TLS verification).
    """
    if os.environ.get("DOXTR_RAG_INGEST_INSECURE") in ("1", "true", "True"):
        return
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"refusing non-http(s) URL: {url!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"refusing URL with no host: {url!r}")
    try:
        ips = _resolved_ips(host)
    except OSError as exc:
        raise ValueError(f"could not resolve host {host!r}: {exc}") from exc
    if not ips:
        raise ValueError(f"could not resolve host {host!r}")
    for ip in ips:
        if not ip.is_global:
            raise ValueError(
                f"refusing URL resolving to non-public address {ip} "
                f"(loopback/link-local/private); set DOXTR_RAG_INGEST_INSECURE=1 "
                f"to override"
            )


def _fetch_url(url: str, timeout: float = 30.0, max_redirects: int = 5) -> tuple[bytes, str]:
    """Fetch a URL, returning (body_bytes, resolved_extension).

    Uses httpx (a doxtr-rag dependency). Credentials, if any, must come from
    the ambient environment/proxy; this never accepts inline secrets. TLS
    trust is resolved from ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` and then
    the system bundle, so it works in containers whose CA store is not the
    Python ``certifi`` bundle.

    Redirects are followed manually (up to ``max_redirects``) and EVERY hop
    — the initial URL and each ``Location`` — is validated by
    :func:`_assert_public_url` to fail closed on an SSRF-via-redirect into a
    loopback/link-local/private endpoint. Set ``DOXTR_RAG_INGEST_INSECURE=1``
    to disable both TLS verification and the SSRF guard (last resort,
    discouraged).
    """
    import httpx

    verify: bool | str = True
    if os.environ.get("DOXTR_RAG_INGEST_INSECURE") in ("1", "true", "True"):
        verify = False
    else:
        for candidate in (
            os.environ.get("SSL_CERT_FILE"),
            os.environ.get("REQUESTS_CA_BUNDLE"),
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/pki/tls/certs/ca-bundle.crt",
        ):
            if candidate and os.path.isfile(candidate):
                verify = candidate
                break

    ext = _url_extension(url)
    current = url
    with httpx.Client(timeout=timeout, follow_redirects=False, verify=verify) as client:
        for _hop in range(max_redirects + 1):
            _assert_public_url(current)
            resp = client.get(current)
            if resp.is_redirect and resp.next_request is not None:
                current = str(resp.next_request.url)
                # Refine the extension hint from the redirect target too.
                ext = _url_extension(current)
                continue
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "").lower()
            # Refine the extension from the content type when the URL had no hint.
            if ext == ".html":
                if "pdf" in ctype:
                    ext = ".pdf"
                elif "json" in ctype or "xml" in ctype:
                    ext = ".xml"
                elif "markdown" in ctype:
                    ext = ".md"
                elif "plain" in ctype:
                    ext = ".txt"
            return resp.content, ext
    raise ValueError(f"too many redirects (> {max_redirects}) fetching {url!r}")


def _provenance_chunk(
    *,
    source_id: str,
    scope: AccessControl,
    source_url: Optional[str],
    tags: Optional[str],
    title: Optional[str],
) -> Optional[DocumentChunk]:
    """Build a small provenance chunk so search surfaces the link + tags.

    Returns None when there is no provenance to record.
    """
    if not source_url and not tags:
        return None
    lines = []
    if title:
        lines.append(f"Title: {title}")
    if source_url:
        lines.append(f"Source URL: {source_url}")
    if tags:
        lines.append(f"Tags: {tags}")
    body = "\n".join(lines)
    return DocumentChunk(
        id=f"{source_id}#provenance",
        text=body,
        metadata={
            "source_file": source_url or source_id,
            "section_title": title or "Source",
            "breadcrumb": [title] if title else ["Source"],
            "access_scope": scope,
            "content_hash": _sha256(body),
            "provenance": {"docname": source_id, "anchor": "provenance"},
        },
    )


def _extract_path(
    path: str,
    scope: AccessControl,
    config: SphinxRAGConfig,
    *,
    logical_source: str,
) -> list[DocumentChunk]:
    """Extract a file into chunks, rewriting ids/source_file to a logical name.

    ``logical_source`` replaces the temp path (for a fetched URL) so chunk ids
    and ``source_file`` metadata are stable and human-meaningful.
    """
    from doxtr_rag.extractors import extract_file

    chunks: list[DocumentChunk] = []
    for chunk in extract_file(
        path,
        scope,
        max_chunk_chars=config.rag_max_chunk_chars,
        chunk_overlap_chars=config.rag_chunk_overlap_chars,
    ):
        new_id = chunk.id.replace(path, logical_source, 1)
        meta = dict(chunk.metadata)
        if meta.get("source_file") == path:
            meta["source_file"] = logical_source
        crumb = meta.get("breadcrumb")
        if isinstance(crumb, list) and crumb and crumb[0] == os.path.basename(path):
            crumb = [os.path.basename(logical_source)] + crumb[1:]
            meta["breadcrumb"] = crumb
        chunks.append(DocumentChunk(id=new_id, text=chunk.text, metadata=meta))
    return chunks


# ---------------------------------------------------------------------------
# Custom (independent) store
# ---------------------------------------------------------------------------
def _custom_store(config: SphinxRAGConfig, name: str) -> Any:
    """Build a writable ChromaVectorStore for an independent named database.

    Co-located with the private embedded store on disk (or the private HTTP
    server), under its own ``database=name`` so it is fully isolated from the
    shared/private collections. Created on demand. Construction goes through the
    single public :func:`doxtr_rag.storage.open_named_store` factory (shared
    with the per-mount sidecars) so a child theme can swap the store backend in
    one place. A custom store is not the org-shared store; it is opened as
    ``LOCAL_PRIVATE`` (the SHARED_ORG isolation guard only fires for a store
    opened as shared).

    NOTE (single-writer assumption): an embedded custom store shares the
    on-disk directory the private loopback Chroma server owns during a build
    (only the ``database`` differs). Running ``ingest --store`` concurrently
    with a live build could open a second writer to that path; in practice the
    two are rarely concurrent and use different databases, so this is accepted
    rather than routed through the loopback server. Prefer running ad-hoc
    ingests when no build is active.
    """
    from doxtr_rag.storage import open_named_store

    return open_named_store(
        config,
        database=name,
        scope=AccessControl.LOCAL_PRIVATE,
        writable=True,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def ingest(
    *,
    config: SphinxRAGConfig,
    file: Optional[str] = None,
    url: Optional[str] = None,
    scope: str = "private",
    store: Optional[str] = None,
    source_url: Optional[str] = None,
    tags: Optional[str] = None,
    title: Optional[str] = None,
    batch_size: int = 100,
) -> dict[str, Any]:
    """Ingest one file or URL into the selected store. Returns per-scope counts.

    Exactly one of ``file`` / ``url`` is required. ``store`` (a custom database
    name) takes precedence over ``scope`` when both are given.
    """
    if bool(file) == bool(url):
        raise ValueError("provide exactly one of --file or --url")

    provisional = (
        AccessControl.SHARED_ORG if scope == "shared" else AccessControl.LOCAL_PRIVATE
    )

    # 1. Resolve content into a local path + a stable logical source id.
    tmp_path: Optional[str] = None
    local_path: str
    logical_source: str
    effective_source_url: Optional[str]
    try:
        if url:
            body, ext = _fetch_url(url)
            fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="doxtr_ingest_")
            with os.fdopen(fd, "wb") as fh:
                fh.write(body)
            local_path = tmp_path
            logical_source = url
            effective_source_url = source_url or url
        else:
            assert file is not None
            if not os.path.isfile(file):
                raise FileNotFoundError(f"file not found: {file}")
            local_path = file
            logical_source = os.path.abspath(file)
            effective_source_url = source_url

        # 2. Extract.
        chunks = _extract_path(
            local_path, provisional, config, logical_source=logical_source
        )

        # 3. Prepend a provenance chunk (URL + tags) so search surfaces them.
        prov = _provenance_chunk(
            source_id=logical_source,
            scope=provisional,
            source_url=effective_source_url,
            tags=tags,
            title=title,
        )
        if prov is not None:
            chunks.insert(0, prov)

        if not chunks:
            logger.warning("doxtr-rag: nothing extracted from %s", logical_source)
            return {"ingested": 0}

        # 4. Route to the selected store.
        if store:
            vs = _custom_store(config, store)
            if not vs.writable:
                raise RuntimeError(f"custom store {store!r} is not writable")
            vs.upsert(chunks, batch_size)
            return {"store": store, "ingested": len(chunks)}

        from doxtr_rag.storage import IngestionEngine

        engine = IngestionEngine(config, batch_size=batch_size)
        engine.open_stores()
        counts = engine.ingest(chunks)
        counts["ingested"] = len(chunks)
        return counts
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
