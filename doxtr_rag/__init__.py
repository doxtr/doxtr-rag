"""doxtr_rag — Multi-tenant, fail-closed Sphinx RAG extension.

This package builds a dual-store (``SHARED_ORG`` vs ``LOCAL_PRIVATE``) ChromaDB
knowledge base from Sphinx documents, mounted multi-format files, and
security-classified Jira/Confluence xlinks. The knowledge base is queried at
agent time by a companion Pi Agent TypeScript extension.

The Sphinx entry point is :func:`setup`, which registers every configuration
value and wires the ``builder-inited`` → ``doctree-resolved`` → ``build-finished``
event lifecycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.1"

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sphinx.application import Sphinx


# ---------------------------------------------------------------------------
# Configuration registration
# ---------------------------------------------------------------------------
#
# Each entry is (name, default, rebuild_trigger). ``rebuild_trigger`` follows
# Sphinx conventions: "env" invalidates the build environment when the value
# changes. These names are the frozen contract in CONTRACTS.md §C6; F1 parses
# them and every producer/consumer reads them by exactly these names.
_CONFIG_VALUES: tuple[tuple[str, Any, str], ...] = (
    # -- The six original conf.py keys -----------------------------------
    ("rag_shared_store_uri", None, "env"),
    ("rag_local_store_path", ".doxtr/chroma", "env"),
    ("rag_external_mounts", [], "env"),
    ("rag_cache_db", ".doxtr/cache.db", "env"),
    ("rag_audit_log_path", ".doxtr/security_audit.jsonl", "env"),
    ("rag_embedding_provider", "bge-m3", "env"),
    # -- ChromaDB deployment keys ----------------------------------------
    ("rag_shared_store_mode", "http", "env"),   # "embedded" | "http"
    ("rag_local_store_mode", "embedded", "env"),  # "embedded" | "http"
    ("rag_shared_store_database", "doxtr_shared", "env"),
    ("rag_local_store_database", "doxtr_private", "env"),
    ("rag_store_tenant", "default_tenant", "env"),
    ("rag_local_server_uri", None, "env"),
)


def _on_builder_inited(app: "Sphinx") -> None:
    """Load config, resolve mounts, and initialize stores.

    Delegates to the ingestion engine (P5) and mount extractor (P3) when those
    modules are available. Missing modules degrade gracefully so that a partial
    build (e.g. during incremental development) never crashes ``sphinx-build``.
    """
    from doxtr_rag._lifecycle import on_builder_inited

    on_builder_inited(app)


def _on_doctree_resolved(app: "Sphinx", doctree: Any, docname: str) -> None:
    """Harvest → classify → cache-check → buffer chunks for one document."""
    from doxtr_rag._lifecycle import on_doctree_resolved

    on_doctree_resolved(app, doctree, docname)


def _on_build_finished(app: "Sphinx", exception: Exception | None) -> None:
    """Batch upsert → prune deleted → flush audit log → close resources."""
    from doxtr_rag._lifecycle import on_build_finished

    on_build_finished(app, exception)


def setup(app: "Sphinx") -> dict[str, Any]:
    """Register the doxtr-rag Sphinx extension.

    Args:
        app: The Sphinx application instance.

    Returns:
        The extension metadata dict. Both ``parallel_read_safe`` and
        ``parallel_write_safe`` are ``True``. The extension is write-safe under
        ``sphinx-build -j`` because every event it hooks runs in the MAIN
        process: ``builder-inited`` and ``build-finished`` are main-process
        events, and ``doctree-resolved`` is emitted from the write phase's
        RESOLVING step (``BuildEnvironment.get_and_resolve_doctree`` in
        ``sphinx/builders/__init__.py``), which runs in the main process
        *before* any write worker is forked. The extension never hooks
        ``write_doc`` (the only forked callback). ``build-finished`` — also
        main-process — is therefore the SOLE writer to the SQLite cache and
        Chroma store (single writer), so no concurrent writes occur. Verified
        against Sphinx 9.1; re-verify the RESOLVING-phase emission point on a
        major Sphinx upgrade (see CONVENTIONS §Concurrency).
    """
    for name, default, rebuild in _CONFIG_VALUES:
        app.add_config_value(name, default, rebuild)

    app.connect("builder-inited", _on_builder_inited)
    app.connect("doctree-resolved", _on_doctree_resolved)
    app.connect("build-finished", _on_build_finished)

    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
