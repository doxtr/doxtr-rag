"""doxtr-rag Test Harness — Base Sphinx Configuration.

This is the BASE configuration for the RAG integration/regression harness. It
mirrors the layout of the ``doxtr-pdf-theme-core`` reference harness: a real
Sphinx project with per-scenario overrides in ``conf_overrides/``.

The harness exercises the doxtr-rag extension end to end against sample
``source/`` (SHARED_ORG candidate), ``noter/`` and ``NDA/`` (always
LOCAL_PRIVATE) content plus mounted multi-format files.

To keep the harness runnable without a live ChromaDB server, scenarios point
the stores at embedded on-disk paths under a scratch ``.doxtr/`` directory.
"""

from __future__ import annotations

import os

# -- Project information -----------------------------------------------------
project = "Doxtr RAG Test Harness"
copyright = "2026, Doc Dash"
author = "Doc Dash"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
master_doc = "index"
extensions = [
    "sphinx.ext.intersphinx",
    "doxtr_rag",
]

exclude_patterns = [
    "_build",
    "conf_overrides",
    "*.md",
    ".doxtr",
]

# -- doxtr-rag configuration (embedded defaults for a self-contained run) ----
_HARNESS_ROOT = os.path.dirname(os.path.abspath(__file__))
_DOXTR_DIR = os.path.join(_HARNESS_ROOT, ".doxtr")

rag_local_store_mode = "embedded"
rag_local_store_path = os.path.join(_DOXTR_DIR, "chroma")
rag_shared_store_mode = "embedded"
rag_shared_store_uri = os.path.join(_DOXTR_DIR, "chroma_shared")
rag_cache_db = os.path.join(_DOXTR_DIR, "cache.db")
rag_audit_log_path = os.path.join(_DOXTR_DIR, "security_audit.jsonl")
rag_embedding_provider = "bge-m3"
rag_store_tenant = "default_tenant"
rag_shared_store_database = "doxtr_shared"
rag_local_store_database = "doxtr_private"

# External mounts: a private mount (default) and an explicitly-shared mount.
rag_external_mounts = [
    {"path": os.path.join(_HARNESS_ROOT, "source", "source")},
    {"path": os.path.join(_HARNESS_ROOT, "source", "noter")},
    {"path": os.path.join(_HARNESS_ROOT, "source", "NDA")},
]

html_theme = "basic"
