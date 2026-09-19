"""Scenario override: external HTTP shared store.

Merged on top of the base ``conf.py`` to exercise the ``http`` shared-store
deployment mode (P5 store resolution). Connection secrets come from env, never
the URI (CONTRACTS §C5).
"""

from __future__ import annotations

rag_shared_store_mode = "http"
rag_shared_store_uri = "http://localhost:8000"
rag_shared_store_database = "doxtr_shared"
rag_store_tenant = "default_tenant"
