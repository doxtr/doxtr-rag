"""Shared pytest fixtures and a fixture factory for doxtr-rag unit tests.

The factory helpers here are intentionally lightweight so every task's test
module can build the objects it needs without importing peer task code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import pytest


@pytest.fixture
def tmp_doxtr(tmp_path: Path) -> Path:
    """Return a temporary ``.doxtr/`` directory (chroma store, cache, audit)."""
    d = tmp_path / ".doxtr"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def sha256_of() -> Callable[[str | bytes], str]:
    """Return a helper that computes a SHA-256 hex digest of str/bytes."""

    def _hash(value: str | bytes) -> str:
        data = value.encode("utf-8") if isinstance(value, str) else value
        return hashlib.sha256(data).hexdigest()

    return _hash


@pytest.fixture
def chunk_metadata_factory() -> Callable[..., dict[str, Any]]:
    """Return a factory building minimal valid DocumentChunk metadata dicts.

    Producers (P2/P3/P4) and consumers (P5/P6) can override individual keys.
    The default provenance uses the ``{docname, anchor}`` shape (C2).
    """

    def _make(**overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "source_file": "source/index.rst",
            "section_title": "Introduction",
            "breadcrumb": ["Docs", "Introduction"],
            "access_scope": "LOCAL_PRIVATE",
            "content_hash": "0" * 64,
            "provenance": {"docname": "index", "anchor": "introduction"},
        }
        base.update(overrides)
        return base

    return _make
