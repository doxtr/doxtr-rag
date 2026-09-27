"""Shared pytest fixtures and a fixture factory for doxtr-rag unit tests.

The factory helpers here are intentionally lightweight so every task's test
module can build the objects it needs without importing peer task code.
"""

from __future__ import annotations

import hashlib
import socket
import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest


def _free_port() -> int:
    """Grab an OS-assigned free loopback port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(autouse=True)
def _isolate_stable_ports(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Give every test unique stable loopback ports.

    The stable-port feature binds fixed ports (8787/8788/8789 for the
    shared/private/embed servers, 8790+ for mount sidecars) and reuses an
    already-running server on a port. Across tests that each start a real
    loopback ``chroma run`` server, a server left alive by one test would be
    reused by a later test pointing at a now-deleted temp path, causing
    "readonly database" write failures. Assigning per-test unique ports (via
    the documented env overrides) keeps each test's servers independent, and
    the teardown reaps any ``chroma run`` subprocess still bound to them.
    """
    ports = {
        "DOXTR_RAG_SHARED_PORT": _free_port(),
        "DOXTR_RAG_PRIVATE_PORT": _free_port(),
        "DOXTR_RAG_EMBED_PORT": _free_port(),
        "DOXTR_RAG_MOUNT_PORT_BASE": _free_port(),
    }
    for var, port in ports.items():
        monkeypatch.setenv(var, str(port))
    yield
    # Reap any loopback chroma server this test left bound to its ports so it
    # cannot be reused by a later test against a deleted temp store path.
    for port in ports.values():
        try:
            out = subprocess.run(
                ["bash", "-c", f"pkill -f 'chroma run.*--port {port}'"],
                capture_output=True,
                timeout=5,
            )
            _ = out
        except Exception:  # noqa: BLE001 - best-effort cleanup only
            pass


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
