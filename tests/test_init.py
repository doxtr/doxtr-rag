"""F0 smoke tests: package import, version, and Sphinx setup() wiring."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

import doxtr_rag


class _FakeSphinx:
    """Minimal Sphinx application double capturing config + event wiring."""

    def __init__(self) -> None:
        self.config_values: dict[str, Any] = {}
        self.connected: dict[str, Any] = {}

    def add_config_value(self, name: str, default: Any, rebuild: str) -> None:
        self.config_values[name] = (default, rebuild)

    def connect(self, event: str, handler: Any) -> None:
        self.connected[event] = handler


def test_version_is_pep440_like() -> None:
    assert isinstance(doxtr_rag.__version__, str)
    parts = doxtr_rag.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])


def test_setup_registers_all_config_keys() -> None:
    app = _FakeSphinx()
    meta = doxtr_rag.setup(app)  # type: ignore[arg-type]

    expected = {
        # six original keys
        "rag_shared_store_uri",
        "rag_local_store_path",
        "rag_external_mounts",
        "rag_cache_db",
        "rag_audit_log_path",
        "rag_embedding_provider",
        # ChromaDB deployment keys
        "rag_shared_store_mode",
        "rag_local_store_mode",
        "rag_shared_store_database",
        "rag_local_store_database",
        "rag_store_tenant",
        "rag_local_server_uri",
    }
    assert expected <= set(app.config_values)
    assert meta["version"] == doxtr_rag.__version__
    assert meta["parallel_read_safe"] is True
    assert meta["parallel_write_safe"] is True


def test_setup_connects_lifecycle_events() -> None:
    app = _FakeSphinx()
    doxtr_rag.setup(app)  # type: ignore[arg-type]
    assert set(app.connected) == {
        "builder-inited",
        "doctree-resolved",
        "build-finished",
    }


def test_registered_handlers_dispatch_to_lifecycle(monkeypatch: Any) -> None:
    """Invoking the connected handlers must route through _lifecycle."""
    from doxtr_rag import _lifecycle

    seen: list[str] = []
    monkeypatch.setattr(
        _lifecycle, "on_builder_inited", lambda app: seen.append("init")
    )
    monkeypatch.setattr(
        _lifecycle,
        "on_doctree_resolved",
        lambda app, doctree, docname: seen.append(f"doc:{docname}"),
    )
    monkeypatch.setattr(
        _lifecycle, "on_build_finished", lambda app, exc: seen.append("finish")
    )

    app = _FakeSphinx()
    doxtr_rag.setup(app)  # type: ignore[arg-type]
    app.connected["builder-inited"](app)
    app.connected["doctree-resolved"](app, object(), "index")
    app.connected["build-finished"](app, None)
    assert seen == ["init", "doc:index", "finish"]


def test_store_mode_defaults_are_valid() -> None:
    app = _FakeSphinx()
    doxtr_rag.setup(app)  # type: ignore[arg-type]
    assert app.config_values["rag_shared_store_mode"][0] in {"embedded", "http"}
    assert app.config_values["rag_local_store_mode"][0] in {"embedded", "http"}
    assert app.config_values["rag_store_tenant"][0] == "default_tenant"
    assert app.config_values["rag_shared_store_database"][0] == "doxtr_shared"
    assert app.config_values["rag_local_store_database"][0] == "doxtr_private"


def test_extension_is_parallel_write_safe_in_real_sphinx(tmp_path: Path) -> None:
    """Regression guard: a real Sphinx app must allow parallel WRITE with the
    extension loaded (no ``not safe for parallel writing`` / serial fallback).

    This is the actual condition that governs the warning the user reported:
    Sphinx calls ``is_parallel_allowed('write')``, which reads
    ``parallel_write_safe`` off the registered extension. If this regresses to
    ``False``, parallel builds silently fall back to slow serial writes.
    """
    sphinx_app = pytest.importorskip("sphinx.application")

    src = tmp_path / "source"
    src.mkdir()
    (src / "conf.py").write_text(
        textwrap.dedent(
            f"""
            extensions = ["doxtr_rag"]
            master_doc = "index"
            rag_local_store_mode = "embedded"
            rag_local_store_path = {str(tmp_path / '.doxtr' / 'chroma')!r}
            rag_shared_store_mode = "embedded"
            rag_shared_store_uri = {str(tmp_path / '.doxtr' / 'shared')!r}
            rag_cache_db = {str(tmp_path / '.doxtr' / 'cache.db')!r}
            rag_audit_log_path = {str(tmp_path / '.doxtr' / 'audit.jsonl')!r}
            """
        ),
        encoding="utf-8",
    )
    (src / "index.rst").write_text("Index\n=====\n\nBody.\n", encoding="utf-8")

    try:
        app = sphinx_app.Sphinx(
            srcdir=str(src),
            confdir=str(src),
            outdir=str(tmp_path / "_build"),
            doctreedir=str(tmp_path / "_doctree"),
            buildername="html",
            parallel=2,
            status=None,
            warning=None,
        )
    except Exception as exc:  # noqa: BLE001 - constrained env (no chroma etc.)
        pytest.skip(f"real Sphinx app unavailable in this environment: {exc}")

    ext = app.extensions["doxtr_rag"]
    assert getattr(ext, "parallel_write_safe") is True
    assert getattr(ext, "parallel_read_safe") is True
    # The decisive check: Sphinx will NOT downgrade to serial write.
    assert app.is_parallel_allowed("write") is True
    assert app.is_parallel_allowed("read") is True
