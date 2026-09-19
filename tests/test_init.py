"""F0 smoke tests: package import, version, and Sphinx setup() wiring."""

from __future__ import annotations

from typing import Any

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
    assert meta["parallel_write_safe"] is False


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
