"""F0 tests for the Sphinx event lifecycle dispatch (graceful degradation)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from doxtr_rag import _lifecycle


class _Env:
    pass


class _App:
    def __init__(self) -> None:
        self.env = _Env()


def test_builder_inited_noop_when_pipeline_absent(monkeypatch: Any) -> None:
    # No doxtr_rag.pipeline module exists yet at F0 -> ImportError path.
    app = _App()
    _lifecycle.on_builder_inited(app)  # type: ignore[arg-type]
    state = getattr(app, "_doxtr_rag_state")
    assert "pipeline" not in state


def test_doctree_resolved_without_pipeline_is_safe() -> None:
    app = _App()
    # No state / pipeline -> must return silently.
    _lifecycle.on_doctree_resolved(app, SimpleNamespace(), "index")  # type: ignore[arg-type]


def test_build_finished_without_pipeline_is_safe() -> None:
    app = _App()
    _lifecycle.on_build_finished(app, None)  # type: ignore[arg-type]


def test_builder_inited_success_path(monkeypatch: Any) -> None:
    import sys
    import types

    calls: list[str] = []

    class _Pipeline:
        @classmethod
        def from_app(cls, app: Any) -> "_Pipeline":
            return cls()

        def builder_inited(self) -> None:
            calls.append("init")

        def doctree_resolved(self, doctree: Any, docname: str) -> None:
            calls.append(f"doc:{docname}")

        def build_finished(self, exception: Exception | None) -> None:
            calls.append("finish")

    fake_mod = types.ModuleType("doxtr_rag.pipeline")
    fake_mod.Pipeline = _Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "doxtr_rag.pipeline", fake_mod)

    app = _App()
    _lifecycle.on_builder_inited(app)  # type: ignore[arg-type]
    _lifecycle.on_doctree_resolved(app, SimpleNamespace(), "index")  # type: ignore[arg-type]
    _lifecycle.on_build_finished(app, None)  # type: ignore[arg-type]
    assert calls == ["init", "doc:index", "finish"]


def test_pipeline_exceptions_are_swallowed(monkeypatch: Any) -> None:
    app = _App()

    class _BadPipeline:
        def builder_inited(self) -> None:
            raise RuntimeError("boom")

        def doctree_resolved(self, doctree: Any, docname: str) -> None:
            raise RuntimeError("boom")

        def build_finished(self, exception: Exception | None) -> None:
            raise RuntimeError("boom")

    state = _lifecycle._get_state(app)  # type: ignore[arg-type]
    state["pipeline"] = _BadPipeline()

    # None of these should raise, per graceful degradation.
    _lifecycle.on_doctree_resolved(app, SimpleNamespace(), "index")  # type: ignore[arg-type]
    _lifecycle.on_build_finished(app, None)  # type: ignore[arg-type]
    assert "pipeline" not in state  # cleared in finally
