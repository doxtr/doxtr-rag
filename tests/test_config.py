"""F1 tests for config parsing + StoreTarget resolution (CONTRACTS §C5/§C6)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from doxtr_rag import config as cfg
from doxtr_rag.config import SphinxRAGConfig, StoreTarget
from doxtr_rag.schema import AccessControl


def test_defaults_parse() -> None:
    c = SphinxRAGConfig.from_mapping({})
    assert c.rag_store_tenant == "default_tenant"
    assert c.rag_shared_store_mode == "http"
    assert c.rag_local_store_mode == "embedded"
    assert c.embedding_dimension == 1024
    assert c.embedding_model == "BAAI/bge-m3"


def test_from_mapping_ignores_unknown_keys() -> None:
    c = SphinxRAGConfig.from_mapping({"unknown": 1, "rag_store_tenant": "acme"})
    assert c.rag_store_tenant == "acme"


def test_bad_mode_rejected() -> None:
    with pytest.raises(ValidationError):
        SphinxRAGConfig(rag_shared_store_mode="ftp")  # type: ignore[arg-type]


def test_bad_dimension_rejected() -> None:
    with pytest.raises(ValidationError):
        SphinxRAGConfig(embedding_dimension=768)


def test_wrong_type_rejected() -> None:
    with pytest.raises(ValidationError):
        SphinxRAGConfig(rag_external_mounts="not-a-list")  # type: ignore[arg-type]


def test_mount_scope_failclosed() -> None:
    c = SphinxRAGConfig.from_mapping({})
    assert c.mount_scope({}) is AccessControl.LOCAL_PRIVATE
    assert c.mount_scope({"scope": "LOCAL_PRIVATE"}) is AccessControl.LOCAL_PRIVATE
    assert c.mount_scope({"scope": "bogus"}) is AccessControl.LOCAL_PRIVATE
    assert c.mount_scope({"scope": "SHARED_ORG"}) is AccessControl.SHARED_ORG
    assert c.mount_scope({"access_control": "SHARED_ORG"}) is AccessControl.SHARED_ORG


def test_embedded_target_writable_probe(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "chroma")
    c = SphinxRAGConfig.from_mapping(
        {"rag_local_store_mode": "embedded", "rag_local_store_path": path}
    )
    target = c.local_target()
    assert target.mode == "embedded"
    assert target.path == path
    assert target.database == "doxtr_private"
    assert target.writable is True  # tmp_path is writable


def test_embedded_target_readonly_probe(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "ro" / "chroma")
    monkeypatch.setattr(cfg.os, "access", lambda *a, **k: False)
    c = SphinxRAGConfig.from_mapping(
        {"rag_local_store_mode": "embedded", "rag_local_store_path": path}
    )
    assert c.local_target().writable is False


def test_embedded_requires_path() -> None:
    c = SphinxRAGConfig.from_mapping(
        {"rag_local_store_mode": "embedded", "rag_local_store_path": ""}
    )
    with pytest.raises(ValueError):
        c.local_target()


def test_http_target_requires_scheme() -> None:
    c = SphinxRAGConfig.from_mapping(
        {"rag_shared_store_mode": "http", "rag_shared_store_uri": "not-a-uri"}
    )
    with pytest.raises(ValueError):
        c.shared_target()


def test_http_target_resolves_and_probes(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Force the writability probe to a known value without real network.
    monkeypatch.setattr(cfg, "_probe_http_writable", lambda *a, **k: True)
    c = SphinxRAGConfig.from_mapping(
        {
            "rag_shared_store_mode": "http",
            "rag_shared_store_uri": "https://chroma.example.com:8000",
        }
    )
    t = c.shared_target()
    assert t.mode == "http"
    assert t.ssl is True
    assert t.writable is True
    assert t.database == "doxtr_shared"


def test_http_headers_from_env_retained_and_scrubbed_view(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(
        "DOXTR_RAG_SHARED_STORE_HEADERS", "X-Api=Bearer ghp_" + "a" * 20 + ";X-Env=prod"
    )
    monkeypatch.setattr(cfg, "_probe_http_writable", lambda *a, **k: False)
    c = SphinxRAGConfig.from_mapping(
        {
            "rag_shared_store_mode": "http",
            "rag_shared_store_uri": "http://localhost:8000",
        }
    )
    t = c.shared_target()
    # Stored headers RETAIN the real secret so P5/P6 can authenticate.
    assert "ghp_" in t.headers["X-Api"]
    assert t.headers["X-Env"] == "prod"
    # The scrubbed VIEW is safe to log.
    assert "ghp_" not in t.scrubbed_headers["X-Api"]


def test_scrubbed_uri_property() -> None:
    t = StoreTarget(mode="http", uri="https://u:pw@h/x", database="doxtr_shared")
    assert "pw" not in (t.scrubbed_uri or "")
    t2 = StoreTarget(mode="embedded", path="/x", database="doxtr_private")
    assert t2.scrubbed_uri is None


def test_probe_http_writable_failclosed_on_error() -> None:
    # An unreachable host must fail closed to non-writable and never raise.
    assert cfg._probe_http_writable(
        "http://127.0.0.1:1/nope", False, {}, "doxtr_shared", "default_tenant"
    ) is False


def test_probe_http_writable_forced(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DOXTR_RAG_ASSUME_WRITABLE", "1")
    assert cfg._probe_http_writable("http://h", False, {}, "db", "t") is True
    monkeypatch.setenv("DOXTR_RAG_ASSUME_WRITABLE", "0")
    assert cfg._probe_http_writable("http://h", False, {}, "db", "t") is False


def test_model_revision_is_canonical() -> None:
    c = SphinxRAGConfig.from_mapping({})
    assert c.model_revision == "bge-m3"


def test_local_http_target_uses_server_uri(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cfg, "_probe_http_writable", lambda *a, **k: True)
    c = SphinxRAGConfig.from_mapping(
        {
            "rag_local_store_mode": "http",
            "rag_local_server_uri": "http://localhost:9000",
        }
    )
    t = c.local_target()
    assert t.mode == "http"
    assert t.uri == "http://localhost:9000"
    assert t.database == "doxtr_private"


def test_probe_http_writable_success(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _Resp:
        def __init__(self, code: int) -> None:
            self.status_code = code

    class _Client:
        def __init__(self, *a, **k) -> None:  # type: ignore[no-untyped-def]
            pass

        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, *a) -> None:  # type: ignore[no-untyped-def]
            return None

        def get(self, url, **k):  # type: ignore[no-untyped-def]
            return _Resp(200)

        def post(self, url, **k):  # type: ignore[no-untyped-def]
            return _Resp(201)

        def delete(self, url, **k):  # type: ignore[no-untyped-def]
            return _Resp(200)

    monkeypatch.setattr(cfg.httpx, "Client", _Client)
    assert cfg._probe_http_writable("http://h:8000", False, {}, "db", "t") is True
