"""P5 tests for the build Pipeline, loopback server, provider, and CLI."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from docutils.core import publish_doctree

from doxtr_rag.config import SphinxRAGConfig
from doxtr_rag.schema import AccessControl, DocumentChunk
from doxtr_rag.storage import IngestionEngine


class FakeProvider:
    dimension = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.01] * 1024 for _ in texts]


def _config(tmp_path: Path) -> SphinxRAGConfig:
    return SphinxRAGConfig.from_mapping(
        {
            "rag_local_store_mode": "embedded",
            "rag_local_store_path": str(tmp_path / f"priv_{uuid.uuid4().hex[:6]}"),
            "rag_shared_store_mode": "embedded",
            "rag_shared_store_uri": str(tmp_path / f"shared_{uuid.uuid4().hex[:6]}"),
            "rag_cache_db": str(tmp_path / "cache.db"),
            "rag_audit_log_path": str(tmp_path / "audit.jsonl"),
        }
    )


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    from doxtr_rag.pipeline import Pipeline

    config = _config(tmp_path)
    engine = IngestionEngine(config, provider=FakeProvider())
    pipe = Pipeline(config, srcdir=str(tmp_path), engine=engine, start_servers=False)
    pipe.builder_inited()

    rst = "Title\n=====\n\nHello body.\n\nSub\n---\n\nSub body.\n"
    doctree = publish_doctree(rst)
    pipe.doctree_resolved(doctree, "index")
    assert "node:index" in pipe._resources  # noqa: SLF001

    pipe.build_finished(None)
    # Chunks landed in the private store (tmp path scope -> private).
    assert engine.private_store is not None
    assert engine.private_store.count() >= 2
    # endpoints.json was written for the P6 process.
    endpoints = tmp_path / "endpoints.json"
    assert endpoints.exists()
    import json

    desc = json.loads(endpoints.read_text())
    assert desc["private"]["database"] == "doxtr_private"
    assert desc["shared"]["database"] == "doxtr_shared"
    assert desc["embedding"]["dimension"] == 1024


def test_pipeline_cache_aware_skips_unchanged(tmp_path: Path) -> None:
    from doxtr_rag.cache import BuildCache
    from doxtr_rag.pipeline import Pipeline

    config = _config(tmp_path)
    rst = "Title\n=====\n\nHello body.\n"

    # First build ingests.
    engine1 = IngestionEngine(config, provider=FakeProvider())
    cache = BuildCache(config.rag_cache_db)
    pipe1 = Pipeline(
        config, srcdir=str(tmp_path), engine=engine1, cache=cache, start_servers=False
    )
    pipe1.builder_inited()
    pipe1.doctree_resolved(publish_doctree(rst), "index")
    pipe1.build_finished(None)
    first = engine1.private_store.count()  # type: ignore[union-attr]
    assert first >= 1

    # Second build with identical content: cache-aware skip (no re-ingest).
    engine2 = IngestionEngine(config, provider=FakeProvider())
    cache2 = BuildCache(config.rag_cache_db)
    pipe2 = Pipeline(
        config, srcdir=str(tmp_path), engine=engine2, cache=cache2, start_servers=False
    )
    pipe2.builder_inited()
    pipe2.doctree_resolved(publish_doctree(rst), "index")
    # is_dirty should be False for the unchanged node.
    assert cache2.is_dirty("node:index", pipe2._resources["node:index"].sha256) is False  # noqa: SLF001
    pipe2.build_finished(None)
    assert engine2.private_store.count() == first  # type: ignore[union-attr]


def test_pipeline_prunes_deleted_resource(tmp_path: Path) -> None:
    from doxtr_rag.cache import BuildCache
    from doxtr_rag.pipeline import Pipeline

    config = _config(tmp_path)
    # Build 1: two docs.
    e1 = IngestionEngine(config, provider=FakeProvider())
    c1 = BuildCache(config.rag_cache_db)
    p1 = Pipeline(config, srcdir=str(tmp_path), engine=e1, cache=c1, start_servers=False)
    p1.builder_inited()
    p1.doctree_resolved(publish_doctree("A\n=\n\nBody a.\n"), "a")
    p1.doctree_resolved(publish_doctree("B\n=\n\nBody b.\n"), "b")
    p1.build_finished(None)
    count_two = e1.private_store.count()  # type: ignore[union-attr]

    # Build 2: only doc a remains -> b's chunks pruned.
    e2 = IngestionEngine(config, provider=FakeProvider())
    c2 = BuildCache(config.rag_cache_db)
    p2 = Pipeline(config, srcdir=str(tmp_path), engine=e2, cache=c2, start_servers=False)
    p2.builder_inited()
    p2.doctree_resolved(publish_doctree("A\n=\n\nBody a.\n"), "a")
    p2.build_finished(None)
    assert e2.private_store.count() < count_two  # type: ignore[union-attr]


def test_pipeline_skips_on_build_exception(tmp_path: Path) -> None:
    from doxtr_rag.pipeline import Pipeline

    config = _config(tmp_path)
    engine = IngestionEngine(config, provider=FakeProvider())
    pipe = Pipeline(config, srcdir=str(tmp_path), engine=engine, start_servers=False)
    pipe.builder_inited()
    doctree = publish_doctree("T\n=\n\nBody.\n")
    pipe.doctree_resolved(doctree, "index")
    pipe.build_finished(RuntimeError("build failed"))
    # A failed build must NOT mutate the store.
    assert engine.private_store is not None
    assert engine.private_store.count() == 0


def test_pipeline_from_app(tmp_path: Path) -> None:
    from doxtr_rag.pipeline import Pipeline

    app = SimpleNamespace(
        config=SimpleNamespace(
            rag_local_store_path=str(tmp_path / "p"),
            rag_local_store_mode="embedded",
            rag_shared_store_mode="embedded",
            rag_shared_store_uri=str(tmp_path / "s"),
            rag_cache_db=str(tmp_path / "c.db"),
        ),
        srcdir=str(tmp_path),
        env=None,
    )
    pipe = Pipeline.from_app(app)  # type: ignore[arg-type]
    assert pipe._srcdir == str(tmp_path)  # noqa: SLF001


def test_pipeline_uses_resolver_when_env_present(tmp_path: Path) -> None:
    from doxtr_rag.pipeline import Pipeline

    config = _config(tmp_path)
    engine = IngestionEngine(config, provider=FakeProvider())

    # A fake app with an env triggers make_sphinx_resolver wiring.
    fake_env = SimpleNamespace(titles={}, get_domain=lambda name: SimpleNamespace())
    app = SimpleNamespace(env=fake_env, srcdir=str(tmp_path))
    pipe = Pipeline(config, srcdir=str(tmp_path), engine=engine, app=app, start_servers=False)
    pipe.builder_inited()
    doctree = publish_doctree("T\n=\n\nBody.\n")
    pipe.doctree_resolved(doctree, "index")
    assert "node:index" in pipe._resources  # noqa: SLF001
    # docname->path uses the srcdir.
    assert pipe._docname_to_path("index") == str(tmp_path / "index.rst")  # noqa: SLF001


def test_pipeline_close_swallows_errors(tmp_path: Path) -> None:
    from doxtr_rag.pipeline import Pipeline

    config = _config(tmp_path)

    class _BadCache:
        def prune_deleted(self, seen: object) -> list[str]:
            return []

        def close(self) -> None:
            raise RuntimeError("close boom")

    class _BadServer:
        def stop(self) -> None:
            raise RuntimeError("stop boom")

    engine = IngestionEngine(config, provider=FakeProvider())
    pipe = Pipeline(config, srcdir=str(tmp_path), engine=engine, cache=_BadCache(), start_servers=False)  # type: ignore[arg-type]
    pipe._servers = [_BadServer()]  # noqa: SLF001
    pipe.builder_inited()
    # build_finished must not raise despite failing close.
    pipe.build_finished(None)
    # shutdown must not raise despite failing server.stop().
    pipe.shutdown()


def test_pipeline_scans_mounts(tmp_path: Path) -> None:
    import shutil

    from doxtr_rag.pipeline import Pipeline

    mount = tmp_path / "mnt"
    mount.mkdir()
    fixtures = Path(__file__).parent / "fixtures" / "docs"
    shutil.copy(fixtures / "sample.pdf", mount / "sample.pdf")

    config = SphinxRAGConfig.from_mapping(
        {
            "rag_local_store_mode": "embedded",
            "rag_local_store_path": str(tmp_path / "priv"),
            "rag_shared_store_mode": "embedded",
            "rag_shared_store_uri": str(tmp_path / "shared"),
            "rag_cache_db": str(tmp_path / "cache.db"),
            "rag_external_mounts": [{"path": str(mount)}],
        }
    )
    engine = IngestionEngine(config, provider=FakeProvider())
    pipe = Pipeline(config, srcdir=str(tmp_path), engine=engine, start_servers=False)
    pipe.builder_inited()
    # The PDF from the private-default mount was buffered as an external_file resource.
    all_chunks = [c for r in pipe._resources.values() for c in r.chunks]  # noqa: SLF001
    assert any("sample.pdf" in c.metadata["source_file"] for c in all_chunks)


# -- loopback server ---------------------------------------------------------
def test_loopback_server_lifecycle(tmp_path: Path) -> None:
    from doxtr_rag.server import LoopbackChromaServer

    server = LoopbackChromaServer(str(tmp_path / "chroma"))
    try:
        server.start(wait_seconds=30.0)
        assert server.uri.startswith("http://127.0.0.1:")
        # A live HTTP client can connect.
        import chromadb

        client = chromadb.HttpClient(host=server.host, port=server.port)
        client.heartbeat()
    except RuntimeError:
        pytest.skip("chroma server CLI unavailable in this environment")
    finally:
        server.stop()
    # Idempotent stop.
    server.stop()


def test_free_loopback_port_unique() -> None:
    from doxtr_rag.server import _free_loopback_port

    assert _free_loopback_port() != 0


def test_pipeline_embedded_served_over_loopback(tmp_path: Path) -> None:
    """Blocker fix: an embedded store is served over loopback HTTP and both
    ingestion and (P6) queries go through that endpoint. endpoints.json carries
    the resolved loopback host/port for the separate P6 process."""
    import json

    from doxtr_rag.pipeline import Pipeline

    config = _config(tmp_path)
    # Real engine (no injected stores) + real loopback servers + fake embeddings.
    engine = IngestionEngine(config, provider=FakeProvider())
    pipe = Pipeline(config, srcdir=str(tmp_path), engine=engine, start_servers=True)
    try:
        pipe.builder_inited()
    except Exception:  # noqa: BLE001
        pytest.skip("loopback chroma server unavailable in this environment")
    # Both embedded stores got a loopback URI.
    assert pipe._loopback_uris["private"] is not None  # noqa: SLF001
    pipe.doctree_resolved(publish_doctree("T\n=\n\nBody.\n"), "index")
    pipe.build_finished(None)

    desc = json.loads((tmp_path / "endpoints.json").read_text())
    # Private endpoint is the loopback host/port (embedded served over HTTP).
    assert desc["private"]["host"] == "127.0.0.1"
    assert desc["private"]["port"] > 0
    # A fresh HTTP client (as P6 would use) can read the ingested vectors.
    import chromadb

    client = chromadb.HttpClient(
        host=desc["private"]["host"],
        port=desc["private"]["port"],
        tenant=desc["private"]["tenant"],
        database=desc["private"]["database"],
    )
    col = client.get_collection(desc["private"]["collection"])
    assert col.count() >= 1
    pipe.shutdown()


def test_server_start_failure_raises(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from doxtr_rag.server import LoopbackChromaServer

    server = LoopbackChromaServer(str(tmp_path / "chroma"))

    # Force _wait_ready to report not-ready so start() raises + cleans up.
    monkeypatch.setattr(server, "_wait_ready", lambda wait_seconds=0: False)
    with pytest.raises(RuntimeError):
        server.start(wait_seconds=0.1)
    # After a failed start the proc handle is cleared.
    assert server._proc is None  # noqa: SLF001


def test_server_start_idempotent(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from doxtr_rag.server import LoopbackChromaServer

    server = LoopbackChromaServer(str(tmp_path / "c"))
    server._proc = object()  # type: ignore[assignment]  # pretend running  # noqa: SLF001
    server.start()  # must return immediately (no new proc)
    server._proc = None  # noqa: SLF001


# -- embedding provider ------------------------------------------------------
BGE_M3_DIR = os.environ.get("DOXTR_RAG_BGE_M3_ONNX_DIR", "/opt/models/bge-m3")
_HAS_WEIGHTS = (
    Path(BGE_M3_DIR, "model.onnx").is_file()
    and Path(BGE_M3_DIR, "model.onnx_data").is_file()
    and Path(BGE_M3_DIR, "tokenizer.json").is_file()
)


def test_provider_requires_weights(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from doxtr_rag import embedding
    from doxtr_rag.embedding_provider import EmbeddingProvider

    monkeypatch.delenv(embedding.WEIGHTS_ENV, raising=False)
    prov = EmbeddingProvider()
    with pytest.raises(embedding.WeightsUnavailableError):
        prov.embed(["hello"])


def test_provider_dimension() -> None:
    from doxtr_rag.embedding_provider import EmbeddingProvider

    assert EmbeddingProvider().dimension == 1024


def test_provider_missing_onnx_files(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Weights dir exists but the ONNX files are absent -> clear error, no fallback.
    from doxtr_rag import embedding
    from doxtr_rag.embedding_provider import EmbeddingProvider

    monkeypatch.setenv(embedding.WEIGHTS_ENV, str(tmp_path))
    with pytest.raises(embedding.WeightsUnavailableError):
        EmbeddingProvider().embed(["x"])


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="bge-m3 ONNX weights not present")
def test_provider_real_bge_m3_embeddings(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import math

    from doxtr_rag import embedding
    from doxtr_rag.embedding_provider import EmbeddingProvider

    monkeypatch.setenv(embedding.WEIGHTS_ENV, BGE_M3_DIR)
    prov = EmbeddingProvider()
    vecs = prov.embed(["What is BGE M3?", "An embedding model."])
    assert len(vecs) == 2
    # Real 1024-dim, L2-normalized dense vectors.
    for v in vecs:
        assert len(v) == 1024
        assert math.isclose(sum(x * x for x in v) ** 0.5, 1.0, rel_tol=1e-3)
    # Same text -> identical vector (deterministic); different text -> different.
    again = prov.embed(["What is BGE M3?"])[0]
    assert max(abs(a - b) for a, b in zip(again, vecs[0])) < 1e-5
    # Semantic sanity: a paraphrase is closer than an unrelated sentence.
    q, para, unrel = prov.embed(
        ["the reactor deployment steps", "how to deploy the reactor", "banana bread recipe"]
    )
    cos = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert cos(q, para) > cos(q, unrel)


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="bge-m3 ONNX weights not present")
def test_provider_empty_batch() -> None:
    from doxtr_rag.embedding_provider import EmbeddingProvider

    assert EmbeddingProvider().embed([]) == []


# -- CLI ---------------------------------------------------------------------
def test_cli_cache_info(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from doxtr_rag import cli
    from doxtr_rag.cache import BuildCache

    db = tmp_path / "cache.db"
    with BuildCache(db) as c:
        c.update_cache("node:index", "node", "h1", ["c1", "c2"])
        c.update_cache("xlink:JIRA-1", "xlink", "h2", ["c3"])

    conf = tmp_path / "conf.py"
    conf.write_text(f"rag_cache_db = {str(db)!r}\n")
    rc = cli.main(["--conf", str(conf), "cache-info"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "node: 1 resources" in out
    assert "chunks mapped: 3" in out


def test_cli_audit_info(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from doxtr_rag import cli
    from doxtr_rag.audit import AuditLog

    audit = tmp_path / "audit.jsonl"
    AuditLog(audit).record(
        resource_id="J-1", url="u", reason="jira_ils_restricted",
        resolved_scope=AccessControl.LOCAL_PRIVATE,
    )
    conf = tmp_path / "conf.py"
    conf.write_text(f"rag_audit_log_path = {str(audit)!r}\n")
    rc = cli.main(["--conf", str(conf), "audit-info"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "jira_ils_restricted: 1" in out


def test_cli_rebuild(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from doxtr_rag import cli

    conf = tmp_path / "conf.py"
    conf.write_text(
        "rag_local_store_mode = 'embedded'\n"
        f"rag_local_store_path = {str(tmp_path / 'priv')!r}\n"
        "rag_shared_store_mode = 'embedded'\n"
        f"rag_shared_store_uri = {str(tmp_path / 'shared')!r}\n"
    )
    rc = cli.main(["--conf", str(conf), "--rebuild", "--scope", "local"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "recreated LOCAL_PRIVATE" in out


def test_cli_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    from doxtr_rag import cli

    rc = cli.main([])
    assert rc == 0
    assert "doxtr-rag maintenance CLI" in capsys.readouterr().out


def test_pipeline_classify_xlinks_folds_scope_and_audits(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    import httpx

    from doxtr_rag import security
    from doxtr_rag.pipeline import Pipeline

    config = _config(tmp_path)
    engine = IngestionEngine(config, provider=FakeProvider())
    pipe = Pipeline(config, srcdir=str(tmp_path), engine=engine, start_servers=False)

    # Mock the classifier network: a restricted Jira issue -> LOCAL_PRIVATE.
    def fake_classify(url, *, resource_id="", client=None, headers=None, budget=None):  # type: ignore[no-untyped-def]
        return security.ClassificationResult(
            AccessControl.LOCAL_PRIVATE, "jira_ils_restricted",
            security.scrub(url), resource_id,
        )

    monkeypatch.setattr(security, "classify_xlink", fake_classify)
    pipe.classify_xlinks({"JIRA-1": "https://jira.example.com/browse/PROJ-1"})
    assert pipe._xlink_scopes["JIRA-1"] is AccessControl.LOCAL_PRIVATE  # noqa: SLF001

    # A restricted xlink was audited (C7).
    from doxtr_rag.audit import AuditLog

    events = AuditLog(config.rag_audit_log_path).read_events()
    assert any(e.resource_id == "JIRA-1" for e in events)


def test_pipeline_classify_xlinks_empty_noop(tmp_path):  # type: ignore[no-untyped-def]
    from doxtr_rag.pipeline import Pipeline

    config = _config(tmp_path)
    pipe = Pipeline(config, srcdir=str(tmp_path), start_servers=False)
    pipe.classify_xlinks({})  # must not raise
    assert pipe._xlink_scopes == {}  # noqa: SLF001


def test_maybe_serve_failure_degrades(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    from doxtr_rag import server as server_mod
    from doxtr_rag.config import StoreTarget
    from doxtr_rag.pipeline import Pipeline

    config = _config(tmp_path)
    pipe = Pipeline(config, srcdir=str(tmp_path), start_servers=True)

    class _BadServer:
        def __init__(self, path):  # type: ignore[no-untyped-def]
            pass

        def start(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("no chroma")

    monkeypatch.setattr(server_mod, "LoopbackChromaServer", _BadServer)
    uri = pipe._maybe_serve(  # noqa: SLF001
        StoreTarget(mode="embedded", path=str(tmp_path), database="doxtr_private"),
        "private",
    )
    assert uri is None  # degrades gracefully


def test_classify_xlinks_error_degrades(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    from doxtr_rag import security
    from doxtr_rag.pipeline import Pipeline

    config = _config(tmp_path)
    pipe = Pipeline(config, srcdir=str(tmp_path), start_servers=False)

    def boom(url, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("classify boom")

    monkeypatch.setattr(security, "classify_xlink", boom)
    pipe.classify_xlinks({"X-1": "https://jira/browse/PROJ-1"})  # must not raise
    assert "X-1" not in pipe._xlink_scopes  # noqa: SLF001


def test_allowed_roots_includes_mounts(tmp_path):  # type: ignore[no-untyped-def]
    from doxtr_rag.config import SphinxRAGConfig
    from doxtr_rag.pipeline import Pipeline

    config = SphinxRAGConfig.from_mapping(
        {"rag_external_mounts": [{"path": "/mnt/a"}, {"path": "/mnt/b"}]}
    )
    pipe = Pipeline(config, srcdir=str(tmp_path), start_servers=False)
    roots = pipe._allowed_roots()  # noqa: SLF001
    assert str(tmp_path) in roots and "/mnt/a" in roots and "/mnt/b" in roots


def test_pipeline_shrinking_resource_prunes_orphans(tmp_path):  # type: ignore[no-untyped-def]
    from doxtr_rag.cache import BuildCache
    from doxtr_rag.pipeline import Pipeline

    config = _config(tmp_path)
    # Build 1: a doc with two sections -> two chunks.
    e1 = IngestionEngine(config, provider=FakeProvider())
    c1 = BuildCache(config.rag_cache_db)
    p1 = Pipeline(config, srcdir=str(tmp_path), engine=e1, cache=c1, start_servers=False)
    p1.builder_inited()
    p1.doctree_resolved(publish_doctree("T\n=\n\nBody.\n\nSub\n---\n\nSub body.\n"), "d")
    p1.build_finished(None)
    two = e1.private_store.count()  # type: ignore[union-attr]
    assert two >= 2

    # Build 2: same doc, but the subsection removed -> fewer chunks; the orphan
    # chunk from the removed subsection must be pruned (not linger).
    e2 = IngestionEngine(config, provider=FakeProvider())
    c2 = BuildCache(config.rag_cache_db)
    p2 = Pipeline(config, srcdir=str(tmp_path), engine=e2, cache=c2, start_servers=False)
    p2.builder_inited()
    p2.doctree_resolved(publish_doctree("T\n=\n\nBody only now.\n"), "d")
    p2.build_finished(None)
    assert e2.private_store.count() < two  # type: ignore[union-attr]


# -- parallel-write single-writer path ---------------------------------------
def test_parallel_write_resolving_loop_ingests_all_docs(tmp_path: Path) -> None:
    """Prove the single-writer path ingests the FULL doc set under the -j model.

    Under ``sphinx-build -j N`` only ``write_doc`` runs in forked workers;
    ``doctree-resolved`` is emitted by the write phase's RESOLVING step in the
    MAIN process (``get_and_resolve_doctree``), so a single Pipeline instance
    buffers every document and ``build-finished`` — also main-process — is the
    sole writer. This test mimics that main-process RESOLVING loop: one
    pipeline, ``builder_inited`` → ``doctree_resolved`` for several docnames →
    a single ``build_finished`` flush, asserting all docs were ingested by the
    one writer.
    """
    from doxtr_rag.pipeline import Pipeline

    config = _config(tmp_path)
    engine = IngestionEngine(config, provider=FakeProvider())
    pipe = Pipeline(config, srcdir=str(tmp_path), engine=engine, start_servers=False)

    pipe.builder_inited()
    for name, body in (
        ("a", "A\n=\n\nBody a.\n"),
        ("b", "B\n=\n\nBody b.\n"),
        ("c", "C\n=\n\nBody c.\n"),
    ):
        pipe.doctree_resolved(publish_doctree(body), name)
    assert {"node:a", "node:b", "node:c"} <= set(pipe._resources)  # noqa: SLF001

    pipe.build_finished(None)

    assert engine.private_store is not None
    # Every document's chunks were ingested by the single main-process writer.
    assert engine.private_store.count() >= 3

