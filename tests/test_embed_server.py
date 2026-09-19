"""Tests for the bge-m3 query-time embed server (P6 parity helper)."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest

from doxtr_rag.embed_server import EmbedServer

_BGE_DIR = os.environ.get("DOXTR_RAG_BGE_M3_ONNX_DIR", "/opt/models/bge-m3")
_HAS_BGE = (
    Path(_BGE_DIR, "model.onnx").is_file()
    and Path(_BGE_DIR, "model.onnx_data").is_file()
    and Path(_BGE_DIR, "tokenizer.json").is_file()
)


class _FakeProvider:
    model = "bge-m3"
    dimension = 1024

    def embed(self, texts):  # type: ignore[no-untyped-def]
        return [[0.1] * 1024 for _ in texts]


def _post(uri: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        uri, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        return e.code, json.loads(e.read())


def test_embed_server_serves_vectors() -> None:
    server = EmbedServer(host="127.0.0.1", port=0, provider=_FakeProvider())  # type: ignore[arg-type]
    server.start()
    try:
        assert server.uri.startswith("http://127.0.0.1:")
        status, payload = _post(server.uri, {"input": "hello", "model": "bge-m3"})
        assert status == 200
        assert len(payload["embedding"]) == 1024
        assert payload["model"] == "bge-m3"
    finally:
        server.stop()


def test_embed_server_error_path() -> None:
    class _Boom:
        model = "bge-m3"

        def embed(self, texts):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    server = EmbedServer(host="127.0.0.1", port=0, provider=_Boom())  # type: ignore[arg-type]
    server.start()
    try:
        status, payload = _post(server.uri, {"input": "x"})
        assert status == 500
        assert "error" in payload
    finally:
        server.stop()


def test_embed_server_context_manager() -> None:
    with EmbedServer(host="127.0.0.1", port=0, provider=_FakeProvider()) as s:  # type: ignore[arg-type]
        status, _ = _post(s.uri, {"input": "hi"})
        assert status == 200


@pytest.mark.skipif(not _HAS_BGE, reason="bge-m3 weights not present")
def test_embed_server_real_bge_m3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOXTR_RAG_BGE_M3_ONNX_DIR", _BGE_DIR)
    with EmbedServer(host="127.0.0.1", port=0) as s:
        status, payload = _post(s.uri, {"input": "reactor deployment", "model": "bge-m3"})
    assert status == 200
    v = payload["embedding"]
    assert len(v) == 1024
    assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-3
