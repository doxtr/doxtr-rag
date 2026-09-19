"""Minimal query-time embedding HTTP server (bge-m3 parity for P6).

The TypeScript query extension (P6) embeds the query at agent time by POSTing
to ``DOXTR_RAG_EMBED_URL``. This module provides a tiny stdlib HTTP server that
serves the SAME pinned ``bge-m3`` weights as Python ingestion (via
:class:`doxtr_rag.embedding_provider.EmbeddingProvider`), guaranteeing
cross-language embedding parity (same model + 1024 dims).

Protocol (matches ``chroma-rag.ts`` defaultEmbedder):
    POST /embed  {"input": "<text>", "model": "bge-m3"}  -> {"embedding": [...1024]}

Run standalone:
    python -m doxtr_rag.embed_server --host 127.0.0.1 --port 8519
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from doxtr_rag.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbedServer:
    """A loopback HTTP server exposing the pinned bge-m3 embedder."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8519,
        provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self.host = host
        self.port = port
        self._provider = provider or EmbeddingProvider()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def uri(self) -> str:
        return f"http://{self.host}:{self.port}/embed"

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        provider = self._provider

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:  # silence access log
                return

            def do_POST(self) -> None:  # noqa: N802 - stdlib signature
                try:
                    length = int(self.headers.get("content-length", 0))
                    body = json.loads(self.rfile.read(length) or b"{}")
                    text = body.get("input", "")
                    vector = provider.embed([text])[0]
                    payload = json.dumps({"embedding": vector, "model": provider.model}).encode()
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as exc:  # noqa: BLE001 - never crash the server
                    msg = json.dumps({"error": str(exc)}).encode()
                    self.send_response(500)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)

        return _Handler

    def start(self) -> None:
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler_class())
        # If port 0 was requested, capture the OS-assigned port.
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def __enter__(self) -> "EmbedServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="doxtr-rag-embed-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8519)
    args = parser.parse_args(argv)
    server = EmbedServer(host=args.host, port=args.port)
    server.start()
    print(f"[doxtr-rag] bge-m3 embed server on {server.uri}")
    try:
        if server._thread is not None:
            server._thread.join()
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
