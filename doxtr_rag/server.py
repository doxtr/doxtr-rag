"""Loopback Chroma HTTP server for embedded stores (P5 serving topology).

The TypeScript query extension (P6) uses a Chroma **HTTP** client and cannot
read an embedded on-disk store directly. When a store is ``embedded``, P5
starts a loopback-bound (``127.0.0.1``) Chroma HTTP server over that store's
path so P6 can query it. When a store is ``http``, no local server is started —
Python and TS both point at the configured URI.

This module shells out to the ``chroma run`` CLI (bundled with the ``chromadb``
package) as a subprocess bound to loopback, and tracks it for clean shutdown at
``build-finished``.
"""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import sys
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LoopbackChromaServer:
    """Manages a loopback-bound ``chroma run`` subprocess over an embedded path."""

    def __init__(self, path: str, port: Optional[int] = None) -> None:
        self.path = path
        self.port = port or _free_loopback_port()
        self.host = "127.0.0.1"
        self._proc: Optional[subprocess.Popen[bytes]] = None

    @property
    def uri(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, *, wait_seconds: float = 15.0) -> None:
        """Start the server and block until it accepts connections (best-effort).

        Raises RuntimeError if the ``chroma`` CLI is unavailable or it never
        becomes reachable within ``wait_seconds``.
        """
        if self._proc is not None:
            return
        exe = shutil.which("chroma")
        cmd = (
            [exe, "run", "--host", self.host, "--port", str(self.port), "--path", self.path]
            if exe
            else [
                sys.executable,
                "-m",
                "chromadb.cli.cli",
                "run",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--path",
                self.path,
            ]
        )
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if not self._wait_ready(wait_seconds):
            self.stop()
            raise RuntimeError(
                f"loopback Chroma server did not become ready on {self.uri}"
            )

    def _wait_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                return False  # process exited early
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex((self.host, self.port)) == 0:
                    return True
            time.sleep(0.2)
        return False

    def stop(self) -> None:
        """Terminate the server subprocess if running."""
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception as exc:  # noqa: BLE001
            logger.debug("doxtr-rag: loopback server stop error: %s", exc)

    def __enter__(self) -> "LoopbackChromaServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
