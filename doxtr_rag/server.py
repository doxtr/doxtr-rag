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


def port_is_serving(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    """Return True if something is already accepting connections on host:port.

    Used to decide whether to reuse an already-running loopback server on a
    stable configured port rather than spawning a duplicate. The ports are
    fixed, loopback-only and below the ephemeral range, so a live listener
    there is our own server from a prior build or the standalone launcher.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


class LoopbackChromaServer:
    """Manages a loopback-bound ``chroma run`` subprocess over an embedded path."""

    def __init__(self, path: str, port: Optional[int] = None) -> None:
        self.path = path
        self.port = port or _free_loopback_port()
        self.host = "127.0.0.1"
        self._proc: Optional[subprocess.Popen[bytes]] = None
        #: True when start() found a server already running on the port and
        #: reused it instead of spawning; stop() then leaves it untouched.
        self.reused = False

    @property
    def uri(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, *, wait_seconds: float = 15.0, reuse_if_running: bool = False) -> None:
        """Start the server and block until it accepts connections (best-effort).

        When ``reuse_if_running`` is True and something is already listening on
        this (stable) port, adopt it: record it as reused and do NOT spawn a
        duplicate subprocess. ``stop()`` will then leave that process running.

        Raises RuntimeError if the ``chroma`` CLI is unavailable or it never
        becomes reachable within ``wait_seconds``.
        """
        if self._proc is not None or self.reused:
            return
        if reuse_if_running and port_is_serving(self.port, self.host):
            self.reused = True
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
        """Terminate the server subprocess if we started one.

        A reused (externally-owned) server is never stopped here.
        """
        if self.reused or self._proc is None:
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
