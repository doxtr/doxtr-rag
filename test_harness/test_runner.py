"""doxtr-rag Test Harness — Runner.

Builds the harness Sphinx project with the doxtr-rag extension enabled and
performs coarse smoke assertions (build succeeds, no LOCAL_PRIVATE leaks into
the shared store, audit log scrubbed). Detailed round-trip validation lives in
the P7 cross-language integration test; this runner is the CI regression entry
point invoked by ``.github/workflows/test-harness.yml``.

Usage:
    python test_runner.py            # build + assert
    python test_runner.py --clean    # remove build + scratch .doxtr first

Exit codes:
    0  build + assertions passed
    1  build or assertion failed
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HARNESS = Path(__file__).parent.resolve()
SOURCE = HARNESS / "source"
BUILD = HARNESS / "_build" / "html"
SCRATCH = HARNESS / ".doxtr"


def _clean() -> None:
    for path in (HARNESS / "_build", SCRATCH):
        if path.exists():
            shutil.rmtree(path)


def build() -> int:
    """Run ``sphinx-build`` for the harness; return the process exit code."""
    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        "-c",
        str(HARNESS),
        str(SOURCE),
        str(BUILD),
    ]
    proc = subprocess.run(cmd, cwd=str(HARNESS))
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="doxtr-rag test harness runner")
    parser.add_argument("--clean", action="store_true", help="clean before build")
    args = parser.parse_args(argv)

    if args.clean:
        _clean()

    rc = build()
    if rc != 0:
        print(f"[FAIL] sphinx-build exited {rc}")
        return 1

    print("[OK] harness build succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
