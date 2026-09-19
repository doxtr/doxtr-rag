"""Shared helper: run a full real ``sphinx-build`` of the dox reference project
with the ``doxtr_rag`` extension enabled, and report what got ingested.

Used by the dox-reference regression test (and a baseline-regen script). This
is the end-to-end guard that proves the extension survives the *actual* Sphinx
event lifecycle against a large, feature-rich real project (sphinx-needs,
drawio, roadmap, xlink, bibtex, d2, ...), and flags when a new documentation
construct or extension breaks doxtr-rag harvesting/ingestion.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DOX_REPO = "https://github.com/doxtr/dox"
DOX_CACHE_CANDIDATES = ("/tmp/pi-gh-QNGb2C/dox", "/tmp/dox")

# conf.py snippet appended to the dox project to enable doxtr_rag and point its
# stores at a scratch dir (embedded, self-contained — no external server).
_CONF_INJECT = """

# --- doxtr_rag full-build regression injection ---
import os as _os
extensions = list(extensions) + ["doxtr_rag"]
_D = _os.environ["DOXTR_RAG_TEST_DIR"]
rag_local_store_mode = "embedded"
rag_local_store_path = _os.path.join(_D, "chroma")
rag_shared_store_mode = "embedded"
rag_shared_store_uri = _os.path.join(_D, "chroma_shared")
rag_cache_db = _os.path.join(_D, "cache.db")
rag_audit_log_path = _os.path.join(_D, "audit.jsonl")
"""


@dataclass
class DoxBuildResult:
    """Outcome of a full dox build with doxtr_rag enabled."""

    sphinx_exit: int
    docs_ingested: int
    total_chunks: int
    per_doc_chunks: dict[str, int] = field(default_factory=dict)
    endpoints: Optional[dict] = None
    log_tail: str = ""
    build_dir: Optional[str] = None
    scratch_dir: Optional[str] = None
    pdf_path: Optional[str] = None


def locate_dox(clone_if_missing: bool = True) -> Optional[str]:
    """Return a path to a dox checkout, cloning to /tmp/dox if needed."""
    for cand in DOX_CACHE_CANDIDATES:
        if Path(cand, "source", "conf.py").is_file():
            return cand
    if not clone_if_missing:
        return None
    if shutil.which("git") is None:
        return None
    dest = "/tmp/dox"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", DOX_REPO, dest],
            check=True,
            capture_output=True,
            timeout=180,
        )
    except Exception:  # noqa: BLE001
        return None
    return dest if Path(dest, "source", "conf.py").is_file() else None


def build_dox_with_rag(
    dox_dir: str,
    *,
    weights_dir: Optional[str] = None,
    timeout: int = 600,
) -> DoxBuildResult:
    """Run ``sphinx-build`` of the dox project with doxtr_rag enabled.

    Returns a :class:`DoxBuildResult`. The scratch store + build output live in
    a temp dir the caller may inspect/clean up (``scratch_dir``/``build_dir``).
    """
    work = tempfile.mkdtemp(prefix="doxtr-rag-dox-")
    src = os.path.join(work, "source")
    shutil.copytree(os.path.join(dox_dir, "source"), src)
    with open(os.path.join(src, "conf.py"), "a", encoding="utf-8") as fh:
        fh.write(_CONF_INJECT)

    scratch = os.path.join(work, ".doxtr")
    os.makedirs(scratch, exist_ok=True)
    build_out = os.path.join(work, "_build_html")

    env = dict(os.environ)
    env["DOXTR_RAG_TEST_DIR"] = scratch
    if weights_dir:
        env["DOXTR_RAG_BGE_M3_ONNX_DIR"] = weights_dir

    proc = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", "-q", src, build_out],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    per_doc: dict[str, int] = {}
    endpoints = None
    cache_db = os.path.join(scratch, "cache.db")
    if os.path.isfile(cache_db):
        con = sqlite3.connect(cache_db)
        con.row_factory = sqlite3.Row
        try:
            for row in con.execute(
                "SELECT resource_id, COUNT(*) n FROM resource_chunks GROUP BY resource_id"
            ):
                per_doc[row["resource_id"]] = row["n"]
        finally:
            con.close()
    ep_path = os.path.join(scratch, "endpoints.json")
    if os.path.isfile(ep_path):
        import json

        endpoints = json.loads(Path(ep_path).read_text(encoding="utf-8"))

    return DoxBuildResult(
        sphinx_exit=proc.returncode,
        docs_ingested=len(per_doc),
        total_chunks=sum(per_doc.values()),
        per_doc_chunks=per_doc,
        endpoints=endpoints,
        log_tail="\n".join((proc.stderr or "").splitlines()[-25:]),
        build_dir=build_out,
        scratch_dir=scratch,
    )


def build_dox_pdf(dox_dir: str, *, timeout: int = 900) -> Optional[str]:
    """Build a real PDF from the dox project (LaTeX → lualatex) and return its path.

    Produces a genuine, non-synthetic multi-page PDF (via the doxtr PDF theme's
    LaTeX config) suitable as a real-world fixture for the P3 PDF extractor.
    Returns the ``.pdf`` path on success, or ``None`` if the LaTeX toolchain is
    unavailable or the build did not produce a PDF. Requires ``lualatex`` +
    ``latexmk`` (present in the ``doxtr/reactor`` container).
    """
    if shutil.which("lualatex") is None or shutil.which("latexmk") is None:
        return None
    work = tempfile.mkdtemp(prefix="doxtr-rag-doxpdf-")
    latex_out = os.path.join(work, "latex")
    # 1. Sphinx -> LaTeX .tex
    proc = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "latex", "-q",
         os.path.join(dox_dir, "source"), latex_out],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        return None
    tex = next((f for f in os.listdir(latex_out) if f.endswith(".tex")), None)
    if tex is None:
        return None
    # 2. lualatex compile (latexmk handles reruns; exit code may be non-zero on
    #    benign LaTeX warnings, so we validate by checking the PDF exists).
    subprocess.run(
        ["latexmk", "-lualatex", "-interaction=nonstopmode", "-halt-on-error", tex],
        cwd=latex_out,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    pdf = os.path.join(latex_out, tex[:-4] + ".pdf")
    return pdf if os.path.isfile(pdf) and os.path.getsize(pdf) > 10_000 else None
