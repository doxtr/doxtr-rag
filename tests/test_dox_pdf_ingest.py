"""Real-world PDF ingest regression test.

Builds a genuine PDF from the ``github.com/doxtr/dox`` reference project
(Sphinx LaTeX → lualatex, via the doxtr PDF theme) and ingests it through the
P3 PDF extractor + P5 storage with real bge-m3 embeddings. This exercises the
PDF path against a large, real-world document — not a hand-crafted fixture — so
a regression in PDF extraction (page chunking, provenance, text extraction) or
in embedding a real corpus is caught end to end.

Slow (LaTeX compile of a ~150-page doc); gated on the LaTeX toolchain +
bge-m3 weights, so it runs in the ``doxtr/reactor`` container and skips
gracefully elsewhere.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterator

import pytest

from doxtr_rag.schema import AccessControl
from tests.helpers.dox_build import build_dox_pdf, locate_dox

BGE_DIR = os.environ.get("DOXTR_RAG_BGE_M3_ONNX_DIR", "/opt/models/bge-m3")
_HAS_BGE = (
    Path(BGE_DIR, "model.onnx").is_file()
    and Path(BGE_DIR, "model.onnx_data").is_file()
    and Path(BGE_DIR, "tokenizer.json").is_file()
)
_HAS_LATEX = shutil.which("lualatex") is not None and shutil.which("latexmk") is not None
_DOX = locate_dox(clone_if_missing=False)

# Baseline floors for the real dox PDF (157-page LaTeX output). Conservative so
# the test is robust to minor dox content changes but still catches breakage.
_MIN_PAGES_WITH_TEXT = 80


@pytest.fixture(scope="module")
def dox_pdf() -> Iterator[str]:
    """Build the real dox PDF once for the module (LaTeX compile is slow)."""
    if _DOX is None or not _HAS_LATEX:
        pytest.skip("dox project or LaTeX toolchain unavailable")
    pdf = build_dox_pdf(_DOX, timeout=900)
    if pdf is None:
        pytest.skip("dox PDF build did not produce a PDF in this environment")
    try:
        yield pdf
    finally:
        shutil.rmtree(Path(pdf).parent.parent, ignore_errors=True)


@pytest.mark.skipif(
    _DOX is None or not _HAS_LATEX,
    reason="dox project or LaTeX toolchain unavailable",
)
def test_dox_pdf_extraction(dox_pdf: str) -> None:
    """The real dox PDF extracts page-level chunks with correct provenance."""
    from doxtr_rag.extractors import extract_file

    chunks = list(extract_file(dox_pdf, AccessControl.SHARED_ORG))
    nonempty = [c for c in chunks if c.text.strip()]

    # Real multi-page extraction with text.
    assert len(nonempty) >= _MIN_PAGES_WITH_TEXT, (
        f"only {len(nonempty)} pages with text extracted from the dox PDF "
        f"(floor {_MIN_PAGES_WITH_TEXT}); PDF extraction may have regressed."
    )
    # Every chunk uses the C2 {page} provenance shape, 1-based.
    pages = [c.metadata["provenance"] for c in chunks]
    assert all(set(p) == {"page"} for p in pages)
    assert all(p["page"] >= 1 for p in pages)
    # Provisional scope preserved (finalized later by P5).
    assert all(c.access_scope is AccessControl.SHARED_ORG for c in chunks)
    # Content sanity: the title page text made it through.
    joined = "\n".join(c.text for c in nonempty[:5]).lower()
    assert "doxamples" in joined or "contents" in joined


@pytest.mark.skipif(
    _DOX is None or not _HAS_LATEX or not _HAS_BGE,
    reason="dox project, LaTeX toolchain, or bge-m3 weights unavailable",
)
def test_dox_pdf_ingest_and_query_real_embeddings(
    dox_pdf: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ingest the real dox PDF into a store with real bge-m3 vectors and query it."""
    import uuid

    from doxtr_rag.config import SphinxRAGConfig
    from doxtr_rag.embedding_provider import EmbeddingProvider
    from doxtr_rag.extractors import extract_file
    from doxtr_rag.storage import IngestionEngine

    monkeypatch.setenv("DOXTR_RAG_BGE_M3_ONNX_DIR", BGE_DIR)
    config = SphinxRAGConfig.from_mapping(
        {
            "rag_local_store_mode": "embedded",
            "rag_local_store_path": str(tmp_path / f"priv_{uuid.uuid4().hex[:6]}"),
            "rag_shared_store_mode": "embedded",
            "rag_shared_store_uri": str(tmp_path / f"shared_{uuid.uuid4().hex[:6]}"),
            "rag_cache_db": str(tmp_path / "cache.db"),
        }
    )
    provider = EmbeddingProvider()  # REAL bge-m3
    engine = IngestionEngine(config, provider=provider, batch_size=64)
    engine.open_stores()

    chunks = [c for c in extract_file(dox_pdf, AccessControl.SHARED_ORG) if c.text.strip()]
    counts = engine.ingest(chunks)
    assert counts["SHARED_ORG"] == len(chunks)
    assert engine.shared_store is not None
    assert engine.shared_store.count() == len(chunks)

    # Real semantic query against the ingested PDF pages.
    qvec = provider.embed(["basic elements and body elements in the documentation"])[0]
    hits = engine.shared_store.query(qvec, 3)
    assert len(hits) >= 1
    assert all(h.metadata["provenance"].get("page") for h in hits)
