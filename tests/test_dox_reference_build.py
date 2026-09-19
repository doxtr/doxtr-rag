"""dox reference full-build regression test (release-gating end-to-end guard).

Builds the entire ``github.com/doxtr/dox`` reference project with ``doxtr_rag``
enabled through a real ``sphinx-build`` and asserts, against a committed
baseline (``tests/baselines/dox_reference.json``):

  * the Sphinx build still succeeds (exit 0),
  * the store is populated through the real builder-inited → doctree-resolved →
    build-finished lifecycle (docs + chunks above baseline floors),
  * scope routing + embedding parity hold (SHARED_ORG, bge-m3 / 1024-dim).

This is the guard that tells us **if a new documentation construct or extension
in dox breaks doxtr-rag's harvesting/ingestion**: a drop below the baseline
floors, or a build failure, means something the extension relied on changed.

Skipped gracefully when the toolchain, the dox project, or the bge-m3 weights
are unavailable (e.g. a minimal CI runner) so it never produces a false red.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from tests.helpers.dox_build import build_dox_with_rag, locate_dox

BASELINE = json.loads(
    (Path(__file__).parent / "baselines" / "dox_reference.json").read_text(encoding="utf-8")
)
BGE_DIR = os.environ.get("DOXTR_RAG_BGE_M3_ONNX_DIR", "/opt/models/bge-m3")
_HAS_BGE = (
    Path(BGE_DIR, "model.onnx").is_file()
    and Path(BGE_DIR, "model.onnx_data").is_file()
    and Path(BGE_DIR, "tokenizer.json").is_file()
)
_DOX = locate_dox(clone_if_missing=False)


@pytest.mark.skipif(
    not _HAS_BGE or _DOX is None,
    reason="dox project or bge-m3 weights unavailable",
)
def test_dox_reference_full_build() -> None:
    result = build_dox_with_rag(_DOX, weights_dir=BGE_DIR, timeout=600)

    # 1. The real Sphinx build of the full reference project still succeeds.
    assert result.sphinx_exit == BASELINE["sphinx_build_exit"], (
        f"sphinx-build exited {result.sphinx_exit} (expected "
        f"{BASELINE['sphinx_build_exit']}). Log tail:\n{result.log_tail}"
    )

    # 2. Ingestion happened through the real event lifecycle, above the floors.
    #    A drop here means a new dox construct/extension broke harvesting.
    assert result.docs_ingested >= BASELINE["min_docs_ingested"], (
        f"only {result.docs_ingested} docs ingested (baseline floor "
        f"{BASELINE['min_docs_ingested']}); a doc construct/extension may have "
        f"broken doxtr-rag harvesting."
    )
    assert result.total_chunks >= BASELINE["min_total_chunks"], (
        f"only {result.total_chunks} chunks (baseline floor "
        f"{BASELINE['min_total_chunks']})."
    )

    # 3. All resources are node-kind and route to the expected scope; parity holds.
    assert all(rid.startswith("node:") for rid in result.per_doc_chunks), (
        "unexpected resource kinds ingested from dox"
    )
    assert result.endpoints is not None
    assert result.endpoints["shared"]["database"] == "doxtr_shared"
    assert result.endpoints["private"]["database"] == "doxtr_private"
    assert result.endpoints["embedding"]["dimension"] == BASELINE["embedding_dimension"]
    assert result.endpoints["embedding"]["model"] == BASELINE["embedding_model"]

    # cleanup scratch/build to keep tmp tidy
    for d in (result.scratch_dir, result.build_dir):
        if d:
            shutil.rmtree(Path(d).parent, ignore_errors=True)
            break


@pytest.mark.skipif(_DOX is None, reason="dox project unavailable")
def test_dox_reference_baseline_is_sane() -> None:
    # Guard the baseline file itself so it can't silently degrade to trivial floors.
    assert BASELINE["min_docs_ingested"] >= 30
    assert BASELINE["min_total_chunks"] >= 100
    assert BASELINE["embedding_dimension"] == 1024
