"""Pinned embedding model strategy (CONVENTIONS §5 — Embedding Parity).

``bge-m3`` is **not** a fastembed built-in model, so both the Python ingestion
engine (P5) and the TypeScript query extension (P6) must load the *same* pinned
``BAAI/bge-m3`` ONNX weights with the *same* 1024-dim dense pooling. This module
is the single Python-side source of truth for that decision; P6 mirrors the
same model id + revision + dimension on the TypeScript side.

Strategy (identical across the language boundary):
    1. The ``doxtr/reactor`` container pre-bakes the ``BAAI/bge-m3`` ONNX
       weights (~2.3GB) at ``DOXTR_RAG_BGE_M3_ONNX_DIR`` (CONVENTIONS §Container
       weights). A first-run download during a Sphinx build is avoided.
    2. Python (P5) runs the weights directly via ``onnxruntime`` + ``tokenizers``
       (bge-m3 is NOT a fastembed built-in), taking the model's L2-normalized
       ``sentence_embedding`` (CLS) dense output — see
       :class:`doxtr_rag.embedding_provider.EmbeddingProvider`.
    3. TypeScript (P6) embeds the query with the same pinned model/dimension
       (server-side bge-m3 or a matching client-side embedding).
    4. If the weights are absent, ingestion/query fails with a clear error
       (never a silent fallback to a different model — that would break
       parity). See :func:`require_weights`.

A dimension mismatch (store or query) is a fatal, build-stopping error.
"""

from __future__ import annotations

import os
from typing import Final

from doxtr_rag.base_store import EMBEDDING_DIMENSION

#: HuggingFace model id for the pinned dense embedder.
EMBEDDING_MODEL_ID: Final[str] = "BAAI/bge-m3"

#: Pinned revision tag (used in the collection name via base_store).
EMBEDDING_MODEL_REVISION: Final[str] = "bge-m3"

#: Dense output dimension (re-exported for convenience; MUST be 1024).
EMBEDDING_DIM: Final[int] = EMBEDDING_DIMENSION

#: Env var pointing at the pre-baked ONNX weights directory.
WEIGHTS_ENV: Final[str] = "DOXTR_RAG_BGE_M3_ONNX_DIR"


class WeightsUnavailableError(RuntimeError):
    """Raised when the pinned bge-m3 ONNX weights cannot be located."""


def weights_dir() -> str | None:
    """Return the configured ONNX weights directory, or ``None`` if unset."""
    return os.environ.get(WEIGHTS_ENV)


def require_weights() -> str:
    """Return the weights directory or raise a clear, parity-preserving error.

    Never falls back to a different model: doing so would violate embedding
    parity (CONVENTIONS §5).
    """
    path = weights_dir()
    if not path or not os.path.isdir(path):
        raise WeightsUnavailableError(
            f"pinned bge-m3 ONNX weights not found (set {WEIGHTS_ENV} to their "
            "directory). Run 'doxtr-rag fetch-weights' to download them, or use "
            "the doxtr/reactor image where they are pre-baked. "
            "(A different model is never substituted — that would break "
            "embedding parity.)"
        )
    return path


def assert_dimension(dim: int) -> None:
    """Assert an embedding/query dimension matches the pinned dimension.

    Args:
        dim: The observed dimension from a store or a query embedding.

    Raises:
        ValueError: If ``dim`` != :data:`EMBEDDING_DIM` (fatal parity error).
    """
    if dim != EMBEDDING_DIM:
        raise ValueError(
            f"embedding dimension parity violation: expected {EMBEDDING_DIM} "
            f"(bge-m3 dense), got {dim}"
        )
