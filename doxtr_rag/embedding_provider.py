"""bge-m3 embedding provider (P5 — CONVENTIONS §5 Embedding Parity).

Runs the pinned ``BAAI/bge-m3`` ONNX weights (described in
:mod:`doxtr_rag.embedding`) directly via ``onnxruntime`` + ``tokenizers`` and
returns the model's L2-normalized dense **CLS** embedding (its
``sentence_embedding`` output), a 1024-dim vector.

Both stores (shared + private) MUST use the same provider and the same 1024-dim
output; a dimension mismatch is fatal. When the pre-baked weights are
unavailable it raises :class:`~doxtr_rag.embedding.WeightsUnavailableError`
(never silently substitutes a different model — that would break parity).

bge-m3 is **not** a fastembed built-in model, so the provider loads the ONNX
weights directly rather than via ``fastembed.TextEmbedding``. The ONNX graph
signature is:

    inputs : input_ids (int64 [B,T]), attention_mask (int64 [B,T])
    outputs: token_embeddings (float [B,T,1024]), sentence_embedding (float [B,1024])

``sentence_embedding`` is the L2-normalized CLS token — bge-m3's dense vector.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

from doxtr_rag.embedding import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_ID,
    assert_dimension,
    require_weights,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

#: Default max sequence length (bge-m3 supports up to 8192; cap for speed/memory).
DEFAULT_MAX_LENGTH = 512

#: ONNX filenames within the weights directory.
_MODEL_FILE = "model.onnx"
_TOKENIZER_FILE = "tokenizer.json"

#: The pooled dense output the model exposes (L2-normalized CLS).
_SENTENCE_OUTPUT = "sentence_embedding"


class EmbeddingProvider:
    """Loads and runs the pinned bge-m3 ONNX embedder (lazy, parity-checked)."""

    def __init__(self, *, require: bool = True, max_length: int = DEFAULT_MAX_LENGTH) -> None:
        self.model = EMBEDDING_MODEL_ID
        self._dim = EMBEDDING_DIM
        self._max_length = max_length
        self._require = require
        self._session: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._output_name: str = _SENTENCE_OUTPUT

    @property
    def dimension(self) -> int:
        return self._dim

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._session is not None and self._tokenizer is not None:
            return
        weights = require_weights()  # raises WeightsUnavailableError if absent
        model_path = os.path.join(weights, _MODEL_FILE)
        tok_path = os.path.join(weights, _TOKENIZER_FILE)
        if not os.path.isfile(model_path) or not os.path.isfile(tok_path):
            from doxtr_rag.embedding import WeightsUnavailableError

            raise WeightsUnavailableError(
                f"bge-m3 ONNX files missing under {weights!r} "
                f"(need {_MODEL_FILE} + {_TOKENIZER_FILE})"
            )
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tok_path)
        tokenizer.enable_truncation(max_length=self._max_length)
        tokenizer.enable_padding()
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        # Prefer the pooled sentence_embedding output; fall back to CLS pooling.
        output_names = {o.name for o in session.get_outputs()}
        self._output_name = (
            _SENTENCE_OUTPUT if _SENTENCE_OUTPUT in output_names else "token_embeddings"
        )
        self._session = session
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts to 1024-dim dense vectors (parity-asserted).

        Returns the L2-normalized CLS dense embedding for each text.
        """
        if not texts:
            return []
        self._ensure_loaded()
        assert self._session is not None and self._tokenizer is not None
        import numpy as np

        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        outputs = self._session.run(
            [self._output_name],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        raw = outputs[0]
        vectors = self._pool(raw, attention_mask)
        result = [list(map(float, v)) for v in vectors]
        for v in result:
            assert_dimension(len(v))
        return result

    def _pool(self, raw: "np.ndarray", attention_mask: "np.ndarray") -> "np.ndarray":
        """Return L2-normalized dense vectors from the model output.

        If the model exposed the pooled ``sentence_embedding`` (already
        normalized), use it directly; otherwise take the CLS token from
        ``token_embeddings`` and L2-normalize (bge-m3 dense pooling).
        """
        import numpy as np

        if self._output_name == _SENTENCE_OUTPUT:
            vectors = raw
        else:
            # token_embeddings [B, T, 1024] -> CLS (first token) per row.
            vectors = raw[:, 0, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        result: "np.ndarray" = vectors / norms
        return result
