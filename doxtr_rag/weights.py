"""Weight provisioning for the pinned bge-m3 ONNX model.

Downloads the pinned ``BAAI/bge-m3`` ONNX weights into
``DOXTR_RAG_BGE_M3_ONNX_DIR`` (default ``/opt/models/bge-m3``) using
``huggingface_hub`` in-process \u2014 no shell, curl, or aria2 dependency, so it
works anywhere Python does. Pinned to a specific commit so the model revision
matches the 1024-dim parity contract (CONVENTIONS \u00a75).

In the ``doxtr/reactor`` image the weights are pre-baked, so this is a no-op
there. For a bare ``pip install`` the user runs ``doxtr-rag fetch-weights``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from doxtr_rag.embedding import WEIGHTS_ENV, weights_dir

#: Pinned HF repo + commit for the bge-m3 ONNX weights (matches the reactor
#: builder Dockerfile and the 1024-dim parity contract).
BGE_M3_REPO = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

#: The ONNX files the provider needs (the model + its external-data + tokenizer).
REQUIRED_FILES = (
    "model.onnx",
    "model.onnx_data",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "Constant_7_attr__value",
)

DEFAULT_WEIGHTS_DIR = "/opt/models/bge-m3"


def target_dir() -> str:
    """Return the directory weights should be installed to."""
    return weights_dir() or DEFAULT_WEIGHTS_DIR


def is_present(directory: Optional[str] = None) -> bool:
    """Return True if the essential bge-m3 ONNX files are present."""
    d = directory or target_dir()
    return (
        os.path.isfile(os.path.join(d, "model.onnx"))
        and os.path.isfile(os.path.join(d, "model.onnx_data"))
        and os.path.isfile(os.path.join(d, "tokenizer.json"))
    )


def fetch_weights(
    directory: Optional[str] = None,
    *,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> str:
    """Download the pinned bge-m3 ONNX weights into ``directory`` (idempotent).

    Uses ``huggingface_hub.snapshot_download`` (resumable, checksum-verified)
    restricted to the repo's ``onnx/`` folder, then flattens the files into
    ``directory`` so the provider finds them at the top level.

    Args:
        directory: Target dir; defaults to ``DOXTR_RAG_BGE_M3_ONNX_DIR`` or
            ``/opt/models/bge-m3``.
        force: Re-download even if the files are already present.
        log: Progress sink (defaults to ``print``).

    Returns:
        The directory the weights were installed to.
    """
    dest = directory or target_dir()
    if is_present(dest) and not force:
        log(f"[doxtr-rag] bge-m3 weights already present at {dest}")
        return dest

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - hf_hub is a declared dep
        raise RuntimeError(
            "huggingface_hub is required to fetch weights; install doxtr-rag's "
            "dependencies (pip install doxtr-rag)."
        ) from exc

    os.makedirs(dest, exist_ok=True)
    log(
        f"[doxtr-rag] downloading {BGE_M3_REPO} ONNX weights (~2.3 GB, pinned "
        f"{BGE_M3_REVISION[:12]}) to {dest} ..."
    )
    snapshot = snapshot_download(
        repo_id=BGE_M3_REPO,
        revision=BGE_M3_REVISION,
        allow_patterns=["onnx/*"],
    )
    onnx_src = os.path.join(snapshot, "onnx")
    # Flatten onnx/* into the target dir the provider reads.
    for name in REQUIRED_FILES:
        src = os.path.join(onnx_src, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(dest, name)
        _link_or_copy(src, dst)

    if not is_present(dest):
        raise RuntimeError(
            f"weight download completed but essential files are missing under {dest}; "
            "check network access and the pinned revision."
        )
    log(f"[doxtr-rag] bge-m3 weights ready at {dest}")
    if not weights_dir():
        log(
            f"[doxtr-rag] note: set {WEIGHTS_ENV}={dest} so the extension finds them "
            "(the doxtr/reactor image sets this for you)."
        )
    return dest


def _link_or_copy(src: str, dst: str) -> None:
    """Hardlink src->dst (fast, no extra disk); fall back to copy across devices."""
    import shutil

    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
