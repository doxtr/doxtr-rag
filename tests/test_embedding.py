"""F1 tests for the pinned embedding strategy (CONVENTIONS §5)."""

from __future__ import annotations

import pytest

from doxtr_rag import embedding


def test_constants() -> None:
    assert embedding.EMBEDDING_MODEL_ID == "BAAI/bge-m3"
    assert embedding.EMBEDDING_MODEL_REVISION == "bge-m3"
    assert embedding.EMBEDDING_DIM == 1024


def test_assert_dimension_ok() -> None:
    embedding.assert_dimension(1024)  # no raise


def test_assert_dimension_mismatch_is_fatal() -> None:
    with pytest.raises(ValueError):
        embedding.assert_dimension(768)


def test_weights_dir_unset(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv(embedding.WEIGHTS_ENV, raising=False)
    assert embedding.weights_dir() is None
    with pytest.raises(embedding.WeightsUnavailableError):
        embedding.require_weights()


def test_require_weights_present(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(embedding.WEIGHTS_ENV, str(tmp_path))
    assert embedding.require_weights() == str(tmp_path)


def test_require_weights_missing_dir(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(embedding.WEIGHTS_ENV, str(tmp_path / "nope"))
    with pytest.raises(embedding.WeightsUnavailableError):
        embedding.require_weights()
