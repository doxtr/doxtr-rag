"""Tests for weight provisioning (doxtr_rag.weights + fetch-weights CLI)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from doxtr_rag import cli, weights
from doxtr_rag.embedding import WEIGHTS_ENV, WeightsUnavailableError, require_weights


def test_pinned_revision_and_required_files() -> None:
    assert weights.BGE_M3_REPO == "BAAI/bge-m3"
    assert len(weights.BGE_M3_REVISION) == 40  # full commit SHA
    assert "model.onnx" in weights.REQUIRED_FILES
    assert "model.onnx_data" in weights.REQUIRED_FILES
    assert "tokenizer.json" in weights.REQUIRED_FILES


def test_target_dir_honors_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "w"))
    assert weights.target_dir() == str(tmp_path / "w")
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    assert weights.target_dir() == weights.DEFAULT_WEIGHTS_DIR


def test_is_present(tmp_path: Path) -> None:
    d = tmp_path / "w"
    d.mkdir()
    assert weights.is_present(str(d)) is False
    for f in ("model.onnx", "model.onnx_data", "tokenizer.json"):
        (d / f).write_text("x")
    assert weights.is_present(str(d)) is True


def test_require_weights_error_names_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    with pytest.raises(WeightsUnavailableError) as exc:
        require_weights()
    assert "doxtr-rag fetch-weights" in str(exc.value)


def test_fetch_weights_noop_when_present(tmp_path: Path) -> None:
    d = tmp_path / "w"
    d.mkdir()
    for f in ("model.onnx", "model.onnx_data", "tokenizer.json"):
        (d / f).write_text("x")
    logs: list[str] = []
    result = weights.fetch_weights(str(d), log=logs.append)
    assert result == str(d)
    assert any("already present" in m for m in logs)


def test_fetch_weights_downloads_and_flattens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Fake a HF snapshot dir with an onnx/ subfolder; assert flatten + is_present.
    snap = tmp_path / "snapshot"
    onnx = snap / "onnx"
    onnx.mkdir(parents=True)
    for f in weights.REQUIRED_FILES:
        (onnx / f).write_text("data")

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: str(snap)
    )
    dest = tmp_path / "dest"
    logs: list[str] = []
    result = weights.fetch_weights(str(dest), log=logs.append)
    assert result == str(dest)
    assert weights.is_present(str(dest))
    assert (dest / "model.onnx").is_file()
    assert (dest / "tokenizer.json").is_file()


def test_fetch_weights_raises_if_incomplete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snap = tmp_path / "snapshot"
    (snap / "onnx").mkdir(parents=True)
    # Only a partial set -> is_present() False -> RuntimeError.
    (snap / "onnx" / "config.json").write_text("x")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(snap))
    with pytest.raises(RuntimeError):
        weights.fetch_weights(str(tmp_path / "dest"))


# -- CLI wiring --------------------------------------------------------------
def test_cli_fetch_weights_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    called: dict[str, object] = {}

    def fake_fetch(directory=None, *, force=False, log=print):  # type: ignore[no-untyped-def]
        called["force"] = force
        return "/tmp/weights"

    monkeypatch.setattr(weights, "fetch_weights", fake_fetch)
    rc = cli.main(["fetch-weights"])
    assert rc == 0
    assert called["force"] is False


def test_cli_fetch_weights_force(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}
    monkeypatch.setattr(
        weights, "fetch_weights",
        lambda directory=None, *, force=False, log=print: called.setdefault("force", force),
    )
    assert cli.main(["fetch-weights", "--force"]) == 0
    assert called["force"] is True


def test_cli_fetch_weights_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def boom(**kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("network down")

    monkeypatch.setattr(weights, "fetch_weights", boom)
    rc = cli.main(["fetch-weights"])
    assert rc == 1
    assert "fetch-weights failed" in capsys.readouterr().err
