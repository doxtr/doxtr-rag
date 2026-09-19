"""Tests for the agent installer (doxtr_rag.agent_install + CLI --install-agent)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from doxtr_rag import agent_install, cli


def test_repo_extension_dir_points_at_packaged_source() -> None:
    ext = agent_install.repo_extension_dir()
    # Source of truth is the packaged doxtr_rag/pi_extension/ dir.
    assert ext.name == "pi_extension"
    assert (ext / "chroma-rag.ts").exists()


def test_packaged_extension_dir_ships_in_package() -> None:
    ext = agent_install.packaged_extension_dir()
    assert ext.name == "pi_extension"
    assert (ext / "chroma-rag.ts").is_file()
    assert (ext / "package.json").is_file()


def test_pi_agent_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PI_AGENT_DIR", str(tmp_path / "custom"))
    assert agent_install.pi_agent_dir() == tmp_path / "custom"
    monkeypatch.delenv("PI_AGENT_DIR", raising=False)
    assert agent_install.pi_agent_dir() == Path.home() / ".pi" / "agent"


def test_install_pi_symlink(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    result = agent_install.install_pi(agent_dir=agent_dir, symlink=True, skip_npm=True)
    assert result.installed
    ext = agent_dir / "extensions" / "chroma-rag"
    assert ext.is_symlink()
    assert ext.resolve() == agent_install.repo_extension_dir().resolve()
    # settings + env snippet written with reactor defaults.
    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings["compaction"]["enabled"] is True
    env = (agent_dir / "doxtr-rag.env").read_text()
    assert "DOXTR_RAG_BGE_M3_ONNX_DIR" in env
    assert "DOXTR_RAG_EMBED_URL" in env
    assert "DOXTR_RAG_ENDPOINTS" in env


def test_install_pi_copy_mode(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    # copy is the DEFAULT now (works from a plain wheel install).
    result = agent_install.install_pi(agent_dir=agent_dir, skip_npm=True)
    ext = agent_dir / "extensions" / "chroma-rag"
    assert ext.is_dir() and not ext.is_symlink()
    assert (ext / "chroma-rag.ts").exists()
    # node_modules is excluded from the copy.
    assert not (ext / "node_modules").exists()
    assert result.installed


def test_install_pi_idempotent_reinstall(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_install.install_pi(agent_dir=agent_dir, symlink=True, skip_npm=True)
    # Second install replaces cleanly without error.
    result = agent_install.install_pi(agent_dir=agent_dir, symlink=True, skip_npm=True)
    ext = agent_dir / "extensions" / "chroma-rag"
    assert ext.is_symlink()
    assert any("removed existing" in a for a in result.actions)


def test_install_preserves_existing_settings(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text(
        json.dumps({"defaultProvider": "c137", "defaultModel": "keep-me"})
    )
    agent_install.install_pi(agent_dir=agent_dir, skip_npm=True)
    settings = json.loads((agent_dir / "settings.json").read_text())
    # Existing keys preserved; defaults added.
    assert settings["defaultProvider"] == "c137"
    assert settings["defaultModel"] == "keep-me"
    assert "compaction" in settings


def test_install_missing_extension_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        agent_install.install_pi(repo_root=tmp_path, agent_dir=tmp_path / "a", skip_npm=True)


def test_install_dispatch_planned_agent_raises() -> None:
    for agent in ("opencode", "claude", "cursor", "codex"):
        with pytest.raises(NotImplementedError):
            agent_install.install(agent)


def test_install_dispatch_unknown_agent_raises() -> None:
    with pytest.raises(ValueError):
        agent_install.install("nonsense")


# -- CLI wiring --------------------------------------------------------------
def test_cli_install_agent_pi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                              capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("PI_AGENT_DIR", str(tmp_path / "agent"))
    rc = cli.main(["--install-agent", "pi", "--symlink", "--skip-npm"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "installed the chroma-rag extension for 'pi'" in out
    assert (tmp_path / "agent" / "extensions" / "chroma-rag").is_symlink()


def test_cli_install_agent_planned_returns_3(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["--install-agent", "opencode"])
    assert rc == 3
    assert "not yet supported" in capsys.readouterr().err


def test_cli_install_agent_unknown_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["--install-agent", "bogus"])
    assert rc == 2
    assert "unknown agent" in capsys.readouterr().err


def test_print_result_next_steps(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = agent_install.install_pi(agent_dir=tmp_path / "agent", skip_npm=True)
    agent_install.print_result(result)
    out = capsys.readouterr().out
    assert "sphinx-build" in out
    assert "search_knowledge_base" in out
