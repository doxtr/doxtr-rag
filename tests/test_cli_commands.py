"""CLI dispatch tests for the ``ingest`` and ``mount-index`` / ``mount-watch``
subcommands added by the extractors+ingest and sidecar patches.

Drives the commands through :func:`doxtr_rag.cli.main` (argument parsing +
dispatch + return codes), using an embedded on-disk store and a fake 1024-dim
provider so no bge-m3 weights are required.
"""

from __future__ import annotations

import os

import pytest

import doxtr_rag.embedding_provider as ep_mod
from doxtr_rag import cli


class FakeProvider:
    dimension = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[(len(t) % 7) / 10.0] * 1024 for t in texts]


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    import doxtr_rag.storage as storage_mod

    monkeypatch.setattr(ep_mod, "EmbeddingProvider", FakeProvider)
    monkeypatch.setattr(storage_mod, "EmbeddingProvider", FakeProvider)


def _conf(tmp_path, *, mount: str | None = None) -> str:
    conf = tmp_path / "conf.py"
    lines = [
        "rag_local_store_mode = 'embedded'",
        f"rag_local_store_path = {os.path.join(str(tmp_path), 'priv')!r}",
        "rag_shared_store_mode = 'embedded'",
        f"rag_shared_store_uri = {os.path.join(str(tmp_path), 'shared')!r}",
    ]
    if mount is not None:
        lines.append(f"rag_external_mounts = [{{'path': {mount!r}}}]")
    conf.write_text("\n".join(lines) + "\n")
    return str(conf)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
def test_cli_ingest_requires_one_source(tmp_path, capsys) -> None:
    conf = _conf(tmp_path)
    rc = cli.main(["--conf", conf, "ingest"])
    assert rc == 2
    assert "exactly one of --file or --url" in capsys.readouterr().err


def test_cli_ingest_file_into_private(tmp_path, capsys) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("# Title\n\nBody.\n")
    conf = _conf(tmp_path)
    rc = cli.main(["--conf", conf, "ingest", "--file", str(doc)])
    assert rc == 0
    assert "ingested into private" in capsys.readouterr().out


def test_cli_ingest_missing_file_reports_error(tmp_path, capsys) -> None:
    conf = _conf(tmp_path)
    rc = cli.main(["--conf", conf, "ingest", "--file", str(tmp_path / "nope.md")])
    assert rc == 1
    assert "ingest failed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# mount-index
# ---------------------------------------------------------------------------
def test_cli_mount_index_requires_target(tmp_path, capsys) -> None:
    conf = _conf(tmp_path)  # no mounts configured
    rc = cli.main(["--conf", conf, "mount-index"])
    assert rc == 2
    assert "specify --mount-path" in capsys.readouterr().err


def test_cli_mount_index_explicit_path(tmp_path, capsys) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    (tmp_path / "vault" / "a.md").write_text("# A\n\nBody.\n")
    conf = _conf(tmp_path)
    rc = cli.main(["--conf", conf, "mount-index", "--mount-path", mount])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mount-index" in out and "ingested_files" in out


def test_cli_mount_index_all_mounts(tmp_path, capsys) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    (tmp_path / "vault" / "a.md").write_text("# A\n\nBody.\n")
    conf = _conf(tmp_path, mount=mount)
    rc = cli.main(["--conf", conf, "mount-index", "--all-mounts"])
    assert rc == 0


def test_cli_mount_index_single_configured_mount(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    (tmp_path / "vault" / "a.md").write_text("# A\n\nBody.\n")
    conf = _conf(tmp_path, mount=mount)
    # Exactly one configured mount -> resolved implicitly.
    rc = cli.main(["--conf", conf, "mount-index"])
    assert rc == 0


# ---------------------------------------------------------------------------
# mount-watch (--once)
# ---------------------------------------------------------------------------
def test_cli_mount_watch_once(tmp_path) -> None:
    mount = str(tmp_path / "vault")
    os.makedirs(mount)
    (tmp_path / "vault" / "a.md").write_text("# A\n\nBody.\n")
    conf = _conf(tmp_path)
    rc = cli.main(
        ["--conf", conf, "mount-watch", "--mount-path", mount, "--once"]
    )
    assert rc == 0
    from doxtr_rag import mount_index as mi

    assert os.path.isdir(mi.sidecar_chroma_path(mount))


def test_cli_mount_watch_requires_single_mount(tmp_path, capsys) -> None:
    conf = _conf(tmp_path)  # no mounts
    rc = cli.main(["--conf", conf, "mount-watch"])
    assert rc == 2
    assert "acts on ONE mount" in capsys.readouterr().err
