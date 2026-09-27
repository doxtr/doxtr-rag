"""Agent installer for the doxtr-rag query extension.

Installs the ``chroma-rag`` Pi Agent extension into a target agent's
configuration. Pi is fully supported and tailored to the ``doxtr/reactor``
devcontainer; other agents (opencode, Claude Code, Cursor, Codex) do **not**
consume Pi's TypeScript ``ExtensionAPI`` and would require a separate bridge
(an MCP server wrapping ``search_knowledge_base``, or each agent's native
plugin format) — those are reported as not-yet-implemented with the concrete
path forward rather than silently faked.

The Pi installer:
  * locates the extension source that **ships inside the package**
    (``doxtr_rag/pi_extension/``), with a fallback to a dev checkout,
  * installs it into the Pi agent dir (``$PI_AGENT_DIR`` or ``~/.pi/agent``)
    under ``extensions/chroma-rag`` (copied by default so a fresh install
    works from the wheel; ``--symlink`` for live dev edits against a checkout),
  * runs ``npm install`` for the extension's Node deps,
  * writes reactor-tailored defaults into ``settings.json`` and an env snippet.

Source of truth is the packaged ``doxtr_rag/pi_extension/`` directory; the
``.pi/extensions/chroma-rag/`` path is a *deployment target* the installer
populates — not the source.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SUPPORTED_AGENTS = ("pi",)
PLANNED_AGENTS = ("opencode", "claude", "cursor", "codex")

def _default_embed_url() -> str:
    """Return the fallback query-time embed URL from the stable embed port.

    Derived from the same stable port the Sphinx build and the standalone
    launcher bind (``SphinxRAGConfig.rag_embed_port`` / ``DOXTR_RAG_EMBED_PORT``)
    so the installer, the build and the launcher can never drift onto different
    ports. It is only a FALLBACK: the Pi extension prefers the ``embed_url``
    published in ``.doxtr/endpoints.json`` when present.
    """
    try:
        from doxtr_rag.config import SphinxRAGConfig

        port = SphinxRAGConfig().embed_port
    except Exception:  # noqa: BLE001 - never let installer defaults crash
        import os as _os

        try:
            port = int(_os.environ.get("DOXTR_RAG_EMBED_PORT", "8789"))
        except ValueError:
            port = 8789
    return f"http://127.0.0.1:{port}/embed"


#: Reactor-tailored defaults for the query-time environment.
REACTOR_DEFAULTS: dict[str, str] = {
    # Pre-baked bge-m3 ONNX weights location in the reactor image / mounted cache.
    "DOXTR_RAG_BGE_M3_ONNX_DIR": "/opt/models/bge-m3",
    # Query-time embedding endpoint fallback (must serve the pinned bge-m3 /
    # 1024-dim). Derived from the stable embed port; endpoints.json wins when
    # present.
    "DOXTR_RAG_EMBED_URL": _default_embed_url(),
    # Resolved store endpoints written by a Sphinx build (relative to the repo).
    "DOXTR_RAG_ENDPOINTS": ".doxtr/endpoints.json",
}


@dataclass
class InstallResult:
    """Outcome of an agent install (returned for tests / reporting)."""

    agent: str
    extension_dir: Path
    settings_path: Optional[Path]
    env_file: Optional[Path]
    actions: list[str]
    installed: bool


def packaged_extension_dir() -> Path:
    """Return the extension source that ships inside the installed package.

    This is the single source of truth: ``doxtr_rag/pi_extension/`` next to this
    module, present in the wheel/sdist as package data. Works from a plain
    ``pip install`` with no source checkout.
    """
    return Path(__file__).resolve().parent / "pi_extension"


def repo_extension_dir(repo_root: Optional[Path] = None) -> Path:
    """Return the extension source directory, preferring the packaged copy.

    Resolution order:
      1. an explicit ``repo_root`` override (tests),
      2. the packaged ``doxtr_rag/pi_extension/`` (normal installs),
      3. a dev-checkout fallback at ``<root>/.pi/extensions/chroma-rag``.
    """
    if repo_root is not None:
        return repo_root / ".pi" / "extensions" / "chroma-rag"
    packaged = packaged_extension_dir()
    if (packaged / "chroma-rag.ts").is_file():
        return packaged
    return Path(__file__).resolve().parents[1] / ".pi" / "extensions" / "chroma-rag"


def pi_agent_dir() -> Path:
    """Return the Pi agent config dir (``$PI_AGENT_DIR`` or ``~/.pi/agent``)."""
    env = os.environ.get("PI_AGENT_DIR")
    if env:
        return Path(os.path.expanduser(env))
    return Path.home() / ".pi" / "agent"


def _run_npm_install(ext_dir: Path, actions: list[str], *, skip: bool) -> None:
    if skip:
        actions.append("skipped npm install (--skip-npm)")
        return
    npm = shutil.which("npm")
    if npm is None:
        actions.append("npm not found on PATH; skipped dependency install")
        return
    if (ext_dir / "node_modules").is_dir():
        actions.append("node_modules present; skipped npm install")
        return
    subprocess.run([npm, "install"], cwd=str(ext_dir), check=True)
    actions.append("ran npm install")


def _merge_settings(settings_path: Path, actions: list[str]) -> None:
    """Ensure the extension dir is discoverable and reactor defaults are set.

    Only ADDS keys; never overwrites an existing provider/model/auth choice.
    """
    data: dict[str, object] = {}
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            actions.append(f"warning: could not parse {settings_path}; leaving it untouched")
            return
    changed = False
    # Reactor-tailored compaction defaults (only if absent).
    if "compaction" not in data:
        data["compaction"] = {
            "enabled": True,
            "reserveTokens": 16384,
            "keepRecentTokens": 20000,
        }
        changed = True
    if changed:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
        actions.append(f"updated {settings_path} (added defaults, preserved existing keys)")
    else:
        actions.append(f"{settings_path} already has defaults; no change")


def _write_env_snippet(agent_dir: Path, actions: list[str]) -> Path:
    """Write a sourceable env snippet with reactor-tailored defaults."""
    env_file = agent_dir / "doxtr-rag.env"
    lines = [
        "# doxtr-rag query-time environment (reactor defaults).",
        "# Source this before starting pi:  source ~/.pi/agent/doxtr-rag.env",
        "# Override any value to match your deployment.",
    ]
    for key, value in REACTOR_DEFAULTS.items():
        # Do not clobber a value already exported in the shell.
        lines.append(f'export {key}="${{{key}:-{value}}}"')
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    actions.append(f"wrote {env_file}")
    return env_file


def install_pi(
    *,
    repo_root: Optional[Path] = None,
    agent_dir: Optional[Path] = None,
    symlink: bool = False,
    skip_npm: bool = False,
) -> InstallResult:
    """Install the chroma-rag extension into the Pi agent (reactor-tailored).

    Copies the packaged extension source into the agent's extensions dir by
    default (works from a plain ``pip install``). ``symlink=True`` instead links
    to the source — only meaningful when installing from a writable dev
    checkout, so repo edits are live.
    """
    ext_src = repo_extension_dir(repo_root)
    if not (ext_src / "chroma-rag.ts").exists():
        raise FileNotFoundError(f"extension source not found at {ext_src}")

    target_agent_dir = agent_dir or pi_agent_dir()
    ext_dst = target_agent_dir / "extensions" / "chroma-rag"
    actions: list[str] = []

    # Replace any existing install cleanly.
    ext_dst.parent.mkdir(parents=True, exist_ok=True)
    if ext_dst.exists() or ext_dst.is_symlink():
        if ext_dst.is_symlink():
            ext_dst.unlink()
        else:
            shutil.rmtree(ext_dst)
        actions.append(f"removed existing {ext_dst}")

    if symlink:
        os.symlink(ext_src, ext_dst, target_is_directory=True)
        actions.append(f"symlinked {ext_dst} -> {ext_src} (dev mode)")
        # Dev mode: deps live alongside the source.
        _run_npm_install(ext_src, actions, skip=skip_npm)
    else:
        shutil.copytree(ext_src, ext_dst, ignore=shutil.ignore_patterns("node_modules"))
        actions.append(f"copied extension to {ext_dst}")
        # Install deps at the destination (the packaged source may be read-only).
        _run_npm_install(ext_dst, actions, skip=skip_npm)

    # Reactor-tailored settings + env snippet.
    settings_path = target_agent_dir / "settings.json"
    _merge_settings(settings_path, actions)
    env_file = _write_env_snippet(target_agent_dir, actions)

    return InstallResult(
        agent="pi",
        extension_dir=ext_dst,
        settings_path=settings_path,
        env_file=env_file,
        actions=actions,
        installed=True,
    )


def install(agent: str, **kwargs: object) -> InstallResult:
    """Dispatch to the per-agent installer."""
    agent = agent.lower()
    if agent == "pi":
        return install_pi(
            symlink=bool(kwargs.get("symlink", False)),
            skip_npm=bool(kwargs.get("skip_npm", False)),
        )
    if agent in PLANNED_AGENTS:
        raise NotImplementedError(
            f"agent {agent!r} is not yet supported: {agent} does not consume Pi's "
            "TypeScript ExtensionAPI. The RAG capability would be exposed to it via "
            "an MCP server wrapping the `search_knowledge_base` tool (opencode and "
            "Claude Code speak MCP) or that agent's native plugin format. This is a "
            "planned follow-up, not a config copy."
        )
    raise ValueError(
        f"unknown agent {agent!r}; supported: {', '.join(SUPPORTED_AGENTS)}; "
        f"planned: {', '.join(PLANNED_AGENTS)}"
    )


def print_result(result: InstallResult, out=None) -> None:  # type: ignore[no-untyped-def]
    if out is None:
        out = sys.stdout
    print(f"[doxtr-rag] installed the chroma-rag extension for '{result.agent}':", file=out)
    for action in result.actions:
        print(f"  - {action}", file=out)
    print("", file=out)
    print("Next steps (inside the doxtr/reactor container):", file=out)
    print("  1. Build the knowledge base so .doxtr/endpoints.json exists:", file=out)
    print("       sphinx-build -b html source/ _build/html", file=out)
    print("  2. Load the query-time env defaults:", file=out)
    print(f"       source {result.env_file}", file=out)
    print("  3. Start pi and ask questions; the agent uses search_knowledge_base.", file=out)
