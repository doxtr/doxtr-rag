"""Maintenance CLI for doxtr-rag (P5 — CONVENTIONS §Maintenance CLI).

Commands:
    doxtr-rag --rebuild [--scope shared|local|all]   drop & reindex a collection
    doxtr-rag cache-info                             inspect the SQLite cache
    doxtr-rag audit-info                             summarize the audit log

Entry point registered in ``pyproject.toml`` as ``doxtr-rag``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from doxtr_rag.cache import BuildCache
from doxtr_rag.config import SphinxRAGConfig


def _load_config(conf_path: Optional[str]) -> SphinxRAGConfig:
    """Load config from a conf.py-like module namespace, or defaults."""
    if not conf_path:
        return SphinxRAGConfig.from_mapping({})
    import runpy

    ns = runpy.run_path(conf_path)
    return SphinxRAGConfig.from_mapping(ns)


def cmd_rebuild(config: SphinxRAGConfig, scope: str) -> int:
    """Drop & recreate the collection(s) for the given scope."""
    from doxtr_rag.base_store import address_from_config
    from doxtr_rag.schema import AccessControl
    from doxtr_rag.storage import _build_client

    targets = []
    if scope in ("shared", "all"):
        targets.append((AccessControl.SHARED_ORG, config.shared_target()))
    if scope in ("local", "all"):
        targets.append((AccessControl.LOCAL_PRIVATE, config.local_target()))
    for ac, target in targets:
        address = address_from_config(ac, config)
        client = _build_client(target, address)
        try:
            client.delete_collection(address.collection)
            print(f"[doxtr-rag] dropped {ac.value} collection {address.collection}")
        except Exception as exc:  # noqa: BLE001
            print(f"[doxtr-rag] {ac.value}: nothing to drop ({exc})")
        client.get_or_create_collection(name=address.collection)
        print(f"[doxtr-rag] recreated {ac.value} collection {address.collection}")
    print("[doxtr-rag] rebuild complete; re-run sphinx-build to reindex.")
    return 0


def cmd_cache_info(config: SphinxRAGConfig) -> int:
    """Print a summary of the SQLite cache contents."""
    cache = BuildCache(config.rag_cache_db)
    try:
        conn = cache._conn  # noqa: SLF001 - CLI inspection
        rows = conn.execute(
            "SELECT resource_kind, COUNT(*) AS n FROM resources GROUP BY resource_kind"
        ).fetchall()
        total_chunks = conn.execute(
            "SELECT COUNT(*) AS n FROM resource_chunks"
        ).fetchone()["n"]
        print(f"cache: {config.rag_cache_db} (schema v{cache.schema_version})")
        for r in rows:
            print(f"  {r['resource_kind']}: {r['n']} resources")
        print(f"  chunks mapped: {total_chunks}")
    finally:
        cache.close()
    return 0


def cmd_audit_info(config: SphinxRAGConfig) -> int:
    """Summarize the security audit log."""
    from doxtr_rag.audit import AuditLog

    log = AuditLog(config.rag_audit_log_path)
    events = log.read_events()
    print(f"audit: {config.rag_audit_log_path} ({len(events)} events)")
    reasons: dict[str, int] = {}
    for ev in events:
        reasons[ev.reason] = reasons.get(ev.reason, 0) + 1
    for reason, n in sorted(reasons.items()):
        print(f"  {reason}: {n}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doxtr-rag", description="doxtr-rag maintenance CLI")
    parser.add_argument("--conf", help="path to a conf.py to load settings from")
    parser.add_argument("--rebuild", action="store_true", help="drop & reindex a collection")
    parser.add_argument(
        "--scope",
        choices=["shared", "local", "all"],
        default="all",
        help="which store scope to rebuild (default: all)",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["cache-info", "audit-info", "fetch-weights"],
        help="inspection command, or 'fetch-weights' to download the bge-m3 ONNX weights",
    )
    # install-agent options
    parser.add_argument(
        "--install-agent",
        metavar="AGENT",
        help="install the chroma-rag query extension into an agent (supported: pi)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="install-agent: symlink to the source (dev mode) instead of copying",
    )
    parser.add_argument(
        "--skip-npm",
        action="store_true",
        help="install-agent: skip running npm install",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="fetch-weights: re-download even if weights are already present",
    )
    return parser


def cmd_install_agent(agent: str, *, symlink: bool, skip_npm: bool) -> int:
    """Install the query extension into the given agent (reactor-tailored)."""
    from doxtr_rag import agent_install

    try:
        result = agent_install.install(agent, symlink=symlink, skip_npm=skip_npm)
    except NotImplementedError as exc:
        print(f"[doxtr-rag] {exc}", file=sys.stderr)
        return 3
    except (ValueError, FileNotFoundError) as exc:
        print(f"[doxtr-rag] {exc}", file=sys.stderr)
        return 2
    agent_install.print_result(result)
    return 0


def cmd_fetch_weights(force: bool) -> int:
    """Download the pinned bge-m3 ONNX weights (Python-only; no shell needed)."""
    from doxtr_rag import weights

    try:
        weights.fetch_weights(force=force)
    except Exception as exc:  # noqa: BLE001
        print(f"[doxtr-rag] fetch-weights failed: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.install_agent:
        return cmd_install_agent(
            args.install_agent, symlink=args.symlink, skip_npm=args.skip_npm
        )
    if args.command == "fetch-weights":
        return cmd_fetch_weights(args.force)
    config = _load_config(args.conf)
    if args.rebuild:
        return cmd_rebuild(config, args.scope)
    if args.command == "cache-info":
        return cmd_cache_info(config)
    if args.command == "audit-info":
        return cmd_audit_info(config)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
