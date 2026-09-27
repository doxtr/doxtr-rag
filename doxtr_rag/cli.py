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
from typing import Any, Optional, Sequence

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
        choices=[
            "cache-info",
            "audit-info",
            "fetch-weights",
            "ingest",
            "mount-index",
            "mount-watch",
        ],
        help="inspection command, 'fetch-weights' to download the bge-m3 ONNX "
        "weights, 'ingest' to add a single file/URL to a store, 'mount-index' "
        "to build/refresh a portable per-mount sidecar index, or 'mount-watch' "
        "to keep it updated in the background",
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
    # ingest options
    ingest_grp = parser.add_argument_group("ingest")
    ingest_grp.add_argument(
        "--file", help="ingest: path to a local file to ingest"
    )
    ingest_grp.add_argument(
        "--url", help="ingest: URL to fetch and ingest (e.g. an xlink target)"
    )
    ingest_grp.add_argument(
        "--ingest-scope",
        choices=["shared", "private"],
        default="private",
        help="ingest: target canonical store (default: private)",
    )
    ingest_grp.add_argument(
        "--store",
        metavar="NAME",
        help="ingest: independent named database (created on demand); "
        "takes precedence over --ingest-scope",
    )
    ingest_grp.add_argument(
        "--source-url",
        help="ingest: source URL to record as provenance (implied by --url)",
    )
    ingest_grp.add_argument(
        "--tags", help="ingest: comma-separated tags to record as provenance"
    )
    ingest_grp.add_argument(
        "--doc-title", help="ingest: human title to record as provenance"
    )
    # mount-index / mount-watch options
    mount_grp = parser.add_argument_group("mount-index / mount-watch")
    mount_grp.add_argument(
        "--mount-path",
        metavar="DIR",
        help="mount-index/mount-watch: the mount directory. If it matches a "
        "configured rag_external_mounts entry, that mount's declared scope is "
        "honoured; otherwise it is treated as a bare LOCAL_PRIVATE path.",
    )
    mount_grp.add_argument(
        "--all-mounts",
        action="store_true",
        help="mount-index: index every mount declared in conf.py's "
        "rag_external_mounts (ignored by mount-watch).",
    )
    mount_grp.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="mount-watch: seconds between polls when 'watchdog' is absent "
        "(default 5.0).",
    )
    mount_grp.add_argument(
        "--once",
        action="store_true",
        help="mount-watch: run a single incremental pass and exit.",
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


def cmd_ingest(config: SphinxRAGConfig, args: argparse.Namespace) -> int:
    """Ingest a single file or URL into the selected store."""
    from doxtr_rag import ingest as ingest_mod

    if bool(args.file) == bool(args.url):
        print("[doxtr-rag] ingest requires exactly one of --file or --url", file=sys.stderr)
        return 2
    try:
        counts = ingest_mod.ingest(
            config=config,
            file=args.file,
            url=args.url,
            scope=args.ingest_scope,
            store=args.store,
            source_url=args.source_url,
            tags=args.tags,
            title=args.doc_title,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[doxtr-rag] ingest failed: {exc}", file=sys.stderr)
        return 1
    target = args.store or args.ingest_scope
    print(f"[doxtr-rag] ingested into {target}: {counts}")
    return 0


def _mounts_for_cli(
    config: SphinxRAGConfig, args: argparse.Namespace
) -> list[Any]:
    """Resolve the mount(s) a mount-* command should act on.

    Priority: an explicit ``--mount-path`` (honouring a matching configured
    mount's scope) > ``--all-mounts`` (every configured mount) > a single
    configured mount when exactly one exists.
    """
    from doxtr_rag.mount_index import find_mount

    if args.mount_path:
        matched = find_mount(config, args.mount_path)
        return [matched if matched is not None else args.mount_path]
    if getattr(args, "all_mounts", False):
        return list(config.rag_external_mounts)
    mounts = list(config.rag_external_mounts)
    if len(mounts) == 1:
        return mounts
    return []


def cmd_mount_index(config: SphinxRAGConfig, args: argparse.Namespace) -> int:
    """Build/refresh the portable sidecar index for one or more mounts."""
    from doxtr_rag import mount_index as mi

    mounts = _mounts_for_cli(config, args)
    if not mounts:
        print(
            "[doxtr-rag] mount-index: specify --mount-path DIR, use "
            "--all-mounts, or configure exactly one rag_external_mounts entry",
            file=sys.stderr,
        )
        return 2
    rc = 0
    for mount in mounts:
        try:
            stats = mi.index_mount(config, mount)
            label = mount.get("path") if isinstance(mount, dict) else mount
            print(f"[doxtr-rag] mount-index {label}: {stats}")
        except Exception as exc:  # noqa: BLE001
            print(f"[doxtr-rag] mount-index failed: {exc}", file=sys.stderr)
            rc = 1
    return rc


def cmd_mount_watch(config: SphinxRAGConfig, args: argparse.Namespace) -> int:
    """Watch a mount and keep its sidecar index up to date (single consumer)."""
    from doxtr_rag import mount_index as mi

    mounts = _mounts_for_cli(config, args)
    if len(mounts) != 1:
        print(
            "[doxtr-rag] mount-watch acts on ONE mount: pass --mount-path DIR "
            "(or configure exactly one rag_external_mounts entry)",
            file=sys.stderr,
        )
        return 2
    poll = args.poll_interval if args.poll_interval is not None else mi.DEFAULT_POLL_INTERVAL
    try:
        mi.watch_mount(config, mounts[0], poll_interval=poll, once=bool(args.once))
    except Exception as exc:  # noqa: BLE001
        print(f"[doxtr-rag] mount-watch failed: {exc}", file=sys.stderr)
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
    if args.command == "ingest":
        return cmd_ingest(config, args)
    if args.command == "mount-index":
        return cmd_mount_index(config, args)
    if args.command == "mount-watch":
        return cmd_mount_watch(config, args)
    if args.command == "cache-info":
        return cmd_cache_info(config)
    if args.command == "audit-info":
        return cmd_audit_info(config)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
