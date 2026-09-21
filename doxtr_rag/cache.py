"""Incremental build cache (CONTRACTS §C8, owned by P2, consulted by P5).

A SQLite-backed hash database at ``.doxtr/cache.db`` tracking change across
three hash domains — document nodes, external file contents, and resolved
xlink payloads — all SHA-256. It also maps each resource to the Chroma chunk
ids it produced so deleted resources can be pruned.

The cache is written by a single process (``build-finished`` runs only in the
main process), so it is safe under ``sphinx-build -j``; the extension declares
``parallel_write_safe = True`` (CONVENTIONS §Concurrency).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal, Optional

ResourceKind = Literal["node", "external_file", "xlink"]

#: Bump when the on-disk schema changes; migrations run in :meth:`_migrate`.
SCHEMA_VERSION = 1


class BuildCache:
    """SQLite hash store for incremental Sphinx builds (C8 surface).

    Args:
        db_path: Path to the SQLite database (created if absent). ``":memory:"``
            is accepted for tests.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            parent = Path(self._db_path).parent
            parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    # ------------------------------------------------------------------
    # Schema / migration
    # ------------------------------------------------------------------
    def _migrate(self) -> None:
        cur = self._conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL)")
        row = cur.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
        current = row["version"] if row else 0
        if current == 0:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS resources (
                    resource_id   TEXT PRIMARY KEY,
                    resource_kind TEXT NOT NULL
                        CHECK (resource_kind IN ('node','external_file','xlink')),
                    sha256_hash   TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS resource_chunks (
                    resource_id     TEXT NOT NULL,
                    chroma_chunk_id TEXT NOT NULL,
                    PRIMARY KEY (resource_id, chroma_chunk_id),
                    FOREIGN KEY (resource_id) REFERENCES resources(resource_id)
                        ON DELETE CASCADE
                )
                """
            )
            cur.execute("DELETE FROM schema_meta")
            cur.execute("INSERT INTO schema_meta (version) VALUES (?)", (SCHEMA_VERSION,))
        # Future migrations: if current < SCHEMA_VERSION, apply steps here.
        self._conn.commit()

    @property
    def schema_version(self) -> int:
        row = self._conn.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
        return int(row["version"]) if row else 0

    # ------------------------------------------------------------------
    # C8 API
    # ------------------------------------------------------------------
    def is_dirty(self, resource_id: str, sha256: str) -> bool:
        """Return True if ``resource_id`` is new or its hash changed.

        A resource unknown to the cache is dirty (must be (re)ingested).
        """
        row = self._conn.execute(
            "SELECT sha256_hash FROM resources WHERE resource_id = ?", (resource_id,)
        ).fetchone()
        if row is None:
            return True
        return str(row["sha256_hash"]) != sha256

    def update_cache(
        self,
        resource_id: str,
        resource_kind: ResourceKind,
        sha256: str,
        chroma_chunk_ids: list[str],
    ) -> None:
        """Record a resource's hash and the chunk ids it produced.

        Upserts the resource row and replaces its chunk-id mapping.
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO resources (resource_id, resource_kind, sha256_hash)
            VALUES (?, ?, ?)
            ON CONFLICT(resource_id) DO UPDATE SET
                resource_kind = excluded.resource_kind,
                sha256_hash   = excluded.sha256_hash
            """,
            (resource_id, resource_kind, sha256),
        )
        cur.execute("DELETE FROM resource_chunks WHERE resource_id = ?", (resource_id,))
        cur.executemany(
            "INSERT OR IGNORE INTO resource_chunks (resource_id, chroma_chunk_id) VALUES (?, ?)",
            [(resource_id, cid) for cid in chroma_chunk_ids],
        )
        self._conn.commit()

    def prune_deleted(self, seen_ids: set[str]) -> list[str]:
        """Delete cache rows for resources not in ``seen_ids``; return their chunk ids.

        Returns the list of ``chroma_chunk_id`` values that the caller (P5) must
        delete from the vector store. Cascade removes the mapping rows.
        """
        all_rows = self._conn.execute("SELECT resource_id FROM resources").fetchall()
        stale = [r["resource_id"] for r in all_rows if r["resource_id"] not in seen_ids]
        if not stale:
            return []
        placeholders = ",".join("?" for _ in stale)
        chunk_rows = self._conn.execute(
            f"SELECT chroma_chunk_id FROM resource_chunks WHERE resource_id IN ({placeholders})",
            stale,
        ).fetchall()
        chunk_ids = [r["chroma_chunk_id"] for r in chunk_rows]
        cur = self._conn.cursor()
        cur.execute(
            f"DELETE FROM resources WHERE resource_id IN ({placeholders})", stale
        )
        self._conn.commit()
        return chunk_ids

    def chunk_ids_for(self, resource_id: str) -> list[str]:
        """Return the chunk ids currently recorded for a resource."""
        rows = self._conn.execute(
            "SELECT chroma_chunk_id FROM resource_chunks WHERE resource_id = ?",
            (resource_id,),
        ).fetchall()
        return [r["chroma_chunk_id"] for r in rows]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "BuildCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
