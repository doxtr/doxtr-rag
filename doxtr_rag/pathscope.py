"""Path-based scope routing with confinement (CONVENTIONS §Scope Resolution).

Deny-list before allow-list: any resolved path containing ``noter/`` or
``NDA/`` anywhere forces ``LOCAL_PRIVATE``; only ``source/`` (and mounts
explicitly declared shared) map to ``SHARED_ORG``. Every path is resolved with
``os.path.realpath`` and asserted to stay within an allowed root; a symlink
that escapes its root is skipped fail-closed.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath

from doxtr_rag.schema import AccessControl

#: Directory segments that always force LOCAL_PRIVATE (deny-list).
_PRIVATE_SEGMENTS: frozenset[str] = frozenset({"noter", "NDA"})

#: Directory segment that affirmatively signals SHARED_ORG (allow-list).
_SHARED_SEGMENT = "source"


def _segments(path: str) -> list[str]:
    normalized = path.replace(os.sep, "/")
    return [s for s in PurePosixPath(normalized).parts if s not in ("/", "")]


def path_is_confined(path: str, allowed_roots: list[str]) -> bool:
    """Return True if ``path``'s realpath stays within one of ``allowed_roots``.

    Blocks ``source/../NDA/x`` traversal and symlink laundering. If
    ``allowed_roots`` is empty, confinement is not enforced (returns True) so
    callers that do not supply roots are unaffected.
    """
    if not allowed_roots:
        return True
    real = os.path.realpath(path)
    for root in allowed_roots:
        real_root = os.path.realpath(root)
        try:
            common = os.path.commonpath([real, real_root])
        except ValueError:
            # Different drives / mixed absolute-relative; treat as escape.
            continue
        if common == real_root:
            return True
    return False


def provisional_scope_for_path(
    path: str, allowed_roots: list[str] | None = None
) -> AccessControl:
    """Return the provisional path-based scope, failing closed.

    Resolution order (CONVENTIONS):
        1. Confinement: a path that escapes its allowed root → ``LOCAL_PRIVATE``.
        2. Deny-list: any ``noter/``/``NDA/`` segment → ``LOCAL_PRIVATE``.
        3. Allow-list: a ``source/`` segment → ``SHARED_ORG``.
        4. Otherwise (ambiguous) → ``LOCAL_PRIVATE``.

    The returned scope is **provisional**; P5 finalizes via ``most_restrictive``.
    """
    roots = allowed_roots or []
    if roots and not path_is_confined(path, roots):
        return AccessControl.LOCAL_PRIVATE

    # Use the realpath segments so symlinked private files are caught.
    segs = _segments(os.path.realpath(path)) if roots else _segments(path)
    lower = {s.lower() for s in segs}

    # 2. deny-list (case-insensitive on 'noter'; 'NDA' matched case-insensitively too)
    if any(s.lower() in {p.lower() for p in _PRIVATE_SEGMENTS} for s in segs):
        return AccessControl.LOCAL_PRIVATE

    # 3. allow-list
    if _SHARED_SEGMENT in lower:
        return AccessControl.SHARED_ORG

    # 4. ambiguous -> fail closed
    return AccessControl.LOCAL_PRIVATE
