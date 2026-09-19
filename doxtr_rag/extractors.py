"""Multi-format extractor & mount scanner (P3 — CONTRACTS §C2 producer).

Provides the extractor **surface** consumed during ingestion:
:func:`extract_file` (dispatch one file by extension) and :func:`scan_mount`
(recursively extract a mount directory). The ``builder-inited`` enumeration of
``rag_external_mounts`` — iterating configured mounts and resolving each mount's
provisional scope via ``SphinxRAGConfig.mount_scope`` (fail-closed to
``LOCAL_PRIVATE`` unless explicitly declared ``SHARED_ORG``) — is wired by P5's
pipeline, which then calls ``scan_mount(path, config.mount_scope(mount))`` per
mount. Keeping enumeration in P5 avoids two owners of the event lifecycle.

Supported formats:

- **PDF** — page-level chunks with ``{page}`` provenance.
- **Spreadsheets** (``.xlsx``, ``.ods``) — each sheet rendered as a Markdown
  table with ``{cell}`` provenance.
- **Presentations** (``.pptx``, ``.keynote``/``.key``) — one atomic chunk per
  slide combining slide title + body + speaker notes, with ``{slide}``
  provenance. ``.pptx`` is guaranteed; ``.keynote``/``.key`` is best-effort
  (skipped with a warning if unparseable).

Corrupt / password-protected / unreadable files fail gracefully (warn, never
crash the build). Password-protected content is never read into any chunk.
Mount content defaults to ``LOCAL_PRIVATE`` unless the mount is explicitly
declared ``SHARED_ORG``. Scope is **provisional**; P5 finalizes via
``most_restrictive``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterator

from doxtr_rag.schema import AccessControl, DocumentChunk

logger = logging.getLogger(__name__)

#: An extractor takes (path, scope) and yields DocumentChunks.
Extractor = Callable[[str, AccessControl], Iterator[DocumentChunk]]

_REGISTRY: dict[str, Extractor] = {}


class ExtractionError(Exception):
    """Raised internally when a file cannot be parsed; handled gracefully."""


def register(*extensions: str) -> Callable[[Extractor], Extractor]:
    """Register an extractor for one or more file extensions (lowercased)."""

    def _decorate(fn: Extractor) -> Extractor:
        for ext in extensions:
            _REGISTRY[ext.lower()] = fn
        return fn

    return _decorate


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base_metadata(
    path: str, title: str, scope: AccessControl, provenance: dict[str, Any], text: str = ""
) -> dict[str, Any]:
    return {
        "source_file": path,
        "section_title": title,
        "breadcrumb": [os.path.basename(path)] + ([title] if title else []),
        "access_scope": scope,
        "content_hash": _sha256(f"{path}:{provenance}:{title}:{text}"),
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
@register(".pdf")
def extract_pdf(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """Page-level PDF chunks. Password-protected PDFs route LOCAL_PRIVATE."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            # Attempt empty-password decrypt; if it stays locked, the file is
            # password-protected -> route LOCAL_PRIVATE and skip content (we
            # cannot and must not read protected content into a shared store).
            unlocked = False
            try:
                unlocked = reader.decrypt("") != 0
            except Exception:  # noqa: BLE001
                unlocked = False
            if not unlocked:
                logger.info(
                    "doxtr-rag: password-protected PDF %s -> LOCAL_PRIVATE (skipped)",
                    path,
                )
                return
    except (PdfReadError, OSError, ValueError) as exc:
        raise ExtractionError(f"unreadable PDF {path}: {exc}") from exc

    title = os.path.basename(path)
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - a bad page shouldn't kill the file
            text = ""
        if not text.strip():
            continue
        yield DocumentChunk(
            id=f"{path}#page={i + 1}",
            text=text,
            metadata=_base_metadata(
                path, title, scope, {"page": i + 1}, text
            ),
        )


# ---------------------------------------------------------------------------
# Spreadsheets
# ---------------------------------------------------------------------------
def _max_cols(rows: list[list[str]]) -> int:
    return max((len(r) for r in rows), default=0)


def _matrix_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [[(c if c is not None else "") for c in r] + [""] * (width - len(r)) for r in rows]
    header = norm[0]
    lines = ["| " + " | ".join(str(c) for c in header) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    for r in norm[1:]:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


@register(".xlsx")
def extract_xlsx(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """Each sheet -> Markdown table with cell provenance."""
    from openpyxl import load_workbook

    try:
        wb = load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"unreadable xlsx {path}: {exc}") from exc
    for sheet in wb.worksheets:
        rows = [[("" if c is None else str(c)) for c in row] for row in sheet.iter_rows(values_only=True)]
        md = _matrix_to_markdown(rows)
        if not md.strip():
            continue
        n_rows = len(rows)
        n_cols = max((len(r) for r in rows), default=0)
        yield DocumentChunk(
            id=f"{path}#sheet={sheet.title}",
            text=md,
            metadata=_base_metadata(
                path,
                sheet.title,
                scope,
                {"cell": f"{sheet.title}!A1:R{n_rows}C{n_cols}"},
                md,
            ),
        )


@register(".ods")
def extract_ods(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """OpenDocument spreadsheet -> Markdown table with cell provenance."""
    from odf.opendocument import load
    from odf.table import Table, TableRow, TableCell
    from odf.text import P

    try:
        doc = load(path)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"unreadable ods {path}: {exc}") from exc
    for table in doc.getElementsByType(Table):
        name = table.getAttribute("name") or "Sheet"
        rows: list[list[str]] = []
        for tr in table.getElementsByType(TableRow):
            cells: list[str] = []
            for tc in tr.getElementsByType(TableCell):
                repeat = int(tc.getAttribute("numbercolumnsrepeated") or 1)
                text = " ".join(str(p) for p in tc.getElementsByType(P))
                cells.extend([text] * repeat)
            rows.append(cells)
        md = _matrix_to_markdown(rows)
        if not md.strip():
            continue
        yield DocumentChunk(
            id=f"{path}#sheet={name}",
            text=md,
            metadata=_base_metadata(
                path,
                name,
                scope,
                {"cell": f"{name}!A1:R{len(rows)}C{_max_cols(rows)}"},
                md,
            ),
        )


# ---------------------------------------------------------------------------
# Presentations
# ---------------------------------------------------------------------------
@register(".pptx")
def extract_pptx(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """One atomic chunk per slide: title + body + speaker notes."""
    from pptx import Presentation

    try:
        prs = Presentation(path)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"unreadable pptx {path}: {exc}") from exc
    for idx, slide in enumerate(prs.slides):
        title = ""
        body_parts: list[str] = []
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            text = shape.text_frame.text
            if shape == slide.shapes.title:
                title = text
            elif text.strip():
                body_parts.append(text)
        notes = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text
        combined = "\n\n".join(
            part
            for part in (
                f"# {title}" if title else "",
                "\n".join(body_parts),
                f"Speaker notes:\n{notes}" if notes.strip() else "",
            )
            if part.strip()
        )
        if not combined.strip():
            continue
        yield DocumentChunk(
            id=f"{path}#slide={idx + 1}",
            text=combined,
            metadata=_base_metadata(
                path, title or f"Slide {idx + 1}", scope, {"slide": idx + 1}, combined
            ),
        )


@register(".keynote", ".key")
def extract_keynote(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """Best-effort Keynote extraction (proprietary bundle).

    There is no reliable cross-platform Keynote parser, so this is a spike:
    skip gracefully with a warning. ``.pptx`` is the guaranteed format.
    """
    logger.warning(
        "doxtr-rag: .keynote is best-effort and unparseable cross-platform; "
        "skipping %s",
        path,
    )
    return iter(())


# ---------------------------------------------------------------------------
# Word / OpenOffice
# ---------------------------------------------------------------------------
@register(".docx")
def extract_docx(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """Heading-aware structural extraction with breadcrumbs."""
    from docx import Document

    try:
        doc = Document(path)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"unreadable docx {path}: {exc}") from exc

    yield from _heading_chunks(
        path,
        scope,
        (
            (
                _heading_level(p.style.name if p.style else ""),
                p.text,
            )
            for p in doc.paragraphs
        ),
    )


def _heading_level(style_name: str) -> int:
    """Return the heading level from a style name, or 0 for body text."""
    name = (style_name or "").lower()
    if name.startswith("heading"):
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 1
    if name in ("title",):
        return 1
    return 0


@register(".odt")
def extract_odt(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """OpenDocument text -> heading-aware structural extraction."""
    from odf.opendocument import load
    from odf.text import H, P

    try:
        doc = load(path)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"unreadable odt {path}: {exc}") from exc

    def _iter() -> Iterator[tuple[int, str]]:
        for el in doc.text.childNodes:
            tag = getattr(el, "qname", ("", ""))[1] if hasattr(el, "qname") else ""
            text = str(el)
            if tag == "h":
                level = int(el.getAttribute("outlinelevel") or 1)
                yield (level, text)
            elif tag == "p":
                yield (0, text)

    yield from _heading_chunks(path, scope, _iter())


def _heading_chunks(
    path: str, scope: AccessControl, items: Iterator[tuple[int, str]]
) -> Iterator[DocumentChunk]:
    """Group (level, text) items into heading-bounded chunks with breadcrumbs."""
    stack: list[tuple[int, str]] = []  # (level, title)
    current_title = os.path.basename(path)
    buffer: list[str] = []
    idx = 0

    def _flush() -> Iterator[DocumentChunk]:
        nonlocal buffer, idx
        body = "\n\n".join(b for b in buffer if b.strip())
        if body.strip():
            crumb = [os.path.basename(path)] + [t for _, t in stack]
            yield DocumentChunk(
                id=f"{path}#sec={idx}",
                text=body,
                metadata={
                    "source_file": path,
                    "section_title": current_title,
                    "breadcrumb": crumb,
                    "access_scope": scope,
                    "content_hash": _sha256(f"{path}:{idx}:{body}"),
                    "provenance": {"docname": os.path.basename(path), "anchor": f"sec{idx}"},
                },
            )
        buffer = []
        idx += 1

    for level, text in items:
        if level > 0:  # a heading
            yield from _flush()
            # pop stack to parent level
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            current_title = text
        elif text.strip():
            buffer.append(text)
    yield from _flush()


# ---------------------------------------------------------------------------
# Mount scanning
# ---------------------------------------------------------------------------
def supported_extensions() -> frozenset[str]:
    """Return the set of registered file extensions."""
    return frozenset(_REGISTRY)


def extract_file(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """Extract chunks from a single file, dispatching by extension.

    Corrupt/unreadable files log a warning and yield nothing (graceful).
    """
    ext = Path(path).suffix.lower()
    extractor = _REGISTRY.get(ext)
    if extractor is None:
        return
    try:
        yield from extractor(path, scope)
    except ExtractionError as exc:
        logger.warning("doxtr-rag: skipping unreadable file: %s", exc)
    except Exception as exc:  # noqa: BLE001 - never crash the build
        logger.warning("doxtr-rag: unexpected extractor error for %s: %s", path, exc)


def scan_mount(
    mount_path: str, scope: AccessControl, *, confine: bool = True
) -> Iterator[DocumentChunk]:
    """Recursively scan a mount directory, extracting all supported files.

    A missing mount directory logs a warning and yields nothing. ``os.walk``
    does not follow symlinked directories, but a symlinked *file* can still
    launder an out-of-mount (e.g. private) file into this mount's ``scope``.
    When ``confine`` is True (default), every candidate file's realpath is
    asserted to stay within ``mount_path``; an escaping symlink is skipped
    fail-closed (logged), blocking symlink laundering into ``SHARED_ORG``.
    """
    from doxtr_rag.pathscope import path_is_confined

    if not os.path.isdir(mount_path):
        logger.warning("doxtr-rag: mount directory not found: %s", mount_path)
        return
    roots = [mount_path]
    for root, _dirs, files in os.walk(mount_path):
        # Never descend into a .doxtr/ runtime directory.
        if ".doxtr" in Path(root).parts:
            continue
        for name in sorted(files):
            fpath = os.path.join(root, name)
            if Path(fpath).suffix.lower() not in _REGISTRY:
                continue
            if confine and not path_is_confined(fpath, roots):
                logger.warning(
                    "doxtr-rag: symlink/path escapes mount %s -> skipped "
                    "(fail-closed): %s",
                    mount_path,
                    fpath,
                )
                continue
            yield from extract_file(fpath, scope)
