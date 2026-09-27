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
import re
from pathlib import Path
from typing import Any, Callable, Iterator

from doxtr_rag.schema import AccessControl, DocumentChunk

logger = logging.getLogger(__name__)

#: An extractor takes (path, scope) and yields DocumentChunks.
Extractor = Callable[[str, AccessControl], Iterator[DocumentChunk]]

_REGISTRY: dict[str, Extractor] = {}

# ---------------------------------------------------------------------------
# Size-based chunk capping
# ---------------------------------------------------------------------------
# The structural extractors above split only on natural boundaries (headings,
# pages, slides, sheets). A boundary-sparse document -- e.g. a meeting-
# transcript .docx with no Word heading styles, or a long section between two
# headings -- can therefore collapse into a single very large chunk. One
# oversized chunk produces one averaged embedding over thousands of tokens,
# which (a) retrieves poorly (any single statement is diluted) and (b) may
# silently exceed the embedding model's input window (bge-m3: 8192 tokens),
# truncating and dropping the tail from the vector entirely.
#
# To bound this, every emitted chunk is passed through a size cap in the
# central ``extract_file`` dispatch point (so ALL formats benefit, not just
# .docx). Sections at or under the cap are emitted unchanged. Oversized
# sections are re-split on paragraph/sentence boundaries into parts of at most
# ``max_chars`` with a small overlap, preserving each part's metadata and
# provenance and giving each a stable, distinct id/hash.
#
# The cap is expressed in characters (a dependency-free proxy for tokens; ~4
# chars/token for prose). Defaults target ~512 tokens with ~15% overlap, the
# common RAG best-practice range for modern embedding models. Both are
# tunable via environment variables so operators can adjust without a code
# change:
#   DOXTR_RAG_MAX_CHUNK_CHARS     (default 2000, ~512 tokens; <=0 disables)
#   DOXTR_RAG_CHUNK_OVERLAP_CHARS (default 300,  ~15% overlap)
DEFAULT_MAX_CHUNK_CHARS = 2000
DEFAULT_CHUNK_OVERLAP_CHARS = 300

# Split preferentially on paragraph breaks, then sentence-ish boundaries.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _int_env(var: str, default: int) -> int:
    """Read a non-negative int from the environment, falling back on error."""
    raw = os.environ.get(var)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "doxtr-rag: %s=%r is not an integer; using default %d", var, raw, default
        )
        return default


def _max_chunk_chars() -> int:
    return _int_env("DOXTR_RAG_MAX_CHUNK_CHARS", DEFAULT_MAX_CHUNK_CHARS)


def _chunk_overlap_chars() -> int:
    return max(0, _int_env("DOXTR_RAG_CHUNK_OVERLAP_CHARS", DEFAULT_CHUNK_OVERLAP_CHARS))


def _resolve_cap(max_chars: int | None, overlap: int | None) -> tuple[int, int]:
    """Resolve effective (max_chars, overlap), env override > arg > default.

    The env vars DOXTR_RAG_MAX_CHUNK_CHARS / DOXTR_RAG_CHUNK_OVERLAP_CHARS win
    when set (operator quick-tuning). Otherwise the explicit argument is used
    (threaded from ``conf.py`` via ``SphinxRAGConfig``); ``None`` falls back to
    the module default.
    """
    env_max = os.environ.get("DOXTR_RAG_MAX_CHUNK_CHARS")
    if env_max is not None and env_max.strip() != "":
        eff_max = _int_env("DOXTR_RAG_MAX_CHUNK_CHARS", DEFAULT_MAX_CHUNK_CHARS)
    else:
        eff_max = DEFAULT_MAX_CHUNK_CHARS if max_chars is None else max_chars
    env_ov = os.environ.get("DOXTR_RAG_CHUNK_OVERLAP_CHARS")
    if env_ov is not None and env_ov.strip() != "":
        eff_ov = _int_env("DOXTR_RAG_CHUNK_OVERLAP_CHARS", DEFAULT_CHUNK_OVERLAP_CHARS)
    else:
        eff_ov = DEFAULT_CHUNK_OVERLAP_CHARS if overlap is None else overlap
    return eff_max, max(0, eff_ov)


def _segments(text: str, max_chars: int) -> list[str]:
    """Break ``text`` into atomic segments no larger than ``max_chars``.

    Prefers paragraph boundaries, then sentence boundaries, and only as a last
    resort hard-cuts a run of text with no usable boundary (e.g. a single very
    long line). Segments are the packing unit for :func:`_pack`.
    """
    out: list[str] = []
    for para in _PARAGRAPH_RE.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            out.append(para)
            continue
        # Paragraph too big: fall back to sentence boundaries.
        for sent in _SENTENCE_RE.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= max_chars:
                out.append(sent)
                continue
            # Sentence still too big: hard-cut on the character grid.
            for i in range(0, len(sent), max_chars):
                out.append(sent[i : i + max_chars])
    return out


def _pack(text: str, max_chars: int, overlap: int) -> list[str]:
    """Pack segments into parts of at most ``max_chars`` with char ``overlap``.

    Overlap is applied by prefixing each part (after the first) with the tail
    of the previous part, so a statement straddling a boundary is not lost.
    The overlap prefix counts toward ``max_chars``: segments are packed into a
    budget of ``max_chars - overlap`` so the final part (prefix + body) never
    exceeds ``max_chars``. ``overlap`` is clamped below ``max_chars``.
    """
    # Clamp overlap to at most half the cap: an overlap approaching (or
    # exceeding) ``max_chars`` degenerates packing and can push a part over
    # the cap. Half is the practical upper bound for useful overlap.
    overlap = max(0, min(overlap, max_chars // 2))
    # Reserve room for the overlap prefix AND its "\n\n" separator so the
    # final part (tail + separator + body) never exceeds ``max_chars``.
    sep = 2 if overlap else 0
    body_budget = max(1, max_chars - overlap - sep)
    segments = _segments(text, body_budget)
    if not segments:
        return []
    parts: list[str] = []
    cur = ""
    for seg in segments:
        if not cur:
            cur = seg
        elif len(cur) + 2 + len(seg) <= body_budget:
            cur = f"{cur}\n\n{seg}"
        else:
            parts.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = f"{tail}\n\n{seg}" if tail else seg
    if cur:
        parts.append(cur)
    return parts


def _split_oversized(chunk: DocumentChunk, max_chars: int, overlap: int) -> Iterator[DocumentChunk]:
    """Yield ``chunk`` unchanged if within ``max_chars``, else split parts.

    Each derived part keeps the original metadata but gets a distinct id
    (``<id>#part=N``), a ``part``/``part_count`` provenance annotation and a
    per-part ``content_hash`` so re-splits are stable and traceable. Capping
    is disabled when ``max_chars <= 0``.
    """
    if max_chars <= 0 or len(chunk.text) <= max_chars:
        yield chunk
        return

    parts = _pack(chunk.text, max_chars, overlap)
    if len(parts) <= 1:
        # Nothing usefully splittable (e.g. cap larger than content after
        # boundary packing); emit as-is rather than fabricate empty parts.
        yield chunk
        return

    base_meta = dict(chunk.metadata)
    base_prov = dict(base_meta.get("provenance") or {})
    total = len(parts)
    logger.info(
        "doxtr-rag: split oversized chunk %s (%d chars) into %d parts "
        "(max_chars=%d, overlap=%d)",
        chunk.id,
        len(chunk.text),
        total,
        max_chars,
        overlap,
    )
    for i, part_text in enumerate(parts):
        prov = dict(base_prov)
        prov["part"] = i
        prov["part_count"] = total
        meta = dict(base_meta)
        meta["provenance"] = prov
        meta["content_hash"] = _sha256(f"{chunk.id}#part={i}:{part_text}")
        yield DocumentChunk(id=f"{chunk.id}#part={i}", text=part_text, metadata=meta)


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
    """Heading-aware structural extraction with breadcrumbs.

    Primary strategy: iterate ``doc.paragraphs`` (standard python-docx API).
    Fallback: if the paragraph pass yields no body content (e.g. the document
    stores text in non-standard runs, content-controls, or table cells that
    python-docx does not surface as paragraphs), reconstruct paragraph text
    directly from the raw ``word/document.xml`` XML (grouping ``<w:t>`` run
    nodes under their ``<w:p>`` parent) and feed the paragraphs through the
    same heading-chunk pipeline so section structure is preserved.
    """
    from docx import Document

    try:
        doc = Document(path)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"unreadable docx {path}: {exc}") from exc

    # --- primary pass: standard paragraph API ---
    items = list(
        (
            _heading_level(p.style.name if p.style else ""),
            p.text,
        )
        for p in doc.paragraphs
    )
    # If any real Heading style exists we can distinguish headings from body,
    # so the primary pass is trustworthy as long as it surfaced body text.
    # If NO real Heading styles exist we cannot tell a section label from body
    # text via the API alone, so fall through to the raw-XML pass (which also
    # recovers text python-docx does not surface as paragraphs, e.g. table
    # cells).
    any_real_heading = any(level > 0 for level, _ in items)
    has_body = any_real_heading and any(
        level == 0 and text.strip() for level, text in items
    )

    if has_body:
        yield from _heading_chunks(path, scope, iter(items))
        return

    # --- fallback: raw XML paragraph reconstruction ---
    logger.debug(
        "doxtr-rag: docx %s has no body paragraphs via python-docx API; "
        "falling back to raw XML <w:t> extraction",
        path,
    )
    yield from _extract_docx_xml(path, scope, items, any_real_heading)


def _extract_docx_xml(
    path: str,
    scope: AccessControl,
    api_items: list[tuple[int, str]],
    any_real_heading: bool,
) -> Iterator[DocumentChunk]:
    """Extract text from docx by reconstructing paragraphs from document.xml.

    The paragraph style information from the python-docx API pass is used to
    detect heading levels; body text is sourced from the raw XML run text so
    content stored in non-standard paragraph styles (e.g. 'InfoLine') or in
    table cells is also captured. Raw ``<w:t>`` run nodes are grouped under
    their ``<w:p>`` parent to reconstruct whole-paragraph text before matching
    against the heading map -- this is essential because python-docx splits a
    paragraph across multiple runs at formatting boundaries, so a heading label
    is often stored as several adjacent ``<w:t>`` nodes rather than one.

    ``any_real_heading`` is passed in from :func:`extract_docx` (rather than
    recomputed here) so the two passes cannot desync on how they classify the
    document.
    """
    import zipfile
    from collections import Counter
    from lxml import etree

    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    # Build a heading map from the API pass.  When the document uses custom
    # styles (e.g. 'InfoLine') that _heading_level does not recognise as
    # headings, the paragraph-API items will all have level 0.  In that case
    # treat every non-empty API item as a level-1 heading so the section
    # labels still produce structural chunks rather than collapsing into one
    # flat body block.  Matching is by exact reconstructed-paragraph text.
    heading_map: dict[str, int] = {}
    # Track how many times each heading label is expected to appear so a label
    # is consumed at most as many times as the API pass saw it -- this stops a
    # body paragraph that happens to equal a label from being promoted, and
    # stops a single label from re-firing for every identical body string.
    remaining: Counter[str] = Counter()
    for level, text in api_items:
        t = text.strip()
        if not t:
            continue
        if any_real_heading:
            if level > 0:
                heading_map[t] = level
                remaining[t] += 1
        else:
            # All items are body-level in the API pass -> promote each to h1.
            heading_map[t] = 1
            remaining[t] += 1

    try:
        with zipfile.ZipFile(path) as z:
            with z.open("word/document.xml") as f:
                tree = etree.parse(f)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"unreadable docx XML {path}: {exc}") from exc

    ns = {"w": W}

    def _paragraph_text(para: etree._Element) -> str:
        """Join all run text under a <w:p> into a single paragraph string."""
        parts = [t.text for t in para.findall(".//w:t", ns) if t.text]
        return "".join(parts).strip()

    # Reconstruct (level, text) items in document order, one per <w:p>.  A
    # paragraph is treated as a heading only when its full text matches a
    # heading label AND that label still has an unconsumed occurrence.
    def _iter() -> Iterator[tuple[int, str]]:
        budget = Counter(remaining)
        for para in tree.iter(f"{{{W}}}p"):
            text = _paragraph_text(para)
            if not text:
                continue
            level = heading_map.get(text, 0)
            if level > 0 and budget[text] > 0:
                budget[text] -= 1
                yield (level, text)
            else:
                yield (0, text)

    yield from _heading_chunks(path, scope, _iter())


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


# ---------------------------------------------------------------------------
# Plain-text family (.txt) and lightweight markup (.md/.rst/.org/.html/.xml)
# ---------------------------------------------------------------------------
# These formats carry no binary structure, so extraction is text-first:
#   - .md / .rst / .org       -> heading-aware chunking from their markup
#   - .html / .htm / .xml     -> tags stripped to text, then heading-aware
#     (HTML) or paragraph (XML) chunking
#   - .txt and anything else   -> blank-line-delimited paragraph chunking
# All of them still pass through the central size cap in ``extract_file``.


def _read_text_file(path: str) -> str:
    """Read a text file as UTF-8, tolerating undecodable bytes."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError as exc:
        raise ExtractionError(f"unreadable text file {path}: {exc}") from exc


def _paragraph_items(text: str) -> Iterator[tuple[int, str]]:
    """Yield (level=0, paragraph) items split on blank lines."""
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if block:
            yield (0, block)


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_MD_SETEXT = re.compile(r"^(=+|-+)\s*$")
_ORG_HEADING = re.compile(r"^(\*{1,6})\s+(.*\S)\s*$")
_RST_UNDERLINE = re.compile(r"^([=\-`:.'\"~^_*+#])\1{1,}\s*$")


def _markdown_items(text: str) -> Iterator[tuple[int, str]]:
    """Yield (level, text) items from Markdown: ATX (#) and setext headings."""
    lines = text.splitlines()
    in_fence = False
    prev: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            if prev is not None:
                yield (0, prev)
                prev = None
            yield (0, line)
            continue
        if in_fence:
            yield (0, line)
            continue
        m = _MD_HEADING.match(line)
        if m:
            if prev is not None:
                yield (0, prev)
                prev = None
            yield (len(m.group(1)), m.group(2).strip())
            continue
        if prev is not None and _MD_SETEXT.match(line) and prev.strip():
            level = 1 if line.strip().startswith("=") else 2
            yield (level, prev.strip())
            prev = None
            continue
        if prev is not None:
            yield (0, prev)
        prev = line
    if prev is not None:
        yield (0, prev)


def _org_items(text: str) -> Iterator[tuple[int, str]]:
    """Yield (level, text) items from Org-mode ``*``-prefixed headings."""
    for line in text.splitlines():
        m = _ORG_HEADING.match(line)
        if m:
            yield (len(m.group(1)), m.group(2).strip())
        else:
            yield (0, line)


def _rst_items(text: str) -> Iterator[tuple[int, str]]:
    """Yield (level, text) items from reStructuredText section underlines.

    RST section levels are defined by first-seen adornment character rather
    than a fixed order; a running map assigns a stable level per character.
    """
    lines = text.splitlines()
    char_levels: dict[str, int] = {}
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < n else ""
        if (
            line.strip()
            and _RST_UNDERLINE.match(nxt)
            and len(nxt.strip()) >= len(line.strip())
        ):
            char = nxt.strip()[0]
            level = char_levels.setdefault(char, len(char_levels) + 1)
            yield (level, line.strip())
            i += 2  # consume title + underline
            continue
        yield (0, line)
        i += 1


def _html_to_items(text: str) -> Iterator[tuple[int, str]]:
    """Parse HTML and yield (level, text): h1-h6 become headings, blocks text."""
    from lxml import html as lxml_html

    try:
        root = lxml_html.fromstring(text)
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"unparseable HTML: {exc}") from exc
    for bad in root.xpath("//script | //style | //noscript | //template"):
        bad.getparent().remove(bad)
    for el in root.iter():
        tag = (el.tag if isinstance(el.tag, str) else "").lower()
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            title = " ".join((el.text_content() or "").split())
            if title:
                yield (int(tag[1]), title)
        elif tag in ("p", "li", "td", "th", "pre", "blockquote", "dd", "dt", "figcaption"):
            body = " ".join((el.text_content() or "").split())
            if body:
                yield (0, body)


def _xml_to_items(text: str) -> Iterator[tuple[int, str]]:
    """Flatten XML to text: yield non-empty element text as paragraphs."""
    from lxml import etree

    try:
        root = etree.fromstring(text.encode("utf-8"))
    except Exception:  # noqa: BLE001 - fall back to raw paragraph split
        yield from _paragraph_items(text)
        return
    for el in root.iter():
        for piece in (el.text, el.tail):
            if piece and piece.strip():
                yield (0, " ".join(piece.split()))


@register(".md", ".markdown")
def extract_markdown(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """Markdown -> heading-aware structural extraction."""
    yield from _heading_chunks(path, scope, _markdown_items(_read_text_file(path)))


@register(".rst")
def extract_rst(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """reStructuredText -> heading-aware structural extraction."""
    yield from _heading_chunks(path, scope, _rst_items(_read_text_file(path)))


@register(".org")
def extract_org(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """Org-mode -> heading-aware structural extraction."""
    yield from _heading_chunks(path, scope, _org_items(_read_text_file(path)))


@register(".html", ".htm", ".xhtml")
def extract_html(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """HTML -> tags stripped, heading-aware structural extraction."""
    yield from _heading_chunks(path, scope, _html_to_items(_read_text_file(path)))


@register(".xml")
def extract_xml(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """XML -> flattened element text, paragraph chunking."""
    yield from _heading_chunks(path, scope, _xml_to_items(_read_text_file(path)))


@register(".txt", ".text")
def extract_plaintext(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """Plain text -> blank-line-delimited paragraph chunking."""
    yield from _heading_chunks(path, scope, _paragraph_items(_read_text_file(path)))


@register(".csv", ".tsv")
def extract_csv(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """Delimited text -> one Markdown table, matching the spreadsheet extractors.

    Keeps CSV/TSV consistent with ``.xlsx``/``.ods`` (structured rows rendered
    as a Markdown table) instead of collapsing a blank-line-free file into one
    opaque paragraph. Delimiter is sniffed, falling back to ``\t`` for ``.tsv``
    and ``,`` otherwise.
    """
    import csv
    import io

    raw = _read_text_file(path)
    delimiter = "\t" if Path(path).suffix.lower() == ".tsv" else ","
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",\t;|")
        delimiter = dialect.delimiter
    except Exception:  # noqa: BLE001 - keep the extension default delimiter
        pass
    rows = [row for row in csv.reader(io.StringIO(raw), delimiter=delimiter)]
    if not rows:
        return
    title = os.path.basename(path)
    yield DocumentChunk(
        id=f"{path}#table",
        text=_matrix_to_markdown(rows),
        metadata=_base_metadata(
            path, title, scope, {"cell": "A1"}, _matrix_to_markdown(rows)
        ),
    )


# ---------------------------------------------------------------------------
# Diagram sources (.drawio / .puml / .d2) -- LABEL-ONLY extraction
# ---------------------------------------------------------------------------
# Diagram source files are mostly syntax, geometry and styling. Ingesting the
# raw source pollutes the vector store with boilerplate (skinparam, mxGeometry,
# style strings) that both retrieves poorly and dilutes the few meaningful
# node/edge labels. So these extractors emit ONLY the human-readable strings:
# node names, edge/arrow labels and notes -- one deduplicated chunk per file.


_INLINE_TAG = re.compile(r"<[^>]+>")


def _clean_label(raw: str) -> str:
    """Normalize a diagram label: unescape HTML, drop inline tags, collapse ws."""
    import html as _html

    text = _html.unescape(raw)
    text = _INLINE_TAG.sub(" ", text)
    text = _html.unescape(text)  # values are sometimes double-escaped
    text = text.replace("\\n", " ")  # diagram line breaks
    return " ".join(text.split())


def _diagram_chunk(
    path: str, scope: AccessControl, labels: list[str]
) -> Iterator[DocumentChunk]:
    """Emit one chunk of newline-joined, order-preserving unique labels."""
    seen: set[str] = set()
    ordered: list[str] = []
    for label in labels:
        label = label.strip()
        if label and label not in seen:
            seen.add(label)
            ordered.append(label)
    if not ordered:
        return
    body = "\n".join(ordered)
    title = os.path.basename(path)
    yield DocumentChunk(
        id=f"{path}#labels",
        text=body,
        metadata=_base_metadata(
            path, title, scope, {"docname": title, "anchor": "labels"}, body
        ),
    )


@register(".drawio")
def extract_drawio(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """draw.io/diagrams.net -> mxCell ``value`` labels only (styling stripped).

    draw.io files are XML; the human-readable text lives in ``mxCell@value``
    attributes, HTML-encoded and often wrapping inline HTML (``<br>``, ``<i>``,
    ``<span style=...>``). Geometry/style attributes are ignored entirely.
    Compressed (``<diagram>`` base64+deflate) content is skipped gracefully.
    """
    from lxml import etree

    raw = _read_text_file(path)
    try:
        root = etree.fromstring(raw.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"unparseable drawio {path}: {exc}") from exc
    labels: list[str] = []
    for cell in root.iter():
        tag = cell.tag if isinstance(cell.tag, str) else ""
        if tag.split("}")[-1] not in ("mxCell", "object", "UserObject"):
            continue
        value = cell.get("value") or cell.get("label")
        if value:
            cleaned = _clean_label(value)
            if cleaned:
                labels.append(cleaned)
    yield from _diagram_chunk(path, scope, labels)


_PUML_ALIAS = re.compile(
    r'^\s*(?:participant|actor|boundary|control|entity|database|collections|queue|component|node|cloud|folder|frame|package|rectangle|usecase|class|interface|state|object)\s+(?:"([^"]+)"|(\w+))',
    re.IGNORECASE,
)
_PUML_ARROW = re.compile(r"^\s*[\w\"]+.*?(?:-+>|<-+|\.+>|<\.+|-+|\.\.)\s*[\w\"]+.*?:\s*(.+?)\s*$")
_PUML_NOTE_INLINE = re.compile(r"^\s*note\b.*?:\s*(.+?)\s*$", re.IGNORECASE)
_PUML_TITLE = re.compile(r"^\s*title\s+(.+?)\s*$", re.IGNORECASE)
# WBS / mindmap nodes: leading * or +/- depth markers, then optional [#color]
# and <color:...>/<style> markup, then the node text.
_PUML_WBS = re.compile(r"^\s*([*+\-]{1,})\s*(?:_\s*)?(.+?)\s*$")
# Gantt tasks: [Task Name] as [alias] ... or [Task Name] happens/requires ...
_PUML_GANTT_TASK = re.compile(r"^\s*\[([^\]]+)\]")
_PUML_COLOR_TAG = re.compile(r"^\s*\[#[0-9A-Fa-f]+\]\s*")
_PUML_DIRECTIVE = re.compile(
    r"^\s*(?:@start\w+|@end\w+|skinparam|scale|!|hide|show|autonumber|header|footer|legend|end\s*legend|alt\b|else\b|opt\b|loop\b|par\b|break\b|critical\b|group\b|end\b|activate|deactivate|destroy|create|return|box|end\s*box|printscale|project\s+starts|saturday\b|sunday\b|monday\b|tuesday\b|wednesday\b|thursday\b|friday\b|<style>|</style>)",
    re.IGNORECASE,
)


@register(".puml", ".plantuml", ".pu", ".iuml")
def extract_plantuml(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """PlantUML -> participant names, arrow/message labels and note text only.

    Directives (``@startuml``, ``skinparam``, ``scale``, control blocks) are
    dropped. Aliased participants contribute their display name; arrows
    contribute their message label; ``note`` blocks contribute their body.
    """
    raw = _read_text_file(path)
    labels: list[str] = []
    in_note = False
    note_buf: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if in_note:
            if low.startswith("end note") or low == "endnote":
                joined = _clean_label(" ".join(note_buf))
                if joined:
                    labels.append(joined)
                note_buf = []
                in_note = False
            else:
                note_buf.append(stripped)
            continue
        if low.startswith("note") and not _PUML_NOTE_INLINE.match(stripped):
            # multi-line note block (note left/right/over ... \n ... end note)
            in_note = True
            continue
        if low.startswith("note"):
            m_inline = _PUML_NOTE_INLINE.match(stripped)
            if m_inline:
                cleaned = _clean_label(m_inline.group(1))
                if cleaned:
                    labels.append(cleaned)
            continue
        m_alias = _PUML_ALIAS.match(stripped)
        if m_alias:
            name = m_alias.group(1) or m_alias.group(2)
            cleaned = _clean_label(name)
            if cleaned:
                labels.append(cleaned)
            continue
        m_title = _PUML_TITLE.match(stripped)
        if m_title:
            cleaned = _clean_label(m_title.group(1))
            if cleaned:
                labels.append(cleaned)
            continue
        if _PUML_DIRECTIVE.match(stripped):
            continue
        m_arrow = _PUML_ARROW.match(stripped)
        if m_arrow:
            cleaned = _clean_label(m_arrow.group(1))
            if cleaned:
                labels.append(cleaned)
            continue
        # WBS / mindmap node (*, **, ***). Strip [#color] prefix; _clean_label
        # then removes <color:...>/<i> style markup and collapses \n.
        m_wbs = _PUML_WBS.match(stripped)
        if m_wbs and set(m_wbs.group(1)) <= {"*", "+", "-"}:
            node = _PUML_COLOR_TAG.sub("", m_wbs.group(2))
            cleaned = _clean_label(node)
            if cleaned:
                labels.append(cleaned)
            continue
        # Gantt task: [Task Name] ...
        m_gantt = _PUML_GANTT_TASK.match(stripped)
        if m_gantt:
            cleaned = _clean_label(m_gantt.group(1))
            if cleaned:
                labels.append(cleaned)
    yield from _diagram_chunk(path, scope, labels)


_D2_CONNECTION = re.compile(r"->|<-|--|<->")
_D2_KEY_LABEL = re.compile(r"^\s*([\w.\- ]+?)\s*:\s*(.+?)\s*(?:\{\s*)?$")
_D2_BARE_KEY = re.compile(r"^\s*([A-Za-z0-9_][\w.\- ]*?)\s*\{\s*$")
_D2_TRAILING_BLOCK = re.compile(r"\s*\{[^}]*\}?\s*$")
_D2_RESERVED = re.compile(
    r"^(?:grid-rows|grid-columns|grid-gap|direction|shape|style|width|height|near|icon|constraint|source-arrowhead|target-arrowhead|label|link|tooltip|classes|vars)\b",
    re.IGNORECASE,
)


@register(".d2")
def extract_d2(path: str, scope: AccessControl) -> Iterator[DocumentChunk]:
    """D2 -> node display labels and connection labels only.

    Extracts ``key: Label`` display names and connection labels
    (``a -> b: label``). Reserved keywords (``grid-rows``, ``style``, ...) and
    pure structural braces are ignored. Bare container keys contribute their
    key name as a fallback label.
    """
    raw = _read_text_file(path)
    labels: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped in ("{", "}"):
            continue
        if _D2_RESERVED.match(stripped):
            continue
        # connection with optional label: a -> b: label
        if _D2_CONNECTION.search(stripped):
            _, _, after = stripped.partition(":")
            cleaned = _clean_label(after)
            if cleaned:
                labels.append(cleaned)
            continue
        m_kv = _D2_KEY_LABEL.match(stripped)
        if m_kv and not _D2_RESERVED.match(m_kv.group(1)):
            value = _D2_TRAILING_BLOCK.sub("", m_kv.group(2)).strip()
            cleaned = _clean_label(value)
            if cleaned:
                labels.append(cleaned)
            continue
        m_bare = _D2_BARE_KEY.match(stripped)
        if m_bare:
            cleaned = _clean_label(m_bare.group(1))
            if cleaned:
                labels.append(cleaned)
    yield from _diagram_chunk(path, scope, labels)


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


def extract_file(
    path: str,
    scope: AccessControl,
    *,
    max_chunk_chars: int | None = None,
    chunk_overlap_chars: int | None = None,
) -> Iterator[DocumentChunk]:
    """Extract chunks from a single file, dispatching by extension.

    Every emitted chunk is passed through a format-agnostic size cap
    (:func:`_split_oversized`) so boundary-sparse documents cannot collapse
    into a single oversized chunk (see the module-level notes). Corrupt or
    unreadable files log a warning and yield nothing (graceful).

    ``max_chunk_chars`` / ``chunk_overlap_chars`` are threaded from
    ``conf.py`` (``SphinxRAGConfig``). ``None`` falls back to the module
    default; the DOXTR_RAG_* env vars override either when set.
    """
    ext = Path(path).suffix.lower()
    extractor = _REGISTRY.get(ext)
    if extractor is None:
        return
    max_chars, overlap = _resolve_cap(max_chunk_chars, chunk_overlap_chars)
    try:
        for chunk in extractor(path, scope):
            yield from _split_oversized(chunk, max_chars, overlap)
    except ExtractionError as exc:
        logger.warning("doxtr-rag: skipping unreadable file: %s", exc)
    except Exception as exc:  # noqa: BLE001 - never crash the build
        logger.warning("doxtr-rag: unexpected extractor error for %s: %s", path, exc)


def scan_mount(
    mount_path: str,
    scope: AccessControl,
    *,
    confine: bool = True,
    max_chunk_chars: int | None = None,
    chunk_overlap_chars: int | None = None,
) -> Iterator[DocumentChunk]:
    """Recursively scan a mount directory, extracting all supported files.

    A missing mount directory logs a warning and yields nothing. ``os.walk``
    does not follow symlinked directories, but a symlinked *file* can still
    launder an out-of-mount (e.g. private) file into this mount's ``scope``.
    When ``confine`` is True (default), every candidate file's realpath is
    asserted to stay within ``mount_path``; an escaping symlink is skipped
    fail-closed (logged), blocking symlink laundering into ``SHARED_ORG``.

    ``max_chunk_chars`` / ``chunk_overlap_chars`` are forwarded to
    :func:`extract_file` (threaded from ``conf.py``); see its docstring for
    resolution order.
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
            yield from extract_file(
                fpath,
                scope,
                max_chunk_chars=max_chunk_chars,
                chunk_overlap_chars=chunk_overlap_chars,
            )
