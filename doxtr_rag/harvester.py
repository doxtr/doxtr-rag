"""Sphinx AST harvester (P2 — CONTRACTS §C2 producer).

A ``doctree-resolved`` listener that traverses ``docutils`` sections, builds
breadcrumb trails from ancestor section titles, resolves ``:ref:``/``:doc:``/
``:term:`` cross-references to their target titles, chunks structurally along
header/section boundaries (never fixed character counts), and assigns a
**provisional** path-based scope.

Chunks are yielded as :class:`~doxtr_rag.schema.DocumentChunk`; raw AST objects
never leak into the output. Scope is provisional — P5 finalizes via
``most_restrictive``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterator, Optional

from docutils import nodes

from doxtr_rag.pathscope import provisional_scope_for_path
from doxtr_rag.schema import AccessControl, DocumentChunk

#: A resolver maps a cross-reference (reftype, target) to a human title, or None.
XRefResolver = Callable[[str, str], Optional[str]]


def resource_id_for(docname: str) -> str:
    """Return the canonical cache ``resource_id`` for a document node (C8).

    This is the SINGLE source of the node-grain resource id so P5 does not
    reverse-engineer it from ``chunk.id``. ``resource_kind`` is ``"node"``.
    """
    return f"node:{docname}"


def document_sha256(chunks: list[DocumentChunk]) -> str:
    """Deterministic document-level hash folded from a doc's chunk hashes.

    P5 uses this as the resource-level SHA-256 for ``is_dirty``/``update_cache``
    so the node grouping + hash derivation lives in ONE place (P2), not P5.
    """
    parts = sorted(str(c.metadata.get("content_hash", "")) for c in chunks)
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _section_title(section: nodes.section) -> str:
    for child in section.children:
        if isinstance(child, nodes.title):
            return str(child.astext())
    return ""


def _resolve_xrefs(node: nodes.Element, resolver: Optional[XRefResolver]) -> None:
    """In-place: annotate pending cross-references with resolved titles.

    Handles ``:ref:``/``:doc:``/``:term:`` reference nodes. Uses the resolver to
    look up a target title; unresolved references keep their original text.
    """
    if resolver is None:
        return
    for ref in node.findall(nodes.reference):
        reftype = ref.get("reftype") or (
            "ref" if ref.get("refid") or ref.get("refuri") else ""
        )
        target = ref.get("reftarget") or ref.get("refid") or ref.get("refuri") or ""
        if not target:
            continue
        title = resolver(reftype, str(target))
        if title:
            ref["_resolved_title"] = title
    # pending_xref (unresolved domain refs) also carry reftype/reftarget.
    for pending in node.findall(nodes.Element):
        if pending.__class__.__name__ != "pending_xref":
            continue
        reftype = str(pending.get("reftype") or "")
        target = str(pending.get("reftarget") or "")
        if reftype in ("ref", "doc", "term") and target:
            title = resolver(reftype, target)
            if title:
                pending["_resolved_title"] = title


def _section_body_text(section: nodes.section, resolver: Optional[XRefResolver]) -> str:
    """Collect the text of a section EXCLUDING nested subsections + own title."""
    parts: list[str] = []
    for child in section.children:
        if isinstance(child, (nodes.title, nodes.section)):
            continue
        if isinstance(child, nodes.Element):
            _resolve_xrefs(child, resolver)
            # Prefer any resolved cross-reference titles inline.
            for ref in list(child.findall(nodes.reference)) + [
                p
                for p in child.findall(nodes.Element)
                if p.__class__.__name__ == "pending_xref"
            ]:
                resolved = ref.get("_resolved_title")
                if resolved and ref.astext() != resolved:
                    ref.replace_self(nodes.Text(resolved))
        parts.append(child.astext())
    return "\n\n".join(p for p in parts if p.strip())


def _document_title(doctree: nodes.document) -> str:
    for child in doctree.children:
        if isinstance(child, nodes.title):
            return str(child.astext())
    return ""


def _document_intro_text(
    doctree: nodes.document, resolver: Optional[XRefResolver]
) -> str:
    """Body text that are direct document children (before the first section).

    Occurs when docutils promotes a single top-level title to the document
    title, leaving intro paragraphs as direct document children.
    """
    parts: list[str] = []
    for child in doctree.children:
        if isinstance(child, (nodes.title, nodes.section)):
            continue
        if isinstance(child, nodes.Element):
            _resolve_xrefs(child, resolver)
            for ref in list(child.findall(nodes.reference)) + [
                p
                for p in child.findall(nodes.Element)
                if p.__class__.__name__ == "pending_xref"
            ]:
                resolved = ref.get("_resolved_title")
                if resolved and ref.astext() != resolved:
                    ref.replace_self(nodes.Text(resolved))
            parts.append(child.astext())
    return "\n\n".join(p for p in parts if p.strip())


def harvest_doctree(
    doctree: nodes.document,
    *,
    docname: str,
    source_path: str,
    allowed_roots: list[str] | None = None,
    resolver: Optional[XRefResolver] = None,
) -> Iterator[DocumentChunk]:
    """Yield structural :class:`DocumentChunk`s for a resolved doctree.

    Args:
        doctree: The resolved ``docutils`` document.
        docname: The Sphinx docname (used in provenance).
        source_path: The on-disk source path (used for path-based scope).
        allowed_roots: Realpath confinement roots (CONVENTIONS §Path confinement).
        resolver: Optional cross-reference title resolver for ``:ref:``/``:doc:``/
            ``:term:``.

    Chunking is **structural**: one chunk per section, carrying that section's
    own body text (nested subsections become their own chunks). Breadcrumbs are
    the ancestor section titles. Scope is provisional (path-based).
    """
    scope = provisional_scope_for_path(source_path, allowed_roots)

    def _emit(
        *, title: str, body: str, crumb: list[str], anchor: Optional[str]
    ) -> Optional[DocumentChunk]:
        if not body.strip():
            return None
        content_hash = _sha256(f"{docname}:{'/'.join(crumb)}:{body}")
        chunk_id = f"{docname}#{anchor or _sha256(title or docname)[:12]}"
        return DocumentChunk(
            id=chunk_id,
            text=body,
            metadata={
                "source_file": source_path,
                "section_title": title,
                "breadcrumb": crumb,
                "access_scope": scope,
                "content_hash": content_hash,
                "provenance": {"docname": docname, "anchor": anchor or ""},
            },
        )

    def _walk(
        section: nodes.section, breadcrumb: list[str]
    ) -> Iterator[DocumentChunk]:
        title = _section_title(section)
        crumb = breadcrumb + ([title] if title else [])
        body = _section_body_text(section, resolver)
        anchor = section.get("ids", [None])[0] if section.get("ids") else None
        chunk = _emit(title=title, body=body, crumb=crumb, anchor=anchor)
        if chunk is not None:
            yield chunk
        for sub in section.children:
            if isinstance(sub, nodes.section):
                yield from _walk(sub, crumb)

    # Case 1: a promoted document title (single top-level header). The document
    # itself acts as the root section; its title anchors the breadcrumb and its
    # direct-child paragraphs form the intro chunk.
    doc_title = _document_title(doctree)
    root_crumb: list[str] = [doc_title] if doc_title else []
    if doc_title:
        intro = _document_intro_text(doctree, resolver)
        anchor = doctree.get("ids", [None])[0] if doctree.get("ids") else None
        chunk = _emit(title=doc_title, body=intro, crumb=root_crumb, anchor=anchor)
        if chunk is not None:
            yield chunk

    top_sections = [c for c in doctree.children if isinstance(c, nodes.section)]
    for sec in top_sections:
        yield from _walk(sec, root_crumb)


def harvest_document(
    doctree: nodes.document,
    *,
    docname: str,
    source_path: str,
    allowed_roots: list[str] | None = None,
    resolver: Optional[XRefResolver] = None,
) -> tuple[str, str, list[DocumentChunk]]:
    """Harvest a doctree and return the cache-ready ``(resource_id, sha, chunks)``.

    This is the cohesive seam P5 consumes for cache-aware upsert/prune: the
    node-grain ``resource_id`` (:func:`resource_id_for`) and document-level hash
    (:func:`document_sha256`) are derived here so P5 does not reinvent them.
    """
    chunks = list(
        harvest_doctree(
            doctree,
            docname=docname,
            source_path=source_path,
            allowed_roots=allowed_roots,
            resolver=resolver,
        )
    )
    return resource_id_for(docname), document_sha256(chunks), chunks


def make_sphinx_resolver(app: Any, env: Any) -> XRefResolver:
    """Build an :data:`XRefResolver` backed by a Sphinx ``env`` (best-effort).

    Resolves ``:doc:`` to a document title, ``:ref:`` to a labeled target title,
    and ``:term:`` to a glossary term. Any lookup failure returns ``None`` so the
    harvester keeps the original reference text (graceful degradation).
    """

    def _resolve(reftype: str, target: str) -> Optional[str]:
        try:
            if reftype == "doc":
                return env.titles.get(target).astext() if target in env.titles else None
            std = env.get_domain("std")
            if reftype == "ref":
                labels = getattr(std, "labels", {})
                if target in labels:
                    _, _, title = labels[target]
                    return str(title)
                anon = getattr(std, "anonlabels", {})
                if target in anon:
                    docn, _ = anon[target]
                    return env.titles.get(docn).astext() if docn in env.titles else None
                return None
            if reftype == "term":
                # Glossary terms live in std domain objects keyed as ('term', name).
                objects = getattr(std, "objects", {})
                if ("term", target) in objects:
                    return target
                return None
        except Exception:  # noqa: BLE001 - graceful degradation
            return None
        return None

    return _resolve
