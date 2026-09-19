"""P2 tests for the harvester + path-based provisional scope (CONTRACTS §C2)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from docutils import nodes
from docutils.core import publish_doctree
from docutils.parsers.rst import roles

from doxtr_rag.harvester import harvest_doctree
from doxtr_rag.pathscope import path_is_confined, provisional_scope_for_path
from doxtr_rag.schema import AccessControl

# Register generic roles so test RST with :ref:/:doc:/:term: parses cleanly.
for _r in ("ref", "doc", "term"):
    roles.register_generic_role(_r, nodes.emphasis)


def _doctree(rst: str) -> nodes.document:
    return publish_doctree(rst)


# -- path scope --------------------------------------------------------------
def test_scope_source_shared() -> None:
    assert provisional_scope_for_path("p/source/a.rst") is AccessControl.SHARED_ORG


@pytest.mark.parametrize("seg", ["noter", "NDA"])
def test_scope_private_dirs(seg: str) -> None:
    assert (
        provisional_scope_for_path(f"p/{seg}/a.rst") is AccessControl.LOCAL_PRIVATE
    )


def test_scope_ambiguous_failclosed() -> None:
    assert provisional_scope_for_path("p/misc/a.rst") is AccessControl.LOCAL_PRIVATE


def test_deny_before_allow_source_then_nda() -> None:
    # A path with both source/ and NDA/ must be LOCAL_PRIVATE (deny wins).
    assert (
        provisional_scope_for_path("p/source/NDA/a.rst") is AccessControl.LOCAL_PRIVATE
    )


def test_confinement_blocks_traversal(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "NDA"
    outside.mkdir()
    escape = str(root / ".." / "NDA" / "secret.rst")
    assert path_is_confined(escape, [str(root)]) is False
    # And the scope of an escaping path fails closed.
    assert provisional_scope_for_path(escape, [str(root)]) is AccessControl.LOCAL_PRIVATE


def test_symlink_escape_failclosed(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    secret_dir = tmp_path / "NDA"
    secret_dir.mkdir()
    secret = secret_dir / "s.rst"
    secret.write_text("private")
    link = root / "link.rst"
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported")
    # Realpath of the symlink lands in NDA/ -> LOCAL_PRIVATE.
    assert (
        provisional_scope_for_path(str(link), [str(root)]) is AccessControl.LOCAL_PRIVATE
    )


def test_confinement_no_roots_is_permissive() -> None:
    assert path_is_confined("/any/path", []) is True


# -- harvesting --------------------------------------------------------------
def test_structural_chunking_and_breadcrumbs() -> None:
    rst = """
Top Title
=========

Intro paragraph.

Sub A
-----

Body A.

Deep
~~~~

Deep body.
"""
    chunks = list(
        harvest_doctree(_doctree(rst), docname="index", source_path="p/source/index.rst")
    )
    titles = [c.metadata["section_title"] for c in chunks]
    assert titles == ["Top Title", "Sub A", "Deep"]
    # Breadcrumb accumulates ancestor titles.
    deep = next(c for c in chunks if c.metadata["section_title"] == "Deep")
    assert deep.metadata["breadcrumb"] == ["Top Title", "Sub A", "Deep"]
    # Structural: Top chunk holds only its intro, not the subsection bodies.
    top = next(c for c in chunks if c.metadata["section_title"] == "Top Title")
    assert "Body A." not in top.text
    assert "Intro paragraph." in top.text


def test_no_raw_ast_leaks() -> None:
    rst = "Title\n=====\n\nBody.\n"
    for c in harvest_doctree(_doctree(rst), docname="d", source_path="p/source/d.rst"):
        assert isinstance(c.text, str)
        assert isinstance(c.metadata["breadcrumb"], list)
        # provenance is the {docname, anchor} shape from C2.
        assert set(c.metadata["provenance"]) == {"docname", "anchor"}


def test_provisional_scope_from_path() -> None:
    rst = "T\n=\n\nB.\n"
    private = list(
        harvest_doctree(_doctree(rst), docname="n", source_path="p/noter/n.rst")
    )
    assert all(c.access_scope is AccessControl.LOCAL_PRIVATE for c in private)


def test_xref_resolution_ref_doc_term() -> None:
    rst = """
Title
=====

See ref-here and doc-here and term-here.
"""
    dt = _doctree(rst)
    # Simulate resolved references by injecting reference nodes.
    para = next(dt.findall(nodes.paragraph))
    ref = nodes.reference("", "ref-here", refid="mylabel")
    ref["reftype"] = "ref"
    ref["reftarget"] = "mylabel"
    para += ref

    calls: list[tuple[str, str]] = []

    def resolver(reftype: str, target: str) -> str | None:
        calls.append((reftype, target))
        return {"mylabel": "Resolved Label"}.get(target)

    chunks = list(
        harvest_doctree(
            dt, docname="index", source_path="p/source/index.rst", resolver=resolver
        )
    )
    assert ("ref", "mylabel") in calls
    body = " ".join(c.text for c in chunks)
    assert "Resolved Label" in body


def test_empty_document_yields_nothing() -> None:
    chunks = list(harvest_doctree(_doctree(""), docname="e", source_path="p/source/e.rst"))
    assert chunks == []


# -- cache-ready seam (resource_id + document hash) --------------------------
def test_harvest_document_returns_cache_seam() -> None:
    from doxtr_rag.harvester import document_sha256, harvest_document, resource_id_for

    rst = "T\n=\n\nBody.\n\nSub\n---\n\nMore.\n"
    rid, sha, chunks = harvest_document(
        _doctree(rst), docname="index", source_path="p/source/index.rst"
    )
    assert rid == resource_id_for("index") == "node:index"
    assert len(sha) == 64
    assert len(chunks) >= 2
    # Deterministic: re-harvesting identical input yields the same doc hash.
    _, sha2, chunks2 = harvest_document(
        _doctree(rst), docname="index", source_path="p/source/index.rst"
    )
    assert sha == sha2
    assert document_sha256(chunks) == document_sha256(chunks2)


# -- Sphinx resolver + pending_xref ------------------------------------------
class _FakeStd:
    def __init__(self) -> None:
        self.labels = {"lbl": ("doc1", "lbl", "Label Title")}
        self.anonlabels = {"anon": ("doc2", "anon")}
        self.objects = {("term", "myterm"): ("gloss", "id", "term")}


class _Title:
    def __init__(self, text: str) -> None:
        self._t = text

    def astext(self) -> str:
        return self._t


class _FakeEnv:
    def __init__(self) -> None:
        self.titles = {"doc1": _Title("Doc One"), "doc2": _Title("Doc Two")}
        self._std = _FakeStd()

    def get_domain(self, name: str) -> _FakeStd:
        return self._std


def test_make_sphinx_resolver() -> None:
    from doxtr_rag.harvester import make_sphinx_resolver

    env = _FakeEnv()
    resolve = make_sphinx_resolver(app=None, env=env)
    assert resolve("doc", "doc1") == "Doc One"
    assert resolve("ref", "lbl") == "Label Title"
    assert resolve("ref", "anon") == "Doc Two"
    assert resolve("ref", "missing") is None
    assert resolve("term", "myterm") == "myterm"
    assert resolve("term", "nope") is None
    assert resolve("doc", "nope") is None


def test_make_sphinx_resolver_degrades_on_error() -> None:
    from doxtr_rag.harvester import make_sphinx_resolver

    class _Boom:
        titles: dict[str, object] = {}

        def get_domain(self, name: str) -> object:
            raise RuntimeError("boom")

    resolve = make_sphinx_resolver(app=None, env=_Boom())
    assert resolve("ref", "x") is None


def test_pending_xref_resolution() -> None:
    rst = "Title\n=====\n\nBody.\n"
    dt = _doctree(rst)
    para = next(dt.findall(nodes.paragraph))

    class _Pending(nodes.Element):
        pass

    _Pending.__name__ = "pending_xref"
    pend = _Pending()
    pend["reftype"] = "term"
    pend["reftarget"] = "myterm"
    pend += nodes.Text("myterm")
    para += pend

    seen: list[tuple[str, str]] = []

    def resolver(rt: str, tgt: str) -> str | None:
        seen.append((rt, tgt))
        return "Glossary Term" if tgt == "myterm" else None

    list(
        harvest_doctree(
            dt, docname="index", source_path="p/source/index.rst", resolver=resolver
        )
    )
    assert ("term", "myterm") in seen
