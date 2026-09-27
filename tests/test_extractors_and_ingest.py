"""Tests for the text/diagram extractors + ``doxtr-rag ingest`` command,
plus the chunk-cap + stable-port changes carried in the same combined patch.

Run with:  python -m pytest test_extractors_and_ingest.py
(or plain:  python test_extractors_and_ingest.py)
"""
from __future__ import annotations

import os

from doxtr_rag import extractors as ex
from doxtr_rag.schema import AccessControl, DocumentChunk


# ---------------------------------------------------------------------------
# Chunk cap (carried over from the chunk-cap patch, folded into this combined
# patch so the maintainer can validate everything in one place).
# ---------------------------------------------------------------------------
def _mk(text: str) -> DocumentChunk:
    return DocumentChunk(
        id="/x/f.docx#sec=0",
        text=text,
        metadata={
            "source_file": "/x/f.docx",
            "section_title": "T",
            "breadcrumb": ["f.docx"],
            "access_scope": AccessControl.LOCAL_PRIVATE,
            "content_hash": "h",
            "provenance": {"docname": "f.docx", "anchor": "sec0"},
        },
    )


_BIG = "\n\n".join(f"Paragraph number {i} with some filler words here." for i in range(30))


def test_small_chunk_unchanged():
    out = list(ex._split_oversized(_mk("short text"), 200, 40))
    assert len(out) == 1 and out[0].id == "/x/f.docx#sec=0"


def test_big_chunk_splits_within_cap():
    out = list(ex._split_oversized(_mk(_BIG), 200, 40))
    assert len(out) > 1 and all(len(o.text) <= 200 for o in out)


def test_cap_disabled():
    assert len(list(ex._split_oversized(_mk(_BIG), 0, 40))) == 1


# ---------------------------------------------------------------------------
# Text-based extractors (per extension).
# ---------------------------------------------------------------------------
def _write(tmp, name, content):
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(content)
    return p


def _extract(path):
    return list(ex.extract_file(path, AccessControl.LOCAL_PRIVATE))


def test_registered_extensions_present():
    exts = ex.supported_extensions()
    for e in (
        ".md", ".markdown", ".rst", ".org", ".html", ".htm", ".xhtml",
        ".xml", ".txt", ".text", ".csv", ".tsv",
        ".drawio", ".puml", ".plantuml", ".d2",
    ):
        assert e in exts, f"missing extractor for {e}"


def test_markdown_headings(tmp_path):
    p = _write(str(tmp_path), "a.md", "# Title\n\nIntro.\n\n## Sub\n\nBody.\n")
    chunks = _extract(p)
    assert len(chunks) == 2
    assert chunks[0].metadata["breadcrumb"] == ["a.md", "Title"]
    assert chunks[1].metadata["breadcrumb"] == ["a.md", "Title", "Sub"]


def test_rst_underline_sections(tmp_path):
    rst = "Main Title\n==========\n\nIntro text.\n\nSub\n---\n\nSub body.\n"
    p = _write(str(tmp_path), "a.rst", rst)
    chunks = _extract(p)
    assert len(chunks) == 2
    assert chunks[0].text.strip() == "Intro text."
    assert chunks[1].metadata["breadcrumb"] == ["a.rst", "Main Title", "Sub"]


def test_org_headings(tmp_path):
    p = _write(str(tmp_path), "a.org", "* A\nBody A.\n** B\nBody B.\n")
    chunks = _extract(p)
    assert len(chunks) == 2 and chunks[0].text.strip() == "Body A."


def test_html_strips_script_style(tmp_path):
    html = (
        "<html><head><style>x{}</style></head><body>"
        "<h1>H</h1><p>Para.</p><script>bad()</script></body></html>"
    )
    p = _write(str(tmp_path), "a.html", html)
    body = " ".join(c.text for c in _extract(p))
    assert "Para." in body and "bad()" not in body and "x{}" not in body


def test_xml_flattens_text(tmp_path):
    p = _write(str(tmp_path), "a.xml", "<root><item>Alpha</item><n>Beta</n></root>")
    body = " ".join(c.text for c in _extract(p))
    assert "Alpha" in body and "Beta" in body


def test_txt_paragraphs(tmp_path):
    p = _write(str(tmp_path), "a.txt", "One.\nTwo.\n\nSecond para.\n")
    chunks = _extract(p)
    assert chunks and "Second para." in chunks[0].text


def test_csv_renders_markdown_table(tmp_path):
    p = _write(str(tmp_path), "a.csv", "name,role\nAlice,PO\nBob,Dev\n")
    chunks = _extract(p)
    assert len(chunks) == 1
    t = chunks[0].text
    assert "| name | role |" in t and "| Alice | PO |" in t


# ---------------------------------------------------------------------------
# Diagram label-only extractors.
# ---------------------------------------------------------------------------
def test_drawio_labels_only(tmp_path):
    drawio = (
        '<mxfile><diagram><mxGraphModel><root>'
        '<mxCell id="1" value="Login Service" vertex="1" '
        'style="rounded=1;whiteSpace=wrap;"><mxGeometry x="1" y="2"/></mxCell>'
        '<mxCell id="2" value="Direct access&lt;br&gt;blocked" edge="1"/>'
        '</root></mxGraphModel></diagram></mxfile>'
    )
    p = _write(str(tmp_path), "a.drawio", drawio)
    body = _extract(p)[0].text
    assert "Login Service" in body
    assert "Direct access blocked" in body
    assert "rounded" not in body and "mxGeometry" not in body


def test_plantuml_participants_arrows_notes(tmp_path):
    puml = (
        "@startuml\nskinparam backgroundColor transparent\n"
        'participant "Relying Party" as RP\n'
        "RP -> op: POST /token\n"
        "note right: keep this\n@enduml\n"
    )
    p = _write(str(tmp_path), "a.puml", puml)
    body = _extract(p)[0].text
    assert "Relying Party" in body
    assert "POST /token" in body
    assert "keep this" in body
    assert "skinparam" not in body and "@startuml" not in body


def test_plantuml_wbs_nodes(tmp_path):
    puml = (
        "@startwbs\ntitle Roles\n"
        "* Root\n"
        "**[#0e6eec] <color:#ffffff>Architect</color>\\n<color:#ffffff>Alice</color>\n"
        "@endwbs\n"
    )
    p = _write(str(tmp_path), "a.puml", puml)
    body = _extract(p)[0].text
    assert "Roles" in body and "Root" in body
    assert "Architect" in body and "Alice" in body
    assert "#0e6eec" not in body and "<color" not in body


def test_d2_labels_and_connections(tmp_path):
    d2 = (
        "scl: Swisscom Login {\n"
        "  grid-rows: 2\n"
        "  login: Login Service {}\n"
        "  nginx -> login: forwards\n"
        "}\n"
    )
    p = _write(str(tmp_path), "a.d2", d2)
    body = _extract(p)[0].text
    assert "Swisscom Login" in body and "Login Service" in body
    assert "forwards" in body
    assert "grid-rows" not in body and "{}" not in body


# ---------------------------------------------------------------------------
# Additional extractor branch coverage (setext, drawio object@label, plantuml
# gantt + multi-line note, d2 bare key, .tsv delimiter default, split #part=).
# ---------------------------------------------------------------------------
def test_markdown_setext_headings(tmp_path):
    # Setext (underline) headings: '===' -> h1, '---' -> h2.
    md = "Big Title\n=========\n\nIntro.\n\nSection Two\n-----------\n\nBody.\n"
    p = _write(str(tmp_path), "s.md", md)
    chunks = _extract(p)
    assert chunks[0].metadata["breadcrumb"] == ["s.md", "Big Title"]
    assert chunks[1].metadata["breadcrumb"] == ["s.md", "Big Title", "Section Two"]


def test_drawio_object_label(tmp_path):
    # Text can live on <object label=...> / <UserObject label=...>, not just
    # mxCell@value.
    xml = (
        '<mxGraphModel><root>'
        '<object label="Order Service" id="1"><mxCell/></object>'
        '<UserObject label="Ships &amp; Handling" id="2"><mxCell/></UserObject>'
        '</root></mxGraphModel>'
    )
    p = _write(str(tmp_path), "o.drawio", xml)
    body = _extract(p)[0].text
    assert "Order Service" in body
    assert "Ships & Handling" in body  # HTML entity unescaped


def test_plantuml_gantt_and_multiline_note(tmp_path):
    puml = (
        "@startgantt\n"
        "[Design Phase] lasts 5 days\n"
        "[Build Phase] lasts 10 days\n"
        "@endgantt\n"
        "@startuml\n"
        "note over Alice\n"
        "  This is a multi-line\n"
        "  note body.\n"
        "end note\n"
        "@enduml\n"
    )
    p = _write(str(tmp_path), "g.puml", puml)
    body = _extract(p)[0].text
    assert "Design Phase" in body and "Build Phase" in body
    assert "multi-line" in body and "note body." in body
    assert "@startgantt" not in body and "lasts 5 days" not in body


def test_d2_bare_container_key(tmp_path):
    # A bare container key (no label, no connection) contributes its key name.
    d2 = "gateway {\n  inner\n}\n"
    p = _write(str(tmp_path), "b.d2", d2)
    body = _extract(p)[0].text
    assert "gateway" in body


def test_tsv_uses_tab_delimiter_default(tmp_path):
    # .tsv defaults to a tab delimiter (rendered as a Markdown table).
    p = _write(str(tmp_path), "t.tsv", "Name\tQty\nWidget\t3\n")
    body = _extract(p)[0].text
    assert "| Name | Qty |" in body
    assert "| Widget | 3 |" in body


def test_split_oversized_part_ids_and_provenance():
    # Oversized chunks get stable #part=N ids, part/part_count provenance and a
    # per-part content_hash.
    parts = list(ex._split_oversized(_mk(_BIG), 200, 40))
    assert len(parts) > 1
    for i, part in enumerate(parts):
        assert part.id == f"/x/f.docx#sec=0#part={i}"
        assert part.metadata["provenance"]["part"] == i
        assert part.metadata["provenance"]["part_count"] == len(parts)
        assert part.metadata["content_hash"]
    # Distinct parts hash distinctly.
    hashes = {p.metadata["content_hash"] for p in parts}
    assert len(hashes) == len(parts)


# ---------------------------------------------------------------------------
# Ingest module (provenance + custom store address). No embedding required.
# ---------------------------------------------------------------------------
def test_provenance_chunk_records_url_and_tags():
    from doxtr_rag.ingest import _provenance_chunk

    c = _provenance_chunk(
        source_id="https://example.com/x",
        scope=AccessControl.LOCAL_PRIVATE,
        source_url="https://example.com/x",
        tags="type:standard, protocol:oauth2",
        title="Example",
    )
    assert c is not None
    assert "Source URL: https://example.com/x" in c.text
    assert "Tags: type:standard, protocol:oauth2" in c.text
    assert c.metadata["source_file"] == "https://example.com/x"


def test_provenance_chunk_none_without_url_or_tags():
    from doxtr_rag.ingest import _provenance_chunk

    assert _provenance_chunk(
        source_id="x", scope=AccessControl.LOCAL_PRIVATE,
        source_url=None, tags=None, title=None,
    ) is None


def test_url_extension_detection():
    from doxtr_rag.ingest import _url_extension

    assert _url_extension("https://x/rfc8693.pdf") == ".pdf"
    assert _url_extension("https://x/page.md") == ".md"
    assert _url_extension("https://x/some/page") == ".html"  # default


def test_custom_store_addressing():
    from doxtr_rag.config import SphinxRAGConfig
    from doxtr_rag.ingest import _custom_store

    cfg = SphinxRAGConfig.from_mapping(
        {"rag_local_store_mode": "embedded", "rag_local_store_path": "/tmp/doxtr_ct"}
    )
    vs = _custom_store(cfg, "my_store")
    # The custom store is addressed by its own database name, scope-independent
    # collection.
    assert vs._address.database == "my_store"
    assert vs._address.collection.endswith("_1024")


if __name__ == "__main__":
    import tempfile

    passed = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        if fn.__code__.co_argcount == 1:
            with tempfile.TemporaryDirectory() as d:
                fn(d)
        else:
            fn()
        print("PASS", name)
        passed += 1
    print(f"all {passed} tests passed")
