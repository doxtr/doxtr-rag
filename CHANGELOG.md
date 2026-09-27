# Changelog

All notable changes to `doxtr-rag` are documented in this file. The format is
loosely based on [Keep a Changelog](https://keepachangelog.com/), and the
project aims to follow semantic versioning once it reaches `1.0.0`.

## [Unreleased]

## [0.1.2]

### Added
- **Ad-hoc `doxtr-rag ingest` command.** Ingest a single local file (`--file`)
  or a fetched URL (`--url`) into a chosen store: the canonical shared/private
  stores via `--ingest-scope`, or an independent named database via `--store`
  (created on demand, co-located with the private embedded store, and kept
  outside the shared/private isolation model). URL ingests fetch and extract
  with the extension-appropriate extractor (defaulting to HTML) and prepend a
  small **provenance chunk** carrying the source URL and tags; `--source-url`,
  `--tags`, and `--doc-title` attach the same provenance to a file ingest.
- **Portable per-mount sidecar index (`doxtr-rag mount-index` / `mount-watch`).**
  A per-mount sidecar under `<mount>/.doxtr/` (a SQLite hash cache plus an
  embedded ChromaDB) lets a checked-out mount be queried immediately without a
  prior `make html`, and lets unchanged files be skipped on re-scan. `mount-index`
  builds/refreshes the sidecar once; `mount-watch` runs a single-consumer
  background loop that incrementally ingests/prunes on create/modify/delete
  (event-driven via `watchdog` when installed, else polling).

### Packaging
- **Optional `watch` extra + `all` aggregate.** `watchdog` is now installable
  via `pip install doxtr-rag[watch]` (event-driven `mount-watch`; absent it
  falls back to polling). A convenience `all` extra (`pip install
  doxtr-rag[all]`) aggregates the optional runtime features via a recursive
  self-reference so it stays in sync as new extras are added. Neither is pulled
  in by a plain `pip install doxtr-rag`.
- **Strict mypy under CI without `watchdog`.** `watchdog.*` is skipped in the
  mypy overrides and the `FileSystemEventHandler` adapter carries a
  `type: ignore[misc]`, so `mypy --strict` passes whether or not the optional
  `watchdog` dependency is installed.

## [0.1.1] - 2026-09-21

### Added
- **`.docx` raw-XML `<w:t>` fallback.** `extract_docx` now falls back to reading
  the raw `<w:t>` text nodes from `word/document.xml` when the standard
  python-docx paragraph API surfaces no body content — e.g. when the document
  stores text in non-standard runs, content-controls, table cells, or custom
  paragraph styles (like `InfoLine`) that python-docx does not expose as
  paragraphs. The fallback reuses the same heading-chunk pipeline (promoting
  known section labels to headings) so section structure is preserved and no
  content is silently dropped.

### Changed
- **Parallel-write safe under `sphinx-build -j`.** The extension now declares
  `parallel_write_safe = True`. Every event it hooks runs in the main process
  (`builder-inited`/`build-finished` are main-process events and
  `doctree-resolved` is emitted from the write phase's RESOLVING step before any
  write worker forks; `write_doc` is never hooked), so `build-finished` is the
  sole writer to the SQLite cache and Chroma store. Parallel builds no longer
  downgrade to serial writes (removing the `not safe for parallel writing` /
  `doing serial write` warnings and the associated slowdown).
- **No-op `embeddingFunction` on collection queries (no `@chroma-core/default-embed`).**
  The TypeScript query extension now passes a no-op `embeddingFunction` to
  `getCollection`, preventing `chromadb` 3.x from instantiating its
  `DefaultEmbeddingFunction`. Query embeddings are always pre-computed by the
  `bge-m3` HTTP server and passed as `queryEmbeddings`, so the collection's
  built-in embedder is never used and the optional `@chroma-core/default-embed`
  package is no longer required.
- **TypeScript extension is now shipped inside the package** at
  `doxtr_rag/pi_extension/` (single source of truth, included in the wheel/sdist as package
  data) instead of living under `.pi/extensions/chroma-rag/`. `.pi/extensions/chroma-rag/` is
  now purely a *deployment target* the installer populates, and is gitignored. This cleanly
  separates product source from a developer's local Pi install.
- **`doxtr-rag --install-agent pi` now works from a plain `pip install`** (no source checkout):
  it copies the packaged extension into `$PI_AGENT_DIR/extensions/chroma-rag`. Copy is the
  default; `--symlink` links to a writable dev checkout for live edits (previously symlink was
  the default and resolved a path that only existed in the dev tree). Verified end to end from
  an isolated wheel install.
- Makefile/CI/cross-language test now reference the packaged `doxtr_rag/pi_extension/` source.

## [0.1.0] - 2026-09-19

First release. A multi-tenant, fail-closed Sphinx RAG extension: it builds a
dual-store (`SHARED_ORG` / `LOCAL_PRIVATE`) ChromaDB knowledge base from Sphinx
docs, mounted multi-format files (PDF/Excel/Word/PPTX/ODF), and
security-classified Jira/Confluence xlinks, and serves it to a Pi Agent
TypeScript extension (`search_knowledge_base`) that queries both stores
concurrently with RRF merging, credential scrubbing, and offline fallback.
Embeddings use a pinned `bge-m3` (1024-dim) model with enforced Python↔TypeScript
parity. Validated end to end against the `github.com/doxtr/dox` reference project
(full `sphinx-build`, real LaTeX PDF ingest, and a real cross-language query
round-trip). `make qa`: pytest ≥90% coverage, `mypy --strict`, `tsc --strict`,
vitest, cross-language + dox build/PDF regressions, extension-verifier.

### Added
- **`doxtr-rag fetch-weights` CLI** — a Python-only command (no shell/curl) that downloads the
  pinned `BAAI/bge-m3` ONNX weights (commit `5617a9f…`) via `huggingface_hub` into
  `DOXTR_RAG_BGE_M3_ONNX_DIR` (default `/opt/models/bge-m3`), resumable and verified. Idempotent
  no-op when already present. The weights-missing error now names this command.
- **Auto-started embed server (Option E)** — a build with the extension enabled auto-starts a
  loopback `bge-m3` embed server serving the SAME provider as ingestion and publishes its URL
  in `.doxtr/endpoints.json` (`embedding.embed_url`); the query extension reads it, so no manual
  `DOXTR_RAG_EMBED_URL` is needed for the common embedded deployment (still overridable). Added
  `IngestionEngine.provider` and a `Pipeline(serve_embed=...)` toggle.
- **Dual-store query proof.** New `test_cross_language_queries_both_shared_and_private` ingests
  distinctive content into BOTH the shared and private stores (two separate loopback servers in
  the dual-embedded case) and runs a single `search_knowledge_base` query for private-only
  content — asserting a `LOCAL_PRIVATE` hit is retrieved and fused, proving the TS side queries
  the private DB, not just the shared one. Added a matching RRF unit test that a chunk unique to
  one input list (shared-only or private-only) survives the fusion.
- **Real-world PDF ingest regression test.** `tests/test_dox_pdf_ingest.py` builds a genuine
  PDF from the `dox` project (Sphinx LaTeX → lualatex via the doxtr PDF theme, ~157 pages) and
  ingests it through the P3 PDF extractor + P5 storage with real bge-m3 embeddings — a
  real-world PDF fixture, not a hand-crafted one. Asserts page-level extraction (≥80 pages
  with text), `{page}` provenance, scope preservation, and a real semantic query round-trip.
  Wired into `make dox-pdf` and the `test-harness` CI. Verified: 139 page-chunks extracted +
  ingested. `build_dox_pdf` added to the shared build helper.
- **dox reference full-build regression test.** `tests/test_dox_reference_build.py` runs a real
  `sphinx-build` of the entire `github.com/doxtr/dox` reference project with `doxtr_rag`
  enabled (all its extensions: sphinx-needs, drawio, roadmap, xlink, bibtex, d2, …) and asserts
  the store is populated through the actual builder-inited → doctree-resolved → build-finished
  lifecycle against a committed baseline (`tests/baselines/dox_reference.json`: build exit 0,
  ≥35 docs / ≥120 chunks, SHARED_ORG routing, bge-m3/1024 parity). This flags when a new
  documentation construct or extension breaks doxtr-rag harvesting/ingestion. Wired into
  `make dox-build` and the `test-harness` CI (reactor container). Skips gracefully when the
  dox project or bge-m3 weights are unavailable. Verified: 40 docs → 150 chunks → 150 real
  bge-m3 vectors ingested, build succeeded.
- **KB-first tool guidance.** `search_knowledge_base` now ships `promptSnippet` +
  `promptGuidelines` instructing the agent to consult the knowledge base before answering
  factual questions on **any** subject (domain-agnostic), with the rationale that the KB may
  hold authoritative or more recent information than the model's training data. Guidance also
  covers citing sources, clean fallback when results are empty/unrelated, and treating the KB
  as only as fresh as its last build (reconcile, don't blindly trust). Strong default, not a
  hard gate (tool use is agent-invoked).
- **Real bge-m3 embeddings.** `EmbeddingProvider` now runs the pinned
  `BAAI/bge-m3` ONNX weights directly via `onnxruntime` + `tokenizers` (bge-m3 is not a
  fastembed built-in), returning the model's L2-normalized 1024-dim dense (`sentence_embedding`
  / CLS) vector. Added `doxtr_rag.embed_server` (`doxtr-rag-embed-server`), a loopback HTTP
  server exposing the same weights for query-time (P6) parity. The cross-language integration
  test gained a real-weights variant: Python ingests real `dox` docs and the TS
  `search_knowledge_base` embeds via the embed server — genuine same-model round-trip.
- **Dependency change.** Replaced `fastembed` with direct `tokenizers` + `numpy` runtime deps
  (both were already transitive via chromadb; now declared directly).

### Changed
- Reactor image guidance: bake the pinned bge-m3 ONNX weights (commit
  `5617a9f61b028005a4858fdac845db406aefb181`) into `/opt/models/bge-m3` via `aria2c` in the
  builder stage; set `DOXTR_RAG_BGE_M3_ONNX_DIR`.

### Added (earlier)
- **Agent installer.** `doxtr-rag --install-agent pi` installs the
  `chroma-rag` query extension into a Pi agent with `doxtr/reactor`-tailored defaults:
  symlink/copy into `$PI_AGENT_DIR/extensions/chroma-rag`, `npm install`, additive
  `settings.json` merge (never overwriting provider/model/auth), and a sourceable
  `doxtr-rag.env` (`DOXTR_RAG_BGE_M3_ONNX_DIR`, `DOXTR_RAG_EMBED_URL`, `DOXTR_RAG_ENDPOINTS`).
  Other agents (opencode/claude/cursor/codex) are reported as not-yet-supported with the
  MCP-bridge path forward, rather than faked.
- **P7 — Quality gate, CI wiring & cross-language integration.** `Makefile` (`make qa`:
  mypy strict, tsc strict, vitest, pytest ≥90% coverage, cross-language round-trip,
  extension-verifier regression); reconciled `.github/workflows/test.yml` (removed the stale
  `doxtr-pdf-theme-core` install; added mypy/tsc/vitest steps; `--cov-fail-under=90`);
  `tests/test_cross_language.py` + `.pi/extensions/xlang-query.ts` (Python ingest → loopback
  → TS `search_knowledge_base` round-trip proving C4.1 addressing + C3 scrub + embedding
  dimension/model parity across the language boundary); `rag-verifier` skill distinct from
  the host-build `extension-verifier` regression guard.
- **P6 — Pi Agent TypeScript RAG extension.** `.pi/extensions/chroma-rag.ts` (dual-store
  concurrent query via `Promise.allSettled` with per-query timeouts, RRF merge + dedup,
  `search_knowledge_base` tool, offline local-only fallback, query-time dimension AND model
  parity assertions, reads resolved endpoints from `.doxtr/endpoints.json` /
  `DOXTR_RAG_ENDPOINTS`); `scrub.ts` (verbatim C3 mirror), `addressing.ts` (C4.1/C2 mirror),
  `rrf.ts`. Uses the `chromadb` npm package (not `@chromadb/api`). `tsc --strict` clean;
  vitest suite green.
- **P5 — Scoped ChromaDB store & ingestion engine.** `doxtr_rag/storage.py` (dual-backend
  `ChromaVectorStore`, `IngestionEngine` with final scope reconciliation via
  `most_restrictive`, defense-in-depth isolation guard at the resolved database boundary,
  runtime writability skip, configurable batches, `resolve_endpoints`); `pipeline.py` (the
  `Pipeline` lifecycle seam — cache-aware upsert/prune keyed on resource ids, loopback
  serving for embedded stores routed through HTTP, `.doxtr/endpoints.json`, xlink scope
  folding + audit); `server.py` (loopback Chroma server); `embedding_provider.py` (bge-m3
  parity); `cli.py` (`--rebuild`, `cache-info`, `audit-info`).
- **P2 — Incremental cache & harvester.** `cache.py` (SQLite 3-domain hash cache, C8),
  `harvester.py` (structural chunking, breadcrumbs, `:ref:`/`:doc:`/`:term:` resolution,
  `harvest_document` cache seam), `pathscope.py` (deny-before-allow path routing with realpath
  confinement + symlink fail-closed).
- **P3 — Multi-format extractors.** `extractors.py` (PDF/xlsx/ods/docx/odt/pptx + keynote
  best-effort; registry pattern; graceful corrupt/password handling; mount scan with
  symlink-escape confinement).
- **P4 — XLink security classifier & audit.** `security.py` (Jira ILS + Confluence restriction
  endpoint + unified `classify_xlink` dispatcher + `NetworkBudget`, allow-list fail-closed,
  resilience policy), `audit.py` (`0600` JSONL audit log + `[INFO]` per redirect).
- **F1 — Domain schemas, configuration & storage contracts (CONTRACTS.md FROZEN).**
  `doxtr_rag/schema.py` (`AccessControl` + `most_restrictive`/`coerce`, `DocumentChunk`
  with credential-key guard, value-level scrub, `validate_assignment`, and the canonical
  `to_store_metadata`/`from_store_metadata` flatten pair, `SecurityAuditEvent`);
  `doxtr_rag/scrub.py` (single value-level credential scrubber); `doxtr_rag/base_store.py`
  (`BaseVectorStore` ABC + `(tenant, database, collection)` addressing, `address_for`,
  `address_from_config`); `doxtr_rag/config.py` (`SphinxRAGConfig` parser, `StoreTarget`
  with a genuine runtime writability probe and env-sourced/retained headers +
  `scrubbed_headers`, `model_revision`); `doxtr_rag/embedding.py` (pinned `BAAI/bge-m3`
  ONNX parity strategy, `require_weights`, `assert_dimension`). `mypy --strict` clean.
- **F0 — Package skeleton & Sphinx registration.** Reconciled the stale
  PlantUML/Gantt substrate into a RAG package: `doxtr_rag/__init__.py` with
  `__version__` and `setup(app)` registering the six original `conf.py` keys
  plus the ChromaDB deployment keys, wiring the
  `builder-inited`/`doctree-resolved`/`build-finished` lifecycle, and returning
  `parallel_read_safe=True`, `parallel_write_safe=False`. Pinned runtime
  dependencies (`pydantic>=2`, `chromadb`, `fastembed`/`onnxruntime`, `pypdf`,
  `openpyxl`, `python-docx`, `python-pptx`, `odfpy`, `httpx`,
  `sphinxcontrib-xlink`). Coverage gate raised to `fail_under = 90`; coverage
  `omit` corrected to `*/conf.py`. TypeScript extension scaffold under
  `.pi/extensions/` (`chromadb` + Pi SDK + TypeBox, `tsc --strict`).
  `.doxtr/` excluded from git and ingestion.
