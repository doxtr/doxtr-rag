# doxtr-rag

A multi-tenant, **fail-closed** Sphinx RAG (retrieval-augmented generation)
extension for the `doxtr` ecosystem. It builds a **dual-store** knowledge base
from your Sphinx documentation, mounted multi-format files, and
security-classified Jira/Confluence cross-links (`xlink`s), then serves it to an
Pi Agent TypeScript extension at agent time.

## 1. What the project is

`doxtr-rag` is a Sphinx extension plus a companion Pi Agent TypeScript extension
that together implement a retrieval knowledge base with a hard security
boundary between two physically-isolated ChromaDB stores:

- **`SHARED_ORG`** — organization-wide, safe-to-share knowledge (the Sphinx
  `source/` tree and mounts explicitly declared shared).
- **`LOCAL_PRIVATE`** — everything else, by default. Any document, node, mount,
  or `xlink` with an ambiguous, unverifiable, or failed permission check routes
  here. There is no code path where an unresolved scope becomes `SHARED_ORG`.

During `sphinx-build` the extension:

1. **Harvests** the resolved Sphinx AST (`doctree-resolved`) into structural,
   section-boundary chunks with breadcrumbs and resolved `:ref:`/`:doc:`/`:term:`
   cross-references.
2. **Extracts** mounted multi-format files — PDF (page-level), spreadsheets
   (`.xlsx`/`.ods` → Markdown tables), presentations (`.pptx` slide title + body
   + speaker notes; `.keynote`/`.key` best-effort), and Word/OpenOffice
   (`.docx`/`.odt` heading-aware).
3. **Classifies** Jira/Confluence `xlink` targets by API-level access
   restriction (allow-list: only affirmatively-unrestricted → `SHARED_ORG`).
4. **Reconciles** every chunk's final scope through a single
   `AccessControl.most_restrictive(...)` resolver and **ingests** it into the
   correct ChromaDB store, addressed by a canonical `(tenant, database,
   collection)` triple, with cache-aware incremental upserts and deletion
   pruning.

At agent time, the Pi Agent TypeScript extension (shipped in the package at
`doxtr_rag/pi_extension/`, installed to `~/.pi/agent/extensions/chroma-rag`)
registers a
`search_knowledge_base` tool that queries **both** stores concurrently, merges
results with **Reciprocal Rank Fusion (RRF)**, scrubs credentials from returned
context, and **degrades gracefully** to local-only results when the shared store
is unreachable.

All embeddings use a pinned `bge-m3` model standardized on **1024-dim** dense
vectors; parity (same model + dimension) is enforced across ingestion (Python)
and query (TypeScript).

## 2. Installation

Requirements:

- **Python ≥ 3.10**
- The **`doxtr/reactor`** devcontainer (`docker.io/doxtr/reactor:0.1.5`) — run
  all Python/pytest/mypy and Node/tsc/vitest commands inside it.
- Node ≥ 20 (bundled in the devcontainer) for the TypeScript extension.
- The pinned **`bge-m3` (1024-dim) ONNX weights** (~2.3 GB). In the
  `doxtr/reactor` image these are **pre-baked** at `/opt/models/bge-m3`
  (`DOXTR_RAG_BGE_M3_ONNX_DIR` is set for you). Elsewhere, download them once
  with the built-in, Python-only command (no shell/curl needed):

  ```bash
  doxtr-rag fetch-weights          # -> $DOXTR_RAG_BGE_M3_ONNX_DIR or /opt/models/bge-m3
  ```

  If the weights are absent at build/query time, doxtr-rag **fails with a clear
  error naming `doxtr-rag fetch-weights`** rather than silently substituting a
  different model (that would break embedding parity).

Install the Python package (editable, with dev extras):

```bash
pip install -e .[dev]
```

The recommended way to install the query extension is the CLI (it copies the
packaged extension into your Pi agent dir and wires the reactor defaults):

```bash
doxtr-rag --install-agent pi
```

To work on the TypeScript extension source directly (in a checkout):

```bash
cd doxtr_rag/pi_extension && npm install
```

Enable the extension in your `conf.py`:

```python
extensions = [
    # ...
    "doxtr_rag",
]
```

## 3. Configuration reference

All keys are registered by `doxtr_rag.setup(app)` and parsed by
`doxtr_rag.config.SphinxRAGConfig`. Set them in your Sphinx `conf.py`.

### The six original keys

| key | type | default | effect |
|-----|------|---------|--------|
| `rag_shared_store_uri` | `str \| None` | `None` | Shared store location. In `http` mode a scheme-qualified URI (`http(s)://host:port`); in `embedded` mode an on-disk path. Secrets are **never** embedded here (env-sourced). |
| `rag_local_store_path` | `str` | `.doxtr/chroma` | Embedded on-disk path for the `LOCAL_PRIVATE` store. |
| `rag_external_mounts` | `list[dict]` | `[]` | Multi-format file mounts to scan. Each entry is `{"path": "...", "scope": "SHARED_ORG"?}`. Content defaults to `LOCAL_PRIVATE` unless the mount **affirmatively** declares `"scope": "SHARED_ORG"`. |
| `rag_cache_db` | `str` | `.doxtr/cache.db` | SQLite incremental-build cache (three SHA-256 hash domains: nodes, external files, xlink payloads). |
| `rag_audit_log_path` | `str` | `.doxtr/security_audit.jsonl` | Append-only, `0600` JSON Lines security audit log. |
| `rag_embedding_provider` | `str` | `bge-m3` | Embedding provider/revision token. |

### ChromaDB deployment keys

| key | type | default | effect |
|-----|------|---------|--------|
| `rag_shared_store_mode` | `"embedded" \| "http"` | `http` | How the `SHARED_ORG` store is reached. Independent of the local store. |
| `rag_local_store_mode` | `"embedded" \| "http"` | `embedded` | How the `LOCAL_PRIVATE` store is reached. |
| `rag_shared_store_database` | `str` | `doxtr_shared` | ChromaDB database name for the shared scope. |
| `rag_local_store_database` | `str` | `doxtr_private` | ChromaDB database name for the private scope. |
| `rag_store_tenant` | `str` | `default_tenant` | ChromaDB tenant shared by both scopes. |
| `rag_local_server_uri` | `str \| None` | `None` | URI for the `LOCAL_PRIVATE` store when `rag_local_store_mode = "http"`. |

### Derived (exposed for parity enforcement)

| key | value | effect |
|-----|-------|--------|
| `embedding_model` | `BAAI/bge-m3` | Pinned dense model. |
| `embedding_dimension` | `1024` | Fatal, build-stopping error on any mismatch. |

### Per-mount `SHARED_ORG` declaration & fail-closed defaults

A mount is shared **only** when it declares it explicitly:

```python
rag_external_mounts = [
    {"path": "/data/handbook"},                          # -> LOCAL_PRIVATE (default)
    {"path": "/data/public-specs", "scope": "SHARED_ORG"},  # -> SHARED_ORG (affirmative)
]
```

Path routing is deny-list-before-allow-list: any resolved path containing
`noter/` or `NDA/` always routes `LOCAL_PRIVATE` regardless of AST hints, only
`source/` (and explicitly-shared mounts) map to `SHARED_ORG`, and every path is
resolved with `realpath` + confinement so `source/../NDA/x` traversal and
symlink laundering fail closed.

### HTTP store auth (env-sourced, never in `conf.py`)

Connection secrets for `http` stores come from the environment as `k=v;k=v`
header pairs, never from the URI or `conf.py`:

```bash
export DOXTR_RAG_SHARED_STORE_HEADERS="Authorization=Bearer $SHARED_TOKEN"
export DOXTR_RAG_LOCAL_STORE_HEADERS="Authorization=Bearer $LOCAL_TOKEN"
```

The resolved URI/headers are scrubbed before any log line.

## 4. ChromaDB deployment guide

Each store can run **embedded** (on-disk) or against an **external HTTP server**,
independently.

### Embedded (on-disk)

```python
rag_local_store_mode = "embedded"
rag_local_store_path = ".doxtr/chroma"
rag_shared_store_mode = "embedded"
rag_shared_store_uri = "/srv/doxtr/chroma_shared"   # embedded path in embedded mode
```

Because the Pi Agent (JS) client cannot read an embedded on-disk store directly,
the build starts a **loopback-bound Chroma HTTP server** over each embedded
store's path; both ingestion and query then go through that HTTP endpoint (one
process — the server — owns the on-disk store). The resolved per-scope endpoints
are written to `.doxtr/endpoints.json` for the query extension to read.

### External HTTP server (worked example)

Run a persistent `chromadb/chroma` server in Docker:

```bash
docker run -d --name doxtr-chroma \
  -p 8000:8000 \
  -v /srv/doxtr/chroma-data:/data \
  -e CHROMA_SERVER_AUTHN_CREDENTIALS="$CHROMA_TOKEN" \
  -e CHROMA_SERVER_AUTHN_PROVIDER="chromadb.auth.token_authn.TokenAuthenticationServerProvider" \
  chromadb/chroma:latest
```

Or via `docker-compose.yml`:

```yaml
services:
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - /srv/doxtr/chroma-data:/data      # persistent volume
    environment:
      CHROMA_SERVER_AUTHN_CREDENTIALS: ${CHROMA_TOKEN}
      CHROMA_SERVER_AUTHN_PROVIDER: chromadb.auth.token_authn.TokenAuthenticationServerProvider
    restart: unless-stopped
```

> **TLS:** for any non-loopback remote, terminate TLS (a reverse proxy or
> `--ssl`) and use an `https://` URI so `StoreTarget.ssl` is `true`.

Point the stores at it in `conf.py`:

```python
rag_shared_store_mode = "http"
rag_shared_store_uri = "https://chroma.internal.example.com:8000"
rag_local_store_mode = "http"
rag_local_server_uri = "https://chroma.internal.example.com:8000"
```

with the auth token supplied via env (see §3).

### Single server hosting both scopes (separate databases, one tenant)

A single external server MAY host both scopes as separate **databases** under
one **tenant** — physical isolation is enforced at the resolved
database/collection boundary, not merely by process:

```python
rag_store_tenant = "default_tenant"
rag_shared_store_mode = "http"
rag_shared_store_uri = "https://chroma.internal.example.com:8000"
rag_shared_store_database = "doxtr_shared"
rag_local_store_mode = "http"
rag_local_server_uri = "https://chroma.internal.example.com:8000"
rag_local_store_database = "doxtr_private"
```

The shared-store adapter raises on any non-`SHARED_ORG` chunk addressed to the
shared database, so `LOCAL_PRIVATE` vectors can never reach the shared scope even
when both live on one host.

## 5. Shared-store writability

Whether the shared store is ingested into is decided by a **runtime writability
probe** of the actual mount/endpoint — **not** by the store mode:

- **`rw` mount / write-accepting server → locally writable:** the build ingests
  `source/` (and shared mounts) into the shared store in place.
- **`ro` mount / read-only server → query-only:** shared ingestion is **skipped
  gracefully** with an `[INFO]` message and the store stays fully queryable.
  Never a hard error.

The `LOCAL_PRIVATE` store (`.doxtr/chroma`) is always read-write.

### `.devcontainer` mount options (requirement #22)

Mount the shared Chroma data directory `rw` for local shared ingestion or `ro`
for query-only nodes:

```jsonc
// .devcontainer/devcontainer.json
"mounts": [
  // read-only (query-only node): shared ingestion is skipped, still queryable
  "source=/srv/doxtr/chroma_shared,target=/workspaces/docs/.doxtr/chroma_shared,type=bind,readonly"
  // ...or omit ",readonly" for a read-write (ingesting) node.
]
```

For an external `chromadb/chroma` server, apply the same `rw`/`ro` policy to
that container's **own** persistent volume instead of a bind mount.

## 6. Pi Agent extension setup

The extension source ships inside the package at `doxtr_rag/pi_extension/`
(single source of truth). Install it into your Pi Agent config with the CLI —
it copies the packaged extension and wires reactor defaults:

```bash
doxtr-rag --install-agent pi
```

This populates `~/.pi/agent/extensions/chroma-rag/` (a subdirectory whose
`package.json` `pi.extensions` manifest makes Pi load **only** `chroma-rag.ts`
and ignore the helper modules `addressing.ts`, `rrf.ts`, `scrub.ts`,
`xlang-query.ts`). Works from a plain `pip install` — no source checkout needed.

The extension registers the **`search_knowledge_base`** tool:

```
search_knowledge_base({ query: string, top_k?: number }) -> Markdown context blocks
```

Behavior:

- Reads the resolved per-scope endpoints from `.doxtr/endpoints.json` (or the
  path in `DOXTR_RAG_ENDPOINTS`) — no addressing is re-derived, no loopback port
  is guessed.
- Queries both stores **concurrently** (`Promise.allSettled`), embeds the query
  with the same pinned `bge-m3`/1024 model as ingestion (dimension **and** model
  parity asserted before any query), and merges with **RRF** (`1/(k+rank)`,
  rank-based, distances ignored, identical chunks deduped).
- Returns credential-scrubbed Markdown context blocks (the TS `scrub()` mirrors
  the Python pattern set verbatim).
- **Offline fallback:** if the shared store is unreachable or times out, returns
  local-only results — no crash, no unhandled rejection, no stack traces in LLM
  context.

**KB-first default.** The tool ships with `promptSnippet` + `promptGuidelines`
that instruct the agent to consult `search_knowledge_base` **before** answering
factual questions on *any* subject — because your knowledge base may hold
authoritative, domain-specific, or more recent information than the model's
training data (your own docs, an indexed website, product specs, or whatever you
ingested). The guidance is domain-agnostic and also tells the agent to cite the
returned sources, to fall back to general knowledge when results are empty or
unrelated (never fabricating a citation), and to treat the KB as only as fresh
as its last build (reconcile rather than blindly trust). This is a strong
default, not a hard gate — tool use is agent-invoked, so a model can still skip
it; for a hard guarantee you would add a harness-level turn hook.

**Query-time embedding.** The query side must embed with the same pinned
`bge-m3`/1024 model as ingestion. By default this is **automatic**: when a build
runs with the extension enabled, the pipeline auto-starts a loopback `bge-m3`
embed server (serving the same weights as ingestion) and publishes its URL into
`.doxtr/endpoints.json` (`embedding.embed_url`). The query extension reads that
URL — no manual endpoint or `DOXTR_RAG_EMBED_URL` needed for the common
embedded deployment.

For deployments where the embed endpoint runs elsewhere, start it yourself and
point the extension at it (overrides the auto-started one):

```bash
# standalone bge-m3 embed server (uses DOXTR_RAG_BGE_M3_ONNX_DIR weights)
doxtr-rag-embed-server --host 127.0.0.1 --port 8519 &
export DOXTR_RAG_EMBED_URL="http://127.0.0.1:8519/embed"
```

### One-command installer (reactor-tailored)

Inside the `doxtr/reactor` container, install the extension into a Pi agent with
defaults tailored to the container:

```bash
doxtr-rag --install-agent pi              # copy the packaged extension (default)
doxtr-rag --install-agent pi --symlink    # or symlink to a dev checkout (live edits)
# if the console script isn't on PATH:
#   python -m doxtr_rag.cli --install-agent pi
```

The installer (`doxtr_rag.agent_install`, exercised by the test suite):

- copies (default) the packaged `doxtr_rag/pi_extension/` into `$PI_AGENT_DIR`
  (default `~/.pi/agent`) under `extensions/chroma-rag` — or `--symlink` to a
  writable dev checkout,
- runs `npm install` for the extension's Node deps at the destination,
- **adds** reactor defaults to `settings.json` without overwriting an existing
  provider/model/auth choice,
- writes a sourceable `~/.pi/agent/doxtr-rag.env` with reactor defaults
  (`DOXTR_RAG_BGE_M3_ONNX_DIR=/opt/models/bge-m3`,
  `DOXTR_RAG_EMBED_URL=http://127.0.0.1:8519/embed`,
  `DOXTR_RAG_ENDPOINTS=.doxtr/endpoints.json`), each overridable from the shell.

Then `source ~/.pi/agent/doxtr-rag.env` and start `pi`.

**Other agents.** `opencode`, `claude`, `cursor`, and `codex` do **not** consume
Pi's TypeScript `ExtensionAPI`, so the installer reports them as not-yet-supported
rather than faking a config copy. Exposing `search_knowledge_base` to them is a
planned follow-up via an **MCP server** wrapping the tool (opencode and Claude
Code speak MCP) or each agent's native plugin format.

## 7. Security model

- **Fail-closed scope routing:** ambiguous/unverifiable → `LOCAL_PRIVATE`;
  `SHARED_ORG` only from an affirmative signal. `AccessControl.most_restrictive`
  is the single final-scope decision.
- **Path confinement:** every path resolved with `realpath` and asserted within
  an allowed root; deny-list (`noter/`, `NDA/`) before allow-list (`source/`);
  escaping symlinks are skipped fail-closed.
- **Credential scrubbing:** a single value-level scrubber (URLs-with-creds,
  JWTs, `Bearer`, `ghp_`/`github_pat_`, `xox[baprs]-`, `AKIA`/`ASIA`, PEM keys)
  runs over all chunk text, metadata values, audit records, and returned LLM
  context — mirrored verbatim in TypeScript. A metadata-key deny-list adds
  defense-in-depth.
- **Physical store isolation:** the shared-store adapter raises on any
  non-`SHARED_ORG` chunk at the resolved database boundary (holds under retries,
  errors, and the one-server-two-databases topology).
- **Resilience:** 5s connect / 10s read timeouts, ≤2 retries only on `5xx`/`429`
  (never `401`/`403`), a per-build network budget, and fail-closed HTTP mapping.
- **Audit log:** restricted-`xlink` routing is recorded to
  `.doxtr/security_audit.jsonl` (`0600`, scrubbed, one `[INFO]` per redirect).

## 8. Maintenance CLI

```bash
doxtr-rag fetch-weights                             # download the pinned bge-m3 ONNX weights
doxtr-rag fetch-weights --force                     # re-download even if present
doxtr-rag --conf conf.py --rebuild --scope all      # drop & reindex collections
doxtr-rag --conf conf.py --rebuild --scope shared   # or just one scope
doxtr-rag --conf conf.py cache-info                 # inspect the SQLite cache
doxtr-rag --conf conf.py audit-info                 # summarize the audit log
doxtr-rag --install-agent pi                        # install the query extension into Pi Agent
```

Collections are tagged with the model revision + dimension; a model/dimension
change drops and rebuilds the affected collection.

## 9. Development & QA

Run the full quality gate (what CI enforces):

```bash
make qa
```

`make qa` runs, and fails on any of:

- `pytest` with **≥ 90 %** coverage (`--cov-fail-under=90`), including the
  Python↔TypeScript **cross-language integration round-trip** and the **dox
  reference full-build regression** (a real `sphinx-build` of
  `github.com/doxtr/dox` with the extension enabled, asserting the store is
  populated through the real event lifecycle against a committed baseline);
- `mypy --strict` over `doxtr_rag/`;
- `tsc --noEmit --strict` over the Pi extension;
- `vitest` for the extension;
- the **extension-verifier** regression smoke check (the host `dox` HTML +
  light/dark PDF build must still succeed within its warning baseline).

Individual targets: `make test`, `make mypy`, `make tsc`, `make tstest`,
`make xlang`, `make dox-build`, `make dox-pdf`, `make verifier`.

The **dox-build regression** (`make dox-build` /
`tests/test_dox_reference_build.py`) is the guard that tells you *if a new
documentation construct or extension breaks doxtr-rag*: it builds the whole
reference project and fails if ingestion drops below the baseline floors in
`tests/baselines/dox_reference.json`. Update that baseline only when the dox
project legitimately changes. The **dox-pdf regression** (`make dox-pdf` /
`tests/test_dox_pdf_ingest.py`) additionally compiles the reference project to a
real PDF (LaTeX → lualatex) and ingests it through the PDF extractor, exercising
the binary-document path against a real-world file. Both run in the
`doxtr/reactor` container (full extension stack + LaTeX + pre-baked bge-m3
weights) and skip gracefully elsewhere.

### Test layout

- `tests/` — fast unit tests (schema, scrub, cache, harvester, extractors,
  security, audit, storage, pipeline, config, embedding) **plus** the
  cross-language round-trip.
- `test_harness/` — a real Sphinx project (`conf.py`, `source/` with
  `source/`/`noter/`/`NDA/` mounts, `conf_overrides/`, `assertions.py`,
  `test_runner.py`) for integration/regression, mirroring the reference
  `doxtr-pdf-theme-core` harness.
- `doxtr_rag/pi_extension/tests/` — the TypeScript `vitest` suite (scrub
  parity, RRF, metadata inflation, dimension/model parity, offline fallback).
