/**
 * chroma-rag.ts — Pi Agent RAG query extension (P6, Wave C).
 *
 * Queries the dual ChromaDB stores (SHARED_ORG + LOCAL_PRIVATE) concurrently
 * over HTTP, merges with Reciprocal Rank Fusion, scrubs returned context with
 * the same rules as `scrub.py` (C3 mirror), asserts query-time embedding
 * dimension parity (1024, bge-m3), and registers the `search_knowledge_base`
 * tool. If the shared store is unreachable it degrades gracefully to local-only
 * results — never throwing, never leaking stack traces into LLM context.
 *
 * Endpoints are read from `.doxtr/endpoints.json` (written by P5) so this
 * separate process connects without re-deriving addressing or guessing the
 * loopback port. Uses the `chromadb` npm package (NOT `@chromadb/api`).
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { ChromaClient } from "chromadb";
import { Type } from "@sinclair/typebox";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  EMBEDDING_DIMENSION,
  type EndpointsDescriptor,
  type MountEndpoint,
  type ScopeEndpoint,
  fromStoreMetadata,
} from "./addressing.js";
import { reciprocalRankFusion, type FusedChunk, type RankedChunk } from "./rrf.js";
import { scrub } from "./scrub.js";

const CONNECT_TIMEOUT_MS = 5000;
const READ_TIMEOUT_MS = 10000;
const DEFAULT_TOP_K = 5;

/**
 * No-op embeddingFunction passed to every `getCollection()` call to suppress
 * chromadb 3.x from instantiating DefaultEmbeddingFunction (which requires the
 * optional `@chroma-core/default-embed` package). We never use a collection's
 * built-in embedder — queryEmbeddings are always pre-computed by the bge-m3
 * HTTP server and passed directly to `collection.query()`. Kept as a single
 * module-level constant so every call site suppresses the default embedder
 * identically (no divergence).
 */
const NOOP_EMBEDDING_FUNCTION = { generate: async (_texts: string[]) => [] as number[][] };

/** Locate and parse `.doxtr/endpoints.json`.
 *
 * Discovery order (first hit wins):
 *   1. `DOXTR_RAG_ENDPOINTS` env var (explicit path) — the robust contract when
 *      the Sphinx build's `rag_cache_db` is absolute or P6 runs from another CWD.
 *   2. `<cwd>/.doxtr/endpoints.json`
 *   3. `<cwd>/endpoints.json`
 * P5 writes the file beside `rag_cache_db`; set `DOXTR_RAG_ENDPOINTS` to that
 * path to decouple the two processes' working directories.
 */
export function loadEndpoints(cwd: string = process.cwd()): EndpointsDescriptor | null {
  const candidates = [
    process.env.DOXTR_RAG_ENDPOINTS,
    path.join(cwd, ".doxtr", "endpoints.json"),
    path.join(cwd, "endpoints.json"),
  ].filter((c): c is string => Boolean(c));
  for (const candidate of candidates) {
    try {
      if (fs.existsSync(candidate)) {
        return JSON.parse(fs.readFileSync(candidate, "utf-8")) as EndpointsDescriptor;
      }
    } catch {
      // fall through to next candidate
    }
  }
  return null;
}

/** Parse `k=v;k=v` auth headers from a per-scope env var (mirror of config._parse_headers_env). */
function headersFromEnv(varName: string): Record<string, string> {
  const raw = process.env[varName] ?? "";
  const headers: Record<string, string> = {};
  for (const pair of raw.split(";")) {
    const trimmed = pair.trim();
    const eq = trimmed.indexOf("=");
    if (eq <= 0) continue;
    headers[trimmed.slice(0, eq).trim()] = trimmed.slice(eq + 1).trim();
  }
  return headers;
}

const SCOPE_HEADER_ENV: Record<string, string> = {
  shared: "DOXTR_RAG_SHARED_STORE_HEADERS",
  private: "DOXTR_RAG_LOCAL_STORE_HEADERS",
};

/** Build a ChromaClient for one scope endpoint. */
function clientFor(scopeKey: "shared" | "private", ep: ScopeEndpoint): ChromaClient {
  return new ChromaClient({
    host: ep.host,
    port: ep.port,
    ssl: ep.ssl,
    tenant: ep.tenant,
    database: ep.database,
    headers: headersFromEnv(SCOPE_HEADER_ENV[scopeKey]),
  });
}

/** Build a ChromaClient for one mount sidecar endpoint (no per-scope headers). */
function clientForMount(ep: MountEndpoint): ChromaClient {
  return new ChromaClient({
    host: ep.host,
    port: ep.port,
    ssl: ep.ssl,
    tenant: ep.tenant,
    database: ep.database,
  });
}

/** Query-time embedder interface (server-side bge-m3 or a matching client embedder). */
export interface QueryEmbedder {
  /** The pinned model id used at query time (must match ingestion parity). */
  readonly model: string;
  embed(text: string): Promise<number[]>;
}

/** Assert the query embedding dimension matches the pinned parity (fatal). */
export function assertDimension(dim: number): void {
  if (dim !== EMBEDDING_DIMENSION) {
    throw new Error(
      `query embedding dimension parity violation: expected ${EMBEDDING_DIMENSION} (bge-m3), got ${dim}`,
    );
  }
}

/** Assert the query-time model matches the ingestion model (fatal parity error). */
export function assertModelParity(queryModel: string, ingestModel: string): void {
  // Compare on the pinned revision token, tolerating an org/ prefix
  // (e.g. "BAAI/bge-m3" vs "bge-m3").
  const norm = (m: string) => m.split("/").pop() ?? m;
  if (norm(queryModel) !== norm(ingestModel)) {
    throw new Error(
      `query-time embedding model parity violation: query uses ${queryModel}, ` +
        `ingestion used ${ingestModel}`,
    );
  }
}

/** Race a promise against a timeout; rejects with a timeout error if exceeded. */
function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    promise.then(
      (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        clearTimeout(timer);
        reject(e);
      },
    );
  });
}

/** Query one scope's collection; returns ranked chunks or throws (caught by caller). */
async function queryScope(
  client: ChromaClient,
  collectionName: string,
  embedding: number[],
  topK: number,
): Promise<RankedChunk[]> {
  // See NOOP_EMBEDDING_FUNCTION: suppress chromadb 3.x DefaultEmbeddingFunction.
  const collection = await client.getCollection({
    name: collectionName,
    embeddingFunction: NOOP_EMBEDDING_FUNCTION,
  });
  const res = await collection.query({
    queryEmbeddings: [embedding],
    nResults: topK,
  });
  const ids = (res.ids?.[0] ?? []) as string[];
  const docs = (res.documents?.[0] ?? []) as (string | null)[];
  const metas = (res.metadatas?.[0] ?? []) as (Record<string, unknown> | null)[];
  return ids.map((id, i) => ({
    id,
    text: docs[i] ?? "",
    metadata: fromStoreMetadata(metas[i] ?? {}),
  }));
}

/** Render fused chunks into scrubbed Markdown context blocks for the LLM. */
export function renderMarkdown(chunks: FusedChunk[]): string {
  if (chunks.length === 0) return "No results found in the knowledge base.";
  return chunks
    .map((c, i) => {
      const crumb = c.metadata.breadcrumb.join(" › ");
      const scope = c.metadata.access_scope;
      const header = `### ${i + 1}. ${scrub(c.metadata.section_title || "(untitled)")}`;
      const meta = `*${scrub(crumb)}* — \`${scope}\` — source: \`${scrub(c.metadata.source_file)}\``;
      return `${header}\n${meta}\n\n${scrub(c.text)}`;
    })
    .join("\n\n---\n\n");
}

/**
 * Search both stores concurrently, RRF-merge, and return scrubbed Markdown.
 * Gracefully degrades to local-only when the shared store is unreachable.
 * Never throws for connectivity issues (only a query-time dimension mismatch,
 * asserted before any network call, is fatal per embedding parity).
 */
export async function searchKnowledgeBase(
  query: string,
  topK: number,
  embedder: QueryEmbedder,
  endpoints: EndpointsDescriptor,
): Promise<string> {
  const embedding = await embedder.embed(query);
  assertDimension(embedding.length);
  // Enforce query-time MODEL parity too (CONVENTIONS §5): a drifted model that
  // still emits 1024 dims must be a fatal error, not a silent mismatch.
  assertModelParity(embedder.model, endpoints.embedding.model);

  const sharedClient = clientFor("shared", endpoints.shared);
  const privateClient = clientFor("private", endpoints.private);

  // Each configured mount sidecar (e.g. a vault/) is queried alongside the two
  // canonical stores and folded into the same RRF merge. A newly-configured
  // vault becomes searchable automatically once its sidecar is built + served
  // (published under endpoints.mounts) — no code change needed here.
  const mounts = endpoints.mounts ?? [];

  const queries: Array<{ label: string; promise: Promise<RankedChunk[]> }> = [
    {
      label: "shared",
      promise: withTimeout(
        queryScope(sharedClient, endpoints.shared.collection, embedding, topK),
        CONNECT_TIMEOUT_MS + READ_TIMEOUT_MS,
        "shared store query",
      ),
    },
    {
      label: "private",
      promise: withTimeout(
        queryScope(privateClient, endpoints.private.collection, embedding, topK),
        CONNECT_TIMEOUT_MS + READ_TIMEOUT_MS,
        "private store query",
      ),
    },
  ];
  for (const mount of mounts) {
    queries.push({
      label: `mount:${mount.path}`,
      promise: withTimeout(
        queryScope(clientForMount(mount), mount.collection, embedding, topK),
        CONNECT_TIMEOUT_MS + READ_TIMEOUT_MS,
        `mount store query (${mount.path})`,
      ),
    });
  }

  const settled = await Promise.allSettled(queries.map((q) => q.promise));

  const lists: RankedChunk[][] = [];
  settled.forEach((outcome, idx) => {
    if (outcome.status === "fulfilled") {
      lists.push(outcome.value);
    } else {
      // Degrade gracefully: log a scrubbed one-liner, never a stack trace.
      const scopeName = queries[idx]?.label ?? `query#${idx}`;
      const reason = scrub(String((outcome as PromiseRejectedResult).reason?.message ?? ""));
      console.error(`[chroma-rag] ${scopeName} store unreachable; degrading: ${reason}`);
    }
  });

  const fused = reciprocalRankFusion(lists).slice(0, topK);
  return renderMarkdown(fused);
}

const paramsSchema = Type.Object({
  query: Type.String({ description: "The natural-language query." }),
  top_k: Type.Optional(
    Type.Number({ description: "Max results to return (default 5).", minimum: 1 }),
  ),
});

/**
 * Query-time embedder using a bge-m3 HTTP endpoint. The URL is taken from the
 * pipeline-published `endpoints.embedding.embed_url` (Option E: the build
 * auto-starts the embed server) when present, else the `DOXTR_RAG_EMBED_URL`
 * env var. The parity contract (same pinned bge-m3/1024 as ingestion) is
 * enforced by assertDimension + assertModelParity.
 */
export function defaultEmbedder(
  embedUrl?: string | null,
  model: string = "bge-m3",
): QueryEmbedder {
  const url = embedUrl || process.env.DOXTR_RAG_EMBED_URL;
  return {
    model,
    async embed(text: string): Promise<number[]> {
      if (!url) {
        throw new Error(
          "no query-time bge-m3 embed endpoint: neither endpoints.embedding.embed_url " +
            "nor DOXTR_RAG_EMBED_URL is set (run a build with the extension, or set " +
            "DOXTR_RAG_EMBED_URL to a bge-m3 server).",
        );
      }
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), CONNECT_TIMEOUT_MS + READ_TIMEOUT_MS);
      try {
        const resp = await fetch(url, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ input: text, model }),
          signal: controller.signal,
        });
        const data = (await resp.json()) as { embedding?: number[]; data?: { embedding: number[] }[] };
        const vec = data.embedding ?? data.data?.[0]?.embedding;
        if (!vec) throw new Error("embedding endpoint returned no vector");
        return vec;
      } finally {
        clearTimeout(timer);
      }
    },
  };
}

export default function activate(api: ExtensionAPI): void {
  api.registerTool({
    name: "search_knowledge_base",
    label: "Search Knowledge Base",
    description:
      "Search the project's retrieval knowledge base — a dual-store (shared + " +
      "local-private) index of the documents, files, and links the user has " +
      "ingested — and return the most relevant, credential-scrubbed Markdown " +
      "context blocks. The knowledge base can contain domain-specific and " +
      "recently-indexed information that is NOT in your training data (for " +
      "example, the user's own documentation, notes, product specs, or any " +
      "subject they have indexed). Merges both stores with Reciprocal Rank " +
      "Fusion and degrades to local-only results if the shared store is " +
      "unreachable.",
    promptSnippet:
      "search_knowledge_base: consult the user's indexed knowledge base for " +
      "factual lookups before answering from prior knowledge.",
    promptGuidelines: [
      "When the user asks a factual question — about any subject — first call " +
        "`search_knowledge_base`, because their knowledge base may hold " +
        "authoritative, domain-specific, or more recent information than your " +
        "training data (e.g. their own docs, an indexed website, or any topic " +
        "they have ingested).",
      "Prefer grounded knowledge-base results over unsourced recall, and cite " +
        "the returned sources. If the results are empty or clearly unrelated to " +
        "the question, say so and fall back to general knowledge — do not " +
        "fabricate a citation.",
      "Treat the knowledge base as potentially newer than your training data, " +
        "but only as fresh as its last build; reconcile rather than blindly " +
        "trust, and prefer indexed sources when they conflict with recall.",
    ],
    parameters: paramsSchema,
    async execute(_toolCallId, params) {
      const topK = params.top_k ?? DEFAULT_TOP_K;
      let text: string;
      try {
        const endpoints = loadEndpoints();
        if (!endpoints) {
          text =
            "The knowledge base is not available (no .doxtr/endpoints.json). " +
            "Run a Sphinx build with doxtr-rag first.";
        } else {
          text = await searchKnowledgeBase(
            params.query,
            topK,
            defaultEmbedder(endpoints.embedding.embed_url, endpoints.embedding.model),
            endpoints,
          );
        }
      } catch (err) {
        // A parity violation (dimension/model) is a fatal configuration fault,
        // NOT a connectivity issue — surface it loudly and distinctly so a
        // misconfigured embedder is not mistaken for an offline store.
        const message = scrub(err instanceof Error ? err.message : String(err));
        if (/parity violation/.test(message)) {
          text = `Knowledge base search FAILED (embedding parity): ${message}`;
        } else {
          text = `Knowledge base search could not complete: ${message}`;
        }
      }
      return {
        content: [{ type: "text", text }],
        details: { query: scrub(params.query), top_k: topK },
      };
    },
  });
}
