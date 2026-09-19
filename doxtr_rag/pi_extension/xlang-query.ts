/**
 * xlang-query.ts — cross-language integration harness entry (P7).
 *
 * Invoked by the Python-driven cross-language integration test. Reads
 * `.doxtr/endpoints.json` (written by a real P5 ingest), queries the loopback
 * Chroma server via the P6 `searchKnowledgeBase` path using a DETERMINISTIC
 * fake bge-m3 embedder (same text -> same 1024-dim vector as the Python side),
 * and prints a JSON result the Python test asserts on.
 *
 * Usage: tsx xlang-query.ts "<query>" [top_k]
 * Env:   DOXTR_RAG_ENDPOINTS -> path to endpoints.json
 */

import { searchKnowledgeBase, defaultEmbedder, type QueryEmbedder } from "./chroma-rag.js";
import { loadEndpoints } from "./chroma-rag.js";
import { EMBEDDING_DIMENSION } from "./addressing.js";

/** Deterministic 1024-dim embedder mirroring the Python test's FakeProvider. */
export function deterministicEmbedder(): QueryEmbedder {
  return {
    model: "bge-m3",
    async embed(text: string): Promise<number[]> {
      // Match the Python cross-language test provider: a constant fill keyed
      // on text length modulo, producing identical vectors for identical text.
      const seed = (text.length % 7) / 10.0;
      return new Array(EMBEDDING_DIMENSION).fill(seed);
    },
  };
}

/** Pick the real bge-m3 embedder when an embed URL is available (endpoints
 *  embed_url from the auto-started server, or DOXTR_RAG_EMBED_URL); otherwise
 *  the deterministic stub for weight-free tests. */
function pickEmbedder(embedUrl?: string | null): QueryEmbedder {
  return embedUrl || process.env.DOXTR_RAG_EMBED_URL
    ? defaultEmbedder(embedUrl)
    : deterministicEmbedder();
}

async function main(): Promise<void> {
  const query = process.argv[2] ?? "";
  const topK = Number(process.argv[3] ?? "5");
  const endpoints = loadEndpoints();
  if (!endpoints) {
    console.log(JSON.stringify({ ok: false, error: "no endpoints.json" }));
    process.exit(2);
  }
  try {
    const markdown = await searchKnowledgeBase(
      query,
      topK,
      pickEmbedder(endpoints.embedding.embed_url),
      endpoints,
    );
    console.log(
      JSON.stringify({
        ok: true,
        markdown,
        embedding: endpoints.embedding,
        shared: endpoints.shared,
        private: endpoints.private,
      }),
    );
  } catch (err) {
    console.log(
      JSON.stringify({ ok: false, error: err instanceof Error ? err.message : String(err) }),
    );
    process.exit(1);
  }
}

void main();
