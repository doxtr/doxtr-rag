import { describe, it, expect, vi, beforeEach } from "vitest";

// Capture every getCollection() call so we can assert on the arguments the
// real queryScope() passes through to chromadb. This exercises the actual
// production code path (searchKnowledgeBase -> queryScope -> getCollection)
// rather than a re-implementation.
const getCollectionCalls: Array<Record<string, unknown>> = [];

// Mock the chromadb package BEFORE importing chroma-rag so the mocked
// ChromaClient is what queryScope constructs. The fake collection returns a
// deterministic query result; the important assertions are about the
// arguments getCollection() receives (the no-op embeddingFunction), which is
// the behavior the patch introduces.
vi.mock("chromadb", () => {
  class ChromaClient {
    constructor(_opts: unknown) {}
    async getCollection(args: Record<string, unknown>) {
      getCollectionCalls.push(args);
      return {
        async query(_q: unknown) {
          return {
            ids: [["chunk-1"]],
            documents: [["hello world"]],
            metadatas: [
              [
                {
                  source_file: "s",
                  section_title: "S",
                  breadcrumb: JSON.stringify(["s"]),
                  access_scope: "SHARED_ORG",
                  content_hash: "h",
                  provenance: JSON.stringify({}),
                },
              ],
            ],
          };
        },
      };
    }
  }
  return { ChromaClient };
});

// Import AFTER the mock is registered.
import { searchKnowledgeBase, type QueryEmbedder } from "../chroma-rag.js";
import type { EndpointsDescriptor } from "../addressing.js";

const endpoints: EndpointsDescriptor = {
  embedding: { model: "bge-m3", dimension: 1024 },
  shared: {
    mode: "http", host: "127.0.0.1", port: 8000, ssl: false,
    tenant: "t", database: "doxtr_shared", collection: "bge_m3_1024", header_keys: [],
  },
  private: {
    mode: "http", host: "127.0.0.1", port: 8000, ssl: false,
    tenant: "t", database: "doxtr_private", collection: "bge_m3_1024", header_keys: [],
  },
};

const embedder: QueryEmbedder = {
  model: "bge-m3",
  embed: async () => new Array(1024).fill(0.01),
};

describe("queryScope no-op embeddingFunction (chromadb 3.x DefaultEmbeddingFunction suppression)", () => {
  beforeEach(() => {
    getCollectionCalls.length = 0;
  });

  it("passes a no-op embeddingFunction to getCollection for every scope", async () => {
    const text = await searchKnowledgeBase("q", 3, embedder, endpoints);
    // Round-trip succeeded (both scopes reachable via the mock).
    expect(text).toContain("hello world");

    // One call per scope (shared + private).
    expect(getCollectionCalls.length).toBe(2);

    for (const args of getCollectionCalls) {
      // The collection name is still passed.
      expect(typeof args.name).toBe("string");
      // The patch's contract: an embeddingFunction MUST be supplied so
      // chromadb 3.x does not instantiate DefaultEmbeddingFunction (which
      // requires the optional @chroma-core/default-embed package). This
      // assertion fails if the embeddingFunction argument is removed.
      expect(args.embeddingFunction).toBeDefined();
      const ef = args.embeddingFunction as { generate?: unknown };
      expect(typeof ef.generate).toBe("function");
    }
  });

  it("supplies an embedder whose generate() is a no-op returning [] (never calls @chroma-core/default-embed)", async () => {
    await searchKnowledgeBase("q", 3, embedder, endpoints);
    expect(getCollectionCalls.length).toBeGreaterThan(0);
    const ef = getCollectionCalls[0].embeddingFunction as {
      generate: (texts: string[]) => Promise<number[][]>;
    };
    // generate() returns an empty matrix without importing or invoking the
    // optional default-embed package — proving the built-in embedder is never
    // exercised.
    const out = await ef.generate(["anything", "here"]);
    expect(out).toEqual([]);
  });
});
