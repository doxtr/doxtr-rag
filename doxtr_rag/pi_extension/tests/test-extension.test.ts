import { describe, it, expect } from "vitest";
import { scrub, scrubValue } from "../scrub.js";
import { reciprocalRankFusion, RRF_K } from "../rrf.js";
import { fromStoreMetadata, EMBEDDING_DIMENSION } from "../addressing.js";
import {
  assertDimension,
  assertModelParity,
  renderMarkdown,
  searchKnowledgeBase,
  loadEndpoints,
  type QueryEmbedder,
} from "../chroma-rag.js";
import type { EndpointsDescriptor } from "../addressing.js";
import type { RankedChunk } from "../rrf.js";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

// ---- scrub (C3 mirror) -----------------------------------------------------
describe("scrub", () => {
  it("redacts bearer/jwt/github/slack/aws/url-cred/pem", () => {
    expect(scrub("Bearer abc.def-ghi_123")).not.toContain("abc.def");
    expect(scrub("https://u:pw@host/x")).not.toContain("pw");
    expect(scrub("ghp_" + "a".repeat(30))).toBe("[REDACTED]");
    expect(scrub("xoxb-123456789-abcdefghij")).not.toContain("xoxb-1");
    expect(scrub("AKIAIOSFODNN7EXAMPLE")).not.toContain("AKIA");
    const jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.SflKxwRJSMeKKF2QT4";
    expect(scrub(`t ${jwt}`)).not.toContain(jwt);
    const pem =
      "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----";
    expect(scrub(pem)).toBe("[REDACTED]");
  });

  it("is idempotent and leaves clean text unchanged", () => {
    const clean = "The quick brown fox in section 42.";
    expect(scrub(clean)).toBe(clean);
    const once = scrub("Bearer tok https://u:p@h/x");
    expect(scrub(once)).toBe(once);
  });

  it("scrubValue recurses", () => {
    const out = scrubValue({ a: "Bearer tok123456", b: ["ghp_" + "z".repeat(20)], n: 5 }) as Record<
      string,
      unknown
    >;
    expect(JSON.stringify(out)).not.toContain("tok123456");
    expect(JSON.stringify(out)).not.toContain("ghp_z");
    expect(out.n).toBe(5);
  });
});

// ---- RRF -------------------------------------------------------------------
function rc(id: string): RankedChunk {
  return {
    id,
    text: `text ${id}`,
    metadata: {
      source_file: "s",
      section_title: id,
      breadcrumb: [id],
      access_scope: "LOCAL_PRIVATE",
      content_hash: "h",
      provenance: {},
    },
  };
}

describe("reciprocalRankFusion", () => {
  it("dedupes identical chunks and sums contributions", () => {
    const listA = [rc("x"), rc("y")];
    const listB = [rc("x"), rc("z")];
    const fused = reciprocalRankFusion([listA, listB]);
    const ids = fused.map((f) => f.id);
    expect(new Set(ids).size).toBe(ids.length); // deduped
    // x appears rank1 in both -> highest score.
    expect(fused[0].id).toBe("x");
    const xScore = 1 / (RRF_K + 1) + 1 / (RRF_K + 1);
    expect(fused[0].score).toBeCloseTo(xScore, 10);
  });

  it("is rank-based, ignoring raw distances", () => {
    // Same items, different insertion order changes rank-based scores only.
    const fused = reciprocalRankFusion([[rc("a"), rc("b")]]);
    expect(fused[0].id).toBe("a");
    expect(fused[0].score).toBeCloseTo(1 / (RRF_K + 1), 10);
  });

  it("handles empty lists", () => {
    expect(reciprocalRankFusion([])).toEqual([]);
    expect(reciprocalRankFusion([[]])).toEqual([]);
  });

  it("keeps items unique to one list (shared-only AND private-only survive)", () => {
    // Models the dual-store merge: shared list and private list each contribute
    // their own results; a private-only chunk must appear in the fused output.
    const shared = [rc("shared-1"), rc("shared-2")];
    const priv = [rc("private-1")];
    const fused = reciprocalRankFusion([shared, priv]);
    const ids = fused.map((f) => f.id);
    expect(ids).toContain("private-1");
    expect(ids).toContain("shared-1");
    expect(ids.length).toBe(3);
  });
});

// ---- metadata inflation (C2 mirror) ----------------------------------------
describe("fromStoreMetadata", () => {
  it("inflates the flat shape and fails scope closed", () => {
    const rich = fromStoreMetadata({
      source_file: "source/a.rst",
      section_title: "S",
      breadcrumb: JSON.stringify(["A", "B"]),
      access_scope: "SHARED_ORG",
      content_hash: "h",
      provenance: JSON.stringify({ page: 3 }),
    });
    expect(rich.breadcrumb).toEqual(["A", "B"]);
    expect(rich.provenance).toEqual({ page: 3 });
    expect(rich.access_scope).toBe("SHARED_ORG");
  });

  it("fails closed for unknown scope and bad json", () => {
    const rich = fromStoreMetadata({
      access_scope: "weird",
      breadcrumb: "not-json",
      provenance: "{bad",
    });
    expect(rich.access_scope).toBe("LOCAL_PRIVATE");
    expect(rich.breadcrumb).toEqual([]);
    expect(rich.provenance).toEqual({});
  });
});

// ---- dimension parity ------------------------------------------------------
describe("assertDimension", () => {
  it("passes for 1024 and throws otherwise", () => {
    expect(() => assertDimension(EMBEDDING_DIMENSION)).not.toThrow();
    expect(() => assertDimension(768)).toThrow(/parity/);
  });
});

describe("assertModelParity", () => {
  it("tolerates org prefix and throws on drift", () => {
    expect(() => assertModelParity("bge-m3", "BAAI/bge-m3")).not.toThrow();
    expect(() => assertModelParity("BAAI/bge-m3", "bge-m3")).not.toThrow();
    expect(() => assertModelParity("bge-large-en", "bge-m3")).toThrow(/model parity/);
  });
});

// ---- rendering -------------------------------------------------------------
describe("renderMarkdown", () => {
  it("scrubs rendered context and reports empty", () => {
    expect(renderMarkdown([])).toMatch(/No results/);
    const md = renderMarkdown([
      { ...rc("a"), score: 0.5, text: "leak Bearer abcdef123456" },
    ]);
    expect(md).not.toContain("abcdef123456");
    expect(md).toContain("LOCAL_PRIVATE");
  });
});

// ---- offline fallback (Promise.allSettled) ---------------------------------
describe("searchKnowledgeBase offline fallback", () => {
  const endpoints: EndpointsDescriptor = {
    embedding: { model: "bge-m3", dimension: 1024 },
    shared: {
      mode: "http", host: "127.0.0.1", port: 1, ssl: false,
      tenant: "t", database: "doxtr_shared", collection: "bge_m3_1024", header_keys: [],
    },
    private: {
      mode: "http", host: "127.0.0.1", port: 1, ssl: false,
      tenant: "t", database: "doxtr_private", collection: "bge_m3_1024", header_keys: [],
    },
  };
  const embedder: QueryEmbedder = { model: "bge-m3", embed: async () => new Array(1024).fill(0.01) };

  it("does not throw when both stores are unreachable", async () => {
    const text = await searchKnowledgeBase("q", 3, embedder, endpoints);
    expect(typeof text).toBe("string");
    expect(text).toMatch(/No results|not available/);
  });

  it("throws only on a fatal dimension mismatch (before any network call)", async () => {
    const bad: QueryEmbedder = { model: "bge-m3", embed: async () => [0.1, 0.2, 0.3] };
    await expect(searchKnowledgeBase("q", 3, bad, endpoints)).rejects.toThrow(/parity/);
  });

  it("throws on a query-time MODEL parity mismatch", async () => {
    const drifted: QueryEmbedder = {
      model: "bge-large-en",
      embed: async () => new Array(1024).fill(0.01),
    };
    await expect(searchKnowledgeBase("q", 3, drifted, endpoints)).rejects.toThrow(
      /model parity/,
    );
  });
});

// ---- endpoints loading -----------------------------------------------------
describe("loadEndpoints", () => {
  it("reads .doxtr/endpoints.json and returns null when absent", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "doxtr-ep-"));
    expect(loadEndpoints(dir)).toBeNull();
    fs.mkdirSync(path.join(dir, ".doxtr"));
    const desc = { embedding: { model: "bge-m3", dimension: 1024 } };
    fs.writeFileSync(path.join(dir, ".doxtr", "endpoints.json"), JSON.stringify(desc));
    const loaded = loadEndpoints(dir);
    expect(loaded?.embedding.dimension).toBe(1024);
  });
});

// ---- tool registration + KB-first guidance ---------------------------------
import activate from "../chroma-rag.js";

describe("search_knowledge_base registration", () => {
  function capture() {
    const tools: Record<string, any> = {};
    const api = {
      registerTool(def: any) {
        tools[def.name] = def;
      },
    } as unknown as import("@earendil-works/pi-coding-agent").ExtensionAPI;
    activate(api);
    return tools;
  }

  it("registers the tool with { query, top_k? } params", () => {
    const tools = capture();
    const t = tools["search_knowledge_base"];
    expect(t).toBeTruthy();
    expect(t.parameters).toBeTruthy();
  });

  it("advertises KB-first guidance, phrased domain-agnostically", () => {
    const t = capture()["search_knowledge_base"];
    expect(typeof t.promptSnippet).toBe("string");
    expect(Array.isArray(t.promptGuidelines)).toBe(true);
    const all = (t.promptGuidelines as string[]).join(" ").toLowerCase();
    // KB-first + freshness rationale present...
    expect(all).toContain("first call");
    expect(all).toContain("more recent");
    expect(all).toContain("training data");
    // ...and it is NOT tied to any single domain (e.g. doxtr).
    expect(all).not.toContain("doxtr");
    // reconcile, don't blindly trust; fall back cleanly when empty.
    expect(all).toContain("reconcile");
    expect(all).toContain("fall back");
  });
});
