/**
 * addressing.ts — mirror of the Python C4.1 addressing + C2 metadata shape.
 *
 * P6 reads the resolved endpoints from `.doxtr/endpoints.json` (written by P5)
 * rather than re-deriving addressing, and mirrors `DocumentChunk.from_store_metadata`
 * so the flattened Chroma metadata inflates identically across the language
 * boundary (CONTRACTS §C2 / §C4.1).
 */

export const EMBEDDING_DIMENSION = 1024;

export type AccessScope = "SHARED_ORG" | "LOCAL_PRIVATE";

/** One scope's resolved connection descriptor from endpoints.json. */
export interface ScopeEndpoint {
  mode: "embedded" | "http";
  host: string;
  port: number;
  ssl: boolean;
  tenant: string;
  database: string;
  collection: string;
  header_keys: string[];
}

export interface EndpointsDescriptor {
  embedding: { model: string; dimension: number; embed_url?: string | null };
  shared: ScopeEndpoint;
  private: ScopeEndpoint;
  /**
   * Portable per-mount sidecar stores (e.g. a `vault/` folder), each served
   * over its own loopback Chroma server. Present when the Python side has
   * built + served the mount's `<mount>/.doxtr/chroma`. Optional for backward
   * compatibility with older endpoints.json files that predate this field.
   */
  mounts?: MountEndpoint[];
}

/** One external mount's resolved connection descriptor. */
export interface MountEndpoint extends ScopeEndpoint {
  /** The mount directory path (provenance / logging only). */
  path: string;
  /** The mount's declared access scope (fail-closed to LOCAL_PRIVATE). */
  scope: AccessScope;
}

/** Inflated (rich) metadata, mirroring DocumentChunk.from_store_metadata. */
export interface RichMetadata {
  source_file: string;
  section_title: string;
  breadcrumb: string[];
  access_scope: AccessScope;
  content_hash: string;
  provenance: Record<string, unknown>;
}

function safeJsonParse<T>(value: unknown, fallback: T): T {
  if (typeof value !== "string") return (value ?? fallback) as T;
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

/** Mirror of DocumentChunk.from_store_metadata: inflate flat Chroma metadata. */
export function fromStoreMetadata(flat: Record<string, unknown>): RichMetadata {
  const scopeRaw = flat.access_scope;
  const access_scope: AccessScope =
    scopeRaw === "SHARED_ORG" ? "SHARED_ORG" : "LOCAL_PRIVATE"; // fail-closed
  return {
    source_file: String(flat.source_file ?? ""),
    section_title: String(flat.section_title ?? ""),
    breadcrumb: safeJsonParse<string[]>(flat.breadcrumb, []),
    access_scope,
    content_hash: String(flat.content_hash ?? ""),
    provenance: safeJsonParse<Record<string, unknown>>(flat.provenance, {}),
  };
}
