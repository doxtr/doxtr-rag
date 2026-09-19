/**
 * rrf.ts — Reciprocal Rank Fusion (rank-based, ignores raw distances).
 *
 * RRF score for a document d: sum over each result list L of 1/(k + rank_L(d)),
 * where rank is 1-based position in L. Identical chunks (same id) are deduped
 * and their contributions summed. Distances are intentionally ignored so the
 * two stores' incomparable distance scales don't skew the merge (P6 requirement).
 */

import type { RichMetadata } from "./addressing.js";

export interface RankedChunk {
  id: string;
  text: string;
  metadata: RichMetadata;
}

export interface FusedChunk extends RankedChunk {
  score: number;
}

export const RRF_K = 60;

/**
 * Merge multiple ranked result lists via RRF, dedup by chunk id, and return
 * fused chunks sorted by descending score.
 */
export function reciprocalRankFusion(
  lists: RankedChunk[][],
  k: number = RRF_K,
): FusedChunk[] {
  const byId = new Map<string, FusedChunk>();
  for (const list of lists) {
    list.forEach((chunk, index) => {
      const rank = index + 1; // 1-based
      const contribution = 1 / (k + rank);
      const existing = byId.get(chunk.id);
      if (existing) {
        existing.score += contribution;
      } else {
        byId.set(chunk.id, { ...chunk, score: contribution });
      }
    });
  }
  return [...byId.values()].sort((a, b) => b.score - a.score);
}
