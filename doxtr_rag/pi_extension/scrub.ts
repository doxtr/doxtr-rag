/**
 * scrub.ts — value-level credential scrubber (MIRROR of doxtr_rag/scrub.py, C3).
 *
 * This MUST implement the identical pattern set as the Python `scrub()` (F1 owns
 * the canonical list; CONTRACTS §C3). If the Python patterns change, this file
 * must re-sync. Applied to all returned text/metadata before it reaches the LLM.
 */

const REDACTION = "[REDACTED]";

// Order mirrors doxtr_rag/scrub.py. Each is applied independently; the result
// is idempotent because "[REDACTED]" contains no credential-shaped substrings.
const PATTERNS: RegExp[] = [
  // PEM private-key blocks (multiline).
  /-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----/g,
  // URLs with embedded userinfo credentials: scheme://user:pass@host
  /([a-zA-Z][a-zA-Z0-9+.-]*:\/\/)[^\s/:@]+:[^\s/@]+@/gi,
  // Authorization: Bearer <token> / bare "Bearer <token>"
  /\bBearer\s+[A-Za-z0-9._\-+/=]+/gi,
  // JWT-shaped: three base64url segments separated by dots.
  /\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/g,
  // GitHub tokens: ghp_, gho_, ghu_, ghs_, ghr_
  /\bgh[pousr]_[A-Za-z0-9]{16,}\b/g,
  // GitHub fine-grained PAT
  /\bgithub_pat_[A-Za-z0-9_]{20,}\b/g,
  // Slack tokens: xoxb-, xoxa-, xoxp-, xoxr-, xoxs-
  /\bxox[baprs]-[A-Za-z0-9-]{8,}\b/g,
  // AWS access key IDs: AKIA / ASIA + 16 uppercase alnum.
  /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/g,
];

/** Redact credential-shaped substrings from a string value (deterministic, idempotent). */
export function scrub(value: string): string {
  let result = value;
  for (const pattern of PATTERNS) {
    result = result.replace(pattern, REDACTION);
  }
  return result;
}

/** Recursively scrub all string values in an arbitrary JSON-like value. */
export function scrubValue(value: unknown): unknown {
  if (typeof value === "string") return scrub(value);
  if (Array.isArray(value)) return value.map(scrubValue);
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = scrubValue(v);
    }
    return out;
  }
  return value;
}
