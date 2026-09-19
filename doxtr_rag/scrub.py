"""Value-level credential scrubber (CONTRACTS §C3, CONVENTIONS Cross-Cutting #3).

This is the single canonical source of truth for credential redaction across
the entire package: schema, harvester, extractors, audit, and storage all call
:func:`scrub`. The TypeScript extension (P6) mirrors this pattern set verbatim;
if the patterns here change, that is a contract change and P6 must re-sync.

Scrubbing is **value-level, not key-name based**: it operates on string values
regardless of the key they are stored under, so a token hidden in a URL, a free
text field, or a metadata value is caught the same way.
"""

from __future__ import annotations

import re
from typing import Final

_REDACTION: Final[str] = "[REDACTED]"

# NOTE: order matters only for readability; each pattern is applied
# independently and the operation is idempotent (a second pass is a no-op
# because "[REDACTED]" contains no credential-shaped substrings).
_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    # PEM private-key blocks (multiline). Must run before generic patterns.
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    # URLs with embedded userinfo credentials: scheme://user:pass@host
    re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),
    # Authorization: Bearer <token>  /  bare "Bearer <token>"
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-+/=]+", re.IGNORECASE),
    # JWT-shaped: three base64url segments separated by dots.
    re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
    # GitHub tokens: ghp_, gho_, ghu_, ghs_, ghr_, github_pat_
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    # Slack tokens: xoxb-, xoxa-, xoxp-, xoxr-, xoxs-
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{8,}\b"),
    # AWS access key IDs: AKIA / ASIA followed by 16 uppercase alnum.
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
)


def scrub(value: str) -> str:
    """Redact credential-shaped substrings from a string value.

    Args:
        value: An arbitrary string value (chunk text, metadata value, URL, or
            log line). Non-``str`` inputs must be coerced by the caller.

    Returns:
        The value with every credential-shaped span replaced by
        ``"[REDACTED]"``. The transformation is deterministic and idempotent:
        ``scrub(scrub(x)) == scrub(x)``.

    Patterns covered:
        - PEM private-key blocks.
        - URLs with embedded ``user:pass@`` credentials.
        - ``Bearer <token>`` authorization values.
        - JWT-shaped tokens (``header.payload.signature``).
        - GitHub tokens (``ghp_``/``gho_``/``ghu_``/``ghs_``/``ghr_``/``github_pat_``).
        - Slack tokens (``xox[baprs]-``).
        - AWS access key IDs (``AKIA``/``ASIA``).
    """
    result = value
    for pattern in _PATTERNS:
        result = pattern.sub(_REDACTION, result)
    return result


def scrub_mapping_values(mapping: dict[str, object]) -> dict[str, object]:
    """Return a copy of ``mapping`` with :func:`scrub` applied to str values.

    Nested dicts and lists are traversed recursively; non-str scalars are left
    untouched. This is a convenience for scrubbing metadata dicts before
    serialization.
    """

    def _walk(obj: object) -> object:
        if isinstance(obj, str):
            return scrub(obj)
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_walk(v) for v in obj)
        return obj

    return {k: _walk(v) for k, v in mapping.items()}
