"""XLink security classifier (P4 — CONTRACTS §C1 provisional scope producer).

Classifies Jira/Confluence xlink targets as an **allow-list**: only an
affirmatively-unrestricted resource yields ``SHARED_ORG``; every error,
ambiguity, or restriction routes to ``LOCAL_PRIVATE`` (fail-closed,
CONVENTIONS §Resilience & Timeout Policy).

- **Jira** — active Issue Level Security (``fields.security != null``).
- **Confluence** — ``/wiki/rest/api/content/{id}/restriction`` read locks.
- **403 / errors / timeouts / unparseable / missing fields** → ``LOCAL_PRIVATE``.

All URLs are scrubbed with the shared :func:`~doxtr_rag.scrub.scrub` before any
logging or auditing. Scope is **provisional**; P5 finalizes via
``most_restrictive``.

**xlink integration note (spike):** ``sphinxcontrib-xlink`` resolves entries to
``id :: Title :: URL :: tags`` and exposes the target URL, not the raw API
payload. The classifier therefore fetches the API payload itself from the
resolved URL.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from doxtr_rag.schema import AccessControl
from doxtr_rag.scrub import scrub

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.monotonic()

# -- Resilience & Timeout Policy (CONVENTIONS) -------------------------------
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 10.0
MAX_RETRIES = 2  # only on 5xx / 429
_RETRY_STATUS = {429, 500, 502, 503, 504}
_FAIL_CLOSED_STATUS = {401, 403, 429}


@dataclass
class ClassificationResult:
    """The outcome of classifying one xlink target."""

    scope: AccessControl
    reason: str
    scrubbed_url: str
    resource_id: str = ""


class NetworkBudget:
    """Per-build network budget (CONVENTIONS §Resilience & Timeout Policy).

    A hard cap so a slow Atlassian instance cannot stall the whole build: once
    the cumulative wall-clock spent on classification network calls exceeds
    ``max_seconds`` (or ``max_calls`` is reached), all remaining classifications
    short-circuit to ``LOCAL_PRIVATE`` without making a request.
    """

    def __init__(self, max_seconds: float = 60.0, max_calls: int = 1000) -> None:
        self.max_seconds = max_seconds
        self.max_calls = max_calls
        self.spent_seconds = 0.0
        self.calls = 0

    def exhausted(self) -> bool:
        return self.spent_seconds >= self.max_seconds or self.calls >= self.max_calls

    def charge(self, seconds: float) -> None:
        self.spent_seconds += seconds
        self.calls += 1


def _timeout() -> "httpx.Timeout":
    return httpx.Timeout(
        connect=CONNECT_TIMEOUT, read=READ_TIMEOUT, write=READ_TIMEOUT, pool=READ_TIMEOUT
    )


def _get_with_policy(
    client: httpx.Client, url: str, headers: Optional[dict[str, str]] = None
) -> Optional[httpx.Response]:
    """GET honoring the resilience policy. Returns None on a fail-closed outcome.

    Retries at most twice on 5xx/429 with exponential backoff; NEVER retries
    401/403 (fail closed immediately). Any transport error returns None.
    """
    import time

    backoff = 0.1
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.get(url, headers=headers or {}, follow_redirects=False)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            logger.info("doxtr-rag: xlink fetch error for %s: %s", scrub(url), exc)
            return None
        if resp.status_code in (401, 403):
            return resp  # fail-closed; caller maps to LOCAL_PRIVATE, no retry
        if resp.status_code in _RETRY_STATUS and attempt < MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2
            continue
        return resp
    return None


def _is_redirect_to_login(resp: httpx.Response) -> bool:
    # A 200 that is actually a login page (redirect-to-login pattern).
    if resp.status_code in (301, 302, 303, 307, 308):
        return True
    location = resp.headers.get("location", "")
    return bool(re.search(r"login|signin|sso", location, re.IGNORECASE))


def classify_jira(
    url: str,
    *,
    resource_id: str = "",
    client: Optional[httpx.Client] = None,
    headers: Optional[dict[str, str]] = None,
    budget: Optional[NetworkBudget] = None,
) -> ClassificationResult:
    """Classify a Jira issue URL by Issue Level Security (allow-list).

    ``fields.security`` present and non-null → restricted → ``LOCAL_PRIVATE``.
    ``fields.security`` explicitly null AND payload parseable → ``SHARED_ORG``.
    Missing field / non-2xx / unparseable / error → ``LOCAL_PRIVATE``.
    """
    safe_url = scrub(url)
    if budget is not None and budget.exhausted():
        return ClassificationResult(
            AccessControl.LOCAL_PRIVATE, "network_budget_exceeded", safe_url, resource_id
        )
    own = client is None
    client = client or httpx.Client(timeout=_timeout())
    _t0 = _now()
    try:
        resp = _get_with_policy(client, url, headers)
        if resp is None:
            return ClassificationResult(
                AccessControl.LOCAL_PRIVATE, "fetch_failed", safe_url, resource_id
            )
        if resp.status_code in _FAIL_CLOSED_STATUS:
            return ClassificationResult(
                AccessControl.LOCAL_PRIVATE, f"http_{resp.status_code}", safe_url, resource_id
            )
        if resp.status_code != 200 or _is_redirect_to_login(resp):
            return ClassificationResult(
                AccessControl.LOCAL_PRIVATE, f"non_ok_{resp.status_code}", safe_url, resource_id
            )
        try:
            payload = resp.json()
            fields = payload["fields"]
        except (ValueError, KeyError, TypeError):
            return ClassificationResult(
                AccessControl.LOCAL_PRIVATE,
                "unparseable_or_missing_fields",
                safe_url,
                resource_id,
            )
        if "security" not in fields:
            return ClassificationResult(
                AccessControl.LOCAL_PRIVATE, "missing_security_field", safe_url, resource_id
            )
        if fields["security"] is not None:
            return ClassificationResult(
                AccessControl.LOCAL_PRIVATE, "jira_ils_restricted", safe_url, resource_id
            )
        return ClassificationResult(
            AccessControl.SHARED_ORG, "jira_unrestricted", safe_url, resource_id
        )
    finally:
        if budget is not None:
            budget.charge(_now() - _t0)
        if own:
            client.close()


def classify_confluence(
    base_url: str,
    content_id: str,
    *,
    resource_id: str = "",
    client: Optional[httpx.Client] = None,
    headers: Optional[dict[str, str]] = None,
    budget: Optional[NetworkBudget] = None,
) -> ClassificationResult:
    """Classify a Confluence page via the restriction endpoint (allow-list).

    Queries ``/wiki/rest/api/content/{id}/restriction``. Any active user/group
    read restriction → ``LOCAL_PRIVATE``. Only an affirmatively-empty restriction
    set (parseable, no read locks) → ``SHARED_ORG``. Errors → ``LOCAL_PRIVATE``.
    """
    endpoint = f"{base_url.rstrip('/')}/wiki/rest/api/content/{content_id}/restriction"
    safe_url = scrub(endpoint)
    if budget is not None and budget.exhausted():
        return ClassificationResult(
            AccessControl.LOCAL_PRIVATE, "network_budget_exceeded", safe_url, resource_id
        )
    own = client is None
    client = client or httpx.Client(timeout=_timeout())
    _t0 = _now()
    try:
        resp = _get_with_policy(client, endpoint, headers)
        if resp is None:
            return ClassificationResult(
                AccessControl.LOCAL_PRIVATE, "fetch_failed", safe_url, resource_id
            )
        if resp.status_code in _FAIL_CLOSED_STATUS:
            return ClassificationResult(
                AccessControl.LOCAL_PRIVATE, f"http_{resp.status_code}", safe_url, resource_id
            )
        if resp.status_code != 200 or _is_redirect_to_login(resp):
            return ClassificationResult(
                AccessControl.LOCAL_PRIVATE, f"non_ok_{resp.status_code}", safe_url, resource_id
            )
        try:
            payload = resp.json()
        except ValueError:
            return ClassificationResult(
                AccessControl.LOCAL_PRIVATE, "unparseable_body", safe_url, resource_id
            )
        if _confluence_has_read_restriction(payload):
            return ClassificationResult(
                AccessControl.LOCAL_PRIVATE, "confluence_read_restricted", safe_url, resource_id
            )
        return ClassificationResult(
            AccessControl.SHARED_ORG, "confluence_unrestricted", safe_url, resource_id
        )
    finally:
        if budget is not None:
            budget.charge(_now() - _t0)
        if own:
            client.close()


# Heuristics for the unified dispatcher.
_JIRA_ISSUE_RE = re.compile(r"/rest/api/\d+/issue/|/browse/[A-Z][A-Z0-9]+-\d+")
_CONFLUENCE_PAGE_RE = re.compile(r"/wiki/(rest/api/content/(\d+)|spaces/|pages/(\d+))")


def classify_xlink(
    url: str,
    *,
    resource_id: str = "",
    client: Optional[httpx.Client] = None,
    headers: Optional[dict[str, str]] = None,
    budget: Optional[NetworkBudget] = None,
) -> ClassificationResult:
    """Unified per-link classifier: detect provider, dispatch, fail closed.

    This is the SINGLE entry point P5 calls per resolved xlink so provider
    detection and Confluence content-id parsing never leak into P5. Anything
    that cannot be affirmatively classified (unknown provider) fails closed to
    ``LOCAL_PRIVATE`` with reason ``unknown_provider``.
    """
    safe_url = scrub(url)
    # Confluence: extract base + content id from the resolved URL.
    m = _CONFLUENCE_PAGE_RE.search(url)
    if "/wiki/" in url and m:
        content_id = m.group(2) or m.group(3) or ""
        base_url = url.split("/wiki/", 1)[0]
        if content_id:
            return classify_confluence(
                base_url,
                content_id,
                resource_id=resource_id,
                client=client,
                headers=headers,
                budget=budget,
            )
    # Jira: REST issue endpoint or human /browse/KEY-123.
    if _JIRA_ISSUE_RE.search(url):
        rest_url = _to_jira_rest_url(url)
        return classify_jira(
            rest_url,
            resource_id=resource_id,
            client=client,
            headers=headers,
            budget=budget,
        )
    return ClassificationResult(
        AccessControl.LOCAL_PRIVATE, "unknown_provider", safe_url, resource_id
    )


def _to_jira_rest_url(url: str) -> str:
    """Rewrite a human Jira ``/browse/KEY-123`` URL to the REST issue endpoint.

    An already-REST URL is returned unchanged. Keeps the Jira URL-shape
    knowledge inside P4 so P5 never rewrites URLs.
    """
    m = re.search(r"/browse/([A-Z][A-Z0-9]+-\d+)", url)
    if m:
        base = url.split("/browse/", 1)[0]
        return f"{base}/rest/api/2/issue/{m.group(1)}"
    return url


def _confluence_has_read_restriction(payload: Any) -> bool:
    """Return True if the restriction payload declares any read lock.

    Fail-closed: a missing/unexpected/malformed shape is treated as RESTRICTED
    (returns True). Only an affirmatively-parseable, affirmatively-empty read
    restriction set returns False (unrestricted).
    """
    try:
        if not isinstance(payload, dict):
            return True  # unparseable shape -> fail closed
        candidates: list[Any] = []
        has_recognizable_shape = False
        if "read" in payload:
            has_recognizable_shape = True
            candidates.append(payload["read"])
        results = payload.get("results")
        if results is not None:
            if not isinstance(results, list):
                return True  # malformed -> fail closed
            has_recognizable_shape = True
            for item in results:
                if isinstance(item, dict) and item.get("operation") == "read":
                    candidates.append(item)
        if not has_recognizable_shape:
            # No 'read' key and no 'results' list at all -> unknown shape.
            return True
        for c in candidates:
            if not isinstance(c, dict):
                return True  # malformed read object -> fail closed
            restrictions = c.get("restrictions", {})
            if not isinstance(restrictions, dict):
                return True
            users = restrictions.get("user", {})
            groups = restrictions.get("group", {})
            for coll in (users, groups):
                if isinstance(coll, dict) and coll.get("results"):
                    return True
                if isinstance(coll, dict) and coll.get("size", 0):
                    return True
        # Parseable, recognizable, and no read locks found -> unrestricted.
        return False
    except Exception:  # noqa: BLE001 - fail closed on any surprise
        return True
