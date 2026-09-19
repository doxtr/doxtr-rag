"""P4 tests for the xlink security classifier (CONTRACTS §C1, Resilience Policy)."""

from __future__ import annotations

import httpx
import pytest

from doxtr_rag import security
from doxtr_rag.schema import AccessControl


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=security._timeout())


# -- Jira Issue Level Security -----------------------------------------------
def test_jira_unrestricted_shared() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"fields": {"security": None}})

    with _client(handler) as c:
        r = security.classify_jira("https://jira/rest/api/2/issue/PROJ-1", client=c)
    assert r.scope is AccessControl.SHARED_ORG
    assert r.reason == "jira_unrestricted"


def test_jira_ils_restricted_private() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"fields": {"security": {"id": "10001"}}})

    with _client(handler) as c:
        r = security.classify_jira("https://jira/rest/api/2/issue/PROJ-2", client=c)
    assert r.scope is AccessControl.LOCAL_PRIVATE
    assert r.reason == "jira_ils_restricted"


def test_jira_missing_security_field_failclosed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"fields": {"summary": "x"}})

    with _client(handler) as c:
        r = security.classify_jira("https://jira/i", client=c)
    assert r.scope is AccessControl.LOCAL_PRIVATE
    assert r.reason == "missing_security_field"


def test_jira_403_failclosed_no_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, json={})

    with _client(handler) as c:
        r = security.classify_jira("https://jira/i", client=c)
    assert r.scope is AccessControl.LOCAL_PRIVATE
    assert r.reason == "http_403"
    assert calls["n"] == 1  # never retried


def test_jira_unparseable_body_failclosed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>login</html>")

    with _client(handler) as c:
        r = security.classify_jira("https://jira/i", client=c)
    assert r.scope is AccessControl.LOCAL_PRIVATE


def test_jira_5xx_retries_then_failclosed() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={})

    with _client(handler) as c:
        r = security.classify_jira("https://jira/i", client=c)
    assert r.scope is AccessControl.LOCAL_PRIVATE
    # initial + 2 retries = 3 attempts
    assert calls["n"] == 3


def test_jira_redirect_to_login_failclosed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://sso/login"})

    with _client(handler) as c:
        r = security.classify_jira("https://jira/i", client=c)
    assert r.scope is AccessControl.LOCAL_PRIVATE


def test_jira_timeout_failclosed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    with _client(handler) as c:
        r = security.classify_jira("https://jira/i", client=c)
    assert r.scope is AccessControl.LOCAL_PRIVATE
    assert r.reason == "fetch_failed"


# -- Confluence restriction endpoint -----------------------------------------
def test_confluence_unrestricted_shared() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/wiki/rest/api/content/123/restriction" in str(request.url)
        return httpx.Response(
            200,
            json={"results": [{"operation": "read", "restrictions": {"user": {"results": []}, "group": {"results": []}}}]},
        )

    with _client(handler) as c:
        r = security.classify_confluence("https://conf", "123", client=c)
    assert r.scope is AccessControl.SHARED_ORG
    assert r.reason == "confluence_unrestricted"


def test_confluence_read_restricted_private() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"operation": "read", "restrictions": {"user": {"results": [{"accountId": "u1"}]}}}]},
        )

    with _client(handler) as c:
        r = security.classify_confluence("https://conf", "123", client=c)
    assert r.scope is AccessControl.LOCAL_PRIVATE
    assert r.reason == "confluence_read_restricted"


def test_confluence_group_restriction_private() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"read": {"restrictions": {"group": {"size": 1}}}}
        )

    with _client(handler) as c:
        r = security.classify_confluence("https://conf", "9", client=c)
    assert r.scope is AccessControl.LOCAL_PRIVATE


def test_confluence_403_failclosed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    with _client(handler) as c:
        r = security.classify_confluence("https://conf", "1", client=c)
    assert r.scope is AccessControl.LOCAL_PRIVATE


def test_confluence_unparseable_failclosed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with _client(handler) as c:
        r = security.classify_confluence("https://conf", "1", client=c)
    assert r.scope is AccessControl.LOCAL_PRIVATE
    assert r.reason == "unparseable_body"


def test_url_scrubbed_in_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    url = "https://user:secret@jira/rest/api/2/issue/X?token=ghp_" + "a" * 20
    with _client(handler) as c:
        r = security.classify_jira(url, client=c)
    assert "secret" not in r.scrubbed_url
    assert "ghp_" not in r.scrubbed_url


def test_restriction_shape_surprise_failclosed() -> None:
    assert security._confluence_has_read_restriction(None) is True
    assert security._confluence_has_read_restriction({"read": "weird"}) is True


# -- unified classify_xlink dispatcher ---------------------------------------
def test_classify_xlink_jira_browse_url() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"fields": {"security": None}})

    with _client(handler) as c:
        r = security.classify_xlink(
            "https://jira.example.com/browse/PROJ-42", resource_id="x1", client=c
        )
    assert r.scope is AccessControl.SHARED_ORG
    assert r.resource_id == "x1"
    # Human /browse/ URL was rewritten to the REST endpoint internally.
    assert "/rest/api/2/issue/PROJ-42" in seen["url"]


def test_classify_xlink_confluence_wiki_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/wiki/rest/api/content/12345/restriction" in str(request.url)
        return httpx.Response(200, json={"results": []})

    with _client(handler) as c:
        r = security.classify_xlink(
            "https://conf.example.com/wiki/rest/api/content/12345", client=c
        )
    assert r.scope is AccessControl.SHARED_ORG


def test_classify_xlink_unknown_provider_failclosed() -> None:
    r = security.classify_xlink("https://example.com/some/page")
    assert r.scope is AccessControl.LOCAL_PRIVATE
    assert r.reason == "unknown_provider"


# -- per-build network budget ------------------------------------------------
def test_network_budget_short_circuits() -> None:
    budget = security.NetworkBudget(max_seconds=0.0, max_calls=1000)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"fields": {"security": None}})

    with _client(handler) as c:
        # First call proceeds and charges time; budget already exhausted (0s).
        r = security.classify_jira("https://jira/i", client=c, budget=budget)
    assert r.scope is AccessControl.LOCAL_PRIVATE
    assert r.reason == "network_budget_exceeded"
    assert calls["n"] == 0  # never made the request


def test_network_budget_charges_and_counts() -> None:
    budget = security.NetworkBudget(max_seconds=100.0, max_calls=1)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"fields": {"security": None}})

    with _client(handler) as c:
        r1 = security.classify_jira("https://jira/i", client=c, budget=budget)
        r2 = security.classify_jira("https://jira/i", client=c, budget=budget)
    assert r1.scope is AccessControl.SHARED_ORG
    # Second call hits max_calls=1 -> short-circuit.
    assert r2.reason == "network_budget_exceeded"
