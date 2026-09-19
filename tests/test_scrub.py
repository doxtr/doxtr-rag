"""F1 tests for the value-level credential scrubber (CONTRACTS §C3)."""

from __future__ import annotations

from doxtr_rag.scrub import scrub, scrub_mapping_values


def test_scrub_bearer_token() -> None:
    assert "REDACTED" in scrub("Authorization: Bearer abc.def-ghi_123")
    assert "abc.def-ghi_123" not in scrub("Bearer abc.def-ghi_123")


def test_scrub_url_with_credentials() -> None:
    out = scrub("https://user:secretpw@example.com/api")
    assert "secretpw" not in out
    assert "[REDACTED]" in out


def test_scrub_jwt() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.SflKxwRJSMeKKF2QT4"
    assert jwt not in scrub(f"token {jwt}")


def test_scrub_github_tokens() -> None:
    assert "ghp_" not in scrub("ghp_" + "a" * 30)
    assert "gho_" not in scrub("gho_" + "b" * 30)
    assert scrub("github_pat_" + "c" * 30) == "[REDACTED]"


def test_scrub_slack_token() -> None:
    assert "xoxb-" not in scrub("xoxb-123456789-abcdefghij")


def test_scrub_aws_key() -> None:
    assert "AKIA" not in scrub("AKIAIOSFODNN7EXAMPLE")
    assert "ASIA" not in scrub("ASIAIOSFODNN7EXAMPLE")


def test_scrub_pem_private_key() -> None:
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEA...\nabcdef\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out = scrub(f"key:\n{pem}\ntrailer")
    assert "MIIEpAIBAAKCAQEA" not in out
    assert "trailer" in out


def test_scrub_idempotent() -> None:
    s = "Bearer sometoken https://u:p@h/x"
    once = scrub(s)
    assert scrub(once) == once


def test_scrub_negative_cases_untouched() -> None:
    # No credentials -> unchanged.
    clean = "The quick brown fox jumps over section 42."
    assert scrub(clean) == clean
    # A word merely containing 'bearer' as substring of another word is left
    # (pattern requires the Bearer keyword form).
    assert scrub("forbearer") == "forbearer"


def test_scrub_mapping_values_recursive() -> None:
    data = {
        "url": "https://u:pw@h/x",
        "nested": {"auth": "Bearer tok123456"},
        "list": ["ghp_" + "z" * 20, "safe"],
        "num": 5,
    }
    out = scrub_mapping_values(data)
    assert "pw" not in str(out["url"])
    assert "tok123456" not in str(out["nested"])
    assert "safe" in out["list"]  # type: ignore[operator]
    assert out["num"] == 5
