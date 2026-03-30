"""Tests for the PII detection, sanitization, and risk scoring module.

Test cases covered
------------------
TC-PII-001  Email address -> verdict SANITIZE
TC-PII-002  SSN -> verdict BLOCK (risk_score >= pii_block_threshold 0.8)
TC-PII-003  Credit card -> risk_score >= 0.8
TC-PII-004  Plain benign prompt -> no detections, risk_score 0.0
TC-PII-005  Multiple low-risk items -> cumulative score capped correctly
TC-PII-006  Sanitized text replaces all detected spans
"""

import pytest

from guardrail_proxy.core.pii import PIIMatch, detect, risk_score, sanitize


# ── TC-PII-02: benign prompt ──────────────────────────────────────────────────

def test_no_pii_in_benign_prompt():
    """A plain question should produce zero detections and a 0.0 risk score."""
    matches = detect("What is the capital of France?")
    assert matches == []
    assert risk_score(matches) == 0.0


# ── TC-S1-003: email triggers sanitize tier ───────────────────────────────────

def test_email_detected():
    matches = detect("Contact me at user@example.com for details.")
    entity_types = {m.entity_type for m in matches}
    assert "EMAIL" in entity_types


def test_email_risk_score_equals_sanitize_threshold():
    """Single email risk score (0.30) should equal the default sanitize threshold."""
    matches = detect("user@example.com")
    score = risk_score(matches)
    assert score == pytest.approx(0.30, abs=1e-4)


# ── TC-S1-004: SSN triggers block tier ───────────────────────────────────────

def test_ssn_detected():
    matches = detect("My SSN is 123-45-6789.")
    types = {m.entity_type for m in matches}
    assert "SSN" in types


def test_ssn_risk_score_exceeds_block_threshold():
    """Single SSN risk score (0.90) must exceed default block threshold (0.80)."""
    matches = detect("123-45-6789")
    score = risk_score(matches)
    assert score >= 0.80


# ── TC-PII-01: credit card ────────────────────────────────────────────────────

def test_credit_card_triggers_block():
    matches = detect("Card: 4111111111111111")
    score = risk_score(matches)
    assert score >= 0.80


# ── TC-PII-03: capped at 1.0 ─────────────────────────────────────────────────

def test_multiple_pii_score_capped_at_one():
    """Cumulative PII risk score must never exceed 1.0."""
    text = "SSN: 123-45-6789, CC: 4111111111111111, email: x@y.com, key: sk_abc123examplekeyhere"
    matches = detect(text)
    score = risk_score(matches)
    assert score <= 1.0


# ── TC-PII-04: sanitize replaces spans ───────────────────────────────────────

def test_sanitize_replaces_email():
    text = "Send results to admin@company.org please."
    matches = detect(text)
    sanitized = sanitize(text, matches)
    assert "admin@company.org" not in sanitized
    assert "[EMAIL]" in sanitized


def test_sanitize_replaces_ssn():
    text = "Client SSN: 987-65-4321 on file."
    matches = detect(text)
    sanitized = sanitize(text, matches)
    assert "987-65-4321" not in sanitized
    assert "[SSN]" in sanitized


def test_sanitize_no_matches_returns_original():
    """Sanitize with no matches must return the original string unchanged."""
    text = "No PII here."
    assert sanitize(text, []) == text
