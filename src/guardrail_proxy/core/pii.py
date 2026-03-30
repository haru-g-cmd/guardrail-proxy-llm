"""Regex-based PII detection and redaction.

Design notes
------------
* Uses pre-compiled patterns for zero per-call compilation cost.
* Returns matches sorted by start offset so ``sanitize()`` can do a single
  left-to-right pass without overlap checks.
* Risk scoring uses per-type weights instead of a flat multiplier:
    - High-risk types (SSN, CREDIT_CARD, API_KEY): 0.9 / 0.85 per match
    - Low-risk types (EMAIL, PHONE_US, IP_ADDRESS): 0.3 per match
  This ensures a single SSN exceeds the block threshold (default 0.8)
  while a single email sits at the sanitize threshold (default 0.3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── PII patterns ──────────────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN",         re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("EMAIL",       re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    )),
    ("PHONE_US",    re.compile(
        r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )),
    ("IP_ADDRESS",  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("API_KEY",     re.compile(
        r"\b(?:sk|pk|api_key|apikey|token)[_\-]?[A-Za-z0-9]{16,}\b",
        re.IGNORECASE,
    )),
]

# Per-type risk weights (capped to 1.0 after summation)
_RISK_WEIGHT: dict[str, float] = {
    "SSN":         0.90,
    "CREDIT_CARD": 0.85,
    "API_KEY":     0.85,
    "EMAIL":       0.30,
    "PHONE_US":    0.25,
    "IP_ADDRESS":  0.20,
}
_DEFAULT_WEIGHT = 0.25


# ── Public data class ─────────────────────────────────────────────────────────

@dataclass
class PIIMatch:
    """A single PII span detected in a prompt."""

    entity_type: str
    start: int
    end: int
    replacement: str


# ── Public API ────────────────────────────────────────────────────────────────

def detect(text: str) -> list[PIIMatch]:
    """
    Return all PII matches found in *text*, sorted by start offset.

    Does not de-duplicate overlapping spans; callers should treat the list
    as authoritative left-to-right.
    """
    found: list[PIIMatch] = []
    for entity_type, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            found.append(PIIMatch(
                entity_type=entity_type,
                start=m.start(),
                end=m.end(),
                replacement=f"[{entity_type}]",
            ))
    return sorted(found, key=lambda x: x.start)


def sanitize(text: str, matches: list[PIIMatch]) -> str:
    """
    Replace every matched PII span with its ``[TYPE]`` placeholder token.

    Operates left-to-right; overlapping spans are skipped automatically
    because ``cursor`` advances past each replaced region.
    """
    if not matches:
        return text

    parts: list[str] = []
    cursor = 0
    for m in sorted(matches, key=lambda x: x.start):
        if m.start < cursor:
            continue  # skip overlapping span
        parts.append(text[cursor:m.start])
        parts.append(m.replacement)
        cursor = m.end
    parts.append(text[cursor:])
    return "".join(parts)


def risk_score(matches: list[PIIMatch]) -> float:
    """
    Return a 0–1 risk score that reflects both PII type severity and quantity.

    A single HIGH-risk item (SSN, credit card, API key) returns ≥ 0.85,
    which exceeds the default block threshold of 0.8.
    A single LOW-risk item (email) returns 0.3, which equals the default
    sanitize threshold.
    """
    if not matches:
        return 0.0
    total = sum(_RISK_WEIGHT.get(m.entity_type, _DEFAULT_WEIGHT) for m in matches)
    return round(min(total, 1.0), 4)
