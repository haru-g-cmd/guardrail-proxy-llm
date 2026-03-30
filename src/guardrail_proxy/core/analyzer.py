"""Unified prompt analysis engine: combines classifier + PII into a single verdict.

Decision logic (evaluated in priority order)
--------------------------------------------
1. Maliciousness score >= ``maliciousness_threshold`` → BLOCK
2. PII risk score >= ``pii_block_threshold``          → BLOCK
3. PII risk score >= ``pii_sanitize_threshold``       → SANITIZE (prompt is redacted)
4. Otherwise                                          → ALLOW
"""

from __future__ import annotations

from dataclasses import dataclass, field

from guardrail_proxy.config.settings import Settings
from guardrail_proxy.core import classifier as _cls_module
from guardrail_proxy.core import pii as _pii_module
from guardrail_proxy.models.contracts import GuardrailVerdict, PIIFinding


@dataclass
class AnalysisResult:
    """Full analysis output returned by ``PromptAnalyzer.analyze()``."""

    verdict: GuardrailVerdict
    maliciousness_score: float
    pii_risk_score: float
    pii_findings: list[PIIFinding]
    sanitized_prompt: str | None
    reason: str | None


class PromptAnalyzer:
    """
    Orchestrates the classifier and PII detector into a single guardrail verdict.

    Thread-safety: instances are stateless after construction; concurrent calls
    to ``analyze()`` are safe.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._classifier = _cls_module.DistilBERTClassifier(settings)

    def analyze(self, prompt: str) -> AnalysisResult:
        """
        Analyse *prompt* and return a structured :class:`AnalysisResult`.

        Steps
        -----
        1. Score maliciousness via DistilBERT (or heuristic fallback).
        2. Detect and quantify PII.
        3. Apply the three-tier verdict logic.
        """
        mal_score = self._classifier.score(prompt)
        raw_matches = _pii_module.detect(prompt)
        pii_score = _pii_module.risk_score(raw_matches)

        findings = [
            PIIFinding(
                entity_type=m.entity_type,
                start=m.start,
                end=m.end,
                replacement=m.replacement,
            )
            for m in raw_matches
        ]

        # ── Tier 1: Block on malicious content ────────────────────────────
        if mal_score >= self._settings.maliciousness_threshold:
            return AnalysisResult(
                verdict=GuardrailVerdict.BLOCK,
                maliciousness_score=mal_score,
                pii_risk_score=pii_score,
                pii_findings=findings,
                sanitized_prompt=None,
                reason=(
                    f"Maliciousness score {mal_score:.3f} ≥ "
                    f"threshold {self._settings.maliciousness_threshold}"
                ),
            )

        # ── Tier 2: Block on high-risk PII ────────────────────────────────
        if pii_score >= self._settings.pii_block_threshold:
            return AnalysisResult(
                verdict=GuardrailVerdict.BLOCK,
                maliciousness_score=mal_score,
                pii_risk_score=pii_score,
                pii_findings=findings,
                sanitized_prompt=None,
                reason=f"High-risk PII detected (score {pii_score:.3f})",
            )

        # ── Tier 3: Sanitise on moderate PII ──────────────────────────────
        if pii_score >= self._settings.pii_sanitize_threshold:
            sanitized = _pii_module.sanitize(prompt, raw_matches)
            return AnalysisResult(
                verdict=GuardrailVerdict.SANITIZE,
                maliciousness_score=mal_score,
                pii_risk_score=pii_score,
                pii_findings=findings,
                sanitized_prompt=sanitized,
                reason=f"PII redacted (score {pii_score:.3f})",
            )

        # ── Tier 4: Allow ─────────────────────────────────────────────────
        return AnalysisResult(
            verdict=GuardrailVerdict.ALLOW,
            maliciousness_score=mal_score,
            pii_risk_score=pii_score,
            pii_findings=findings,
            sanitized_prompt=None,
            reason=None,
        )
