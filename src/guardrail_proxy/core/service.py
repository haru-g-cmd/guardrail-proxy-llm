"""GuardrailService: orchestrates analysis, caching, and audit persistence.

Failure modes
-------------
All storage operations (Redis cache reads/writes, Postgres audit writes) are
individually wrapped in try/except so that a storage outage **never** stops
the proxy from screening prompts.  The service degrades gracefully:

* Redis down  → cache is bypassed; every prompt is re-analysed.
* Postgres down → audit records are silently dropped.

This is an intentional security-first decision: it is safer to analyse a prompt
twice than to allow an unscreened prompt because the cache was unavailable.
"""

from __future__ import annotations

import hashlib

from guardrail_proxy.config.settings import Settings
from guardrail_proxy.core.analyzer import AnalysisResult, PromptAnalyzer
from guardrail_proxy.models.contracts import CheckResponse, GuardrailVerdict
from guardrail_proxy.storage.cache import AssessmentCache
from guardrail_proxy.storage.database import build_engine, build_session_factory, Base
from guardrail_proxy.storage.entities import AuditRecord


class GuardrailService:
    """
    High-level service that drives the full check pipeline.

    Sequence for ``check()``:
    1. Redis cache lookup (skip analysis on hit).
    2. ``PromptAnalyzer.analyze()`` (classifier + PII).
    3. Cache the result.
    4. Persist an audit record.
    5. Return the ``CheckResponse``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._analyzer = PromptAnalyzer(settings)
        self._cache = AssessmentCache(settings)
        self._session_factory = build_session_factory(settings)
        self._ensure_schema()

    # ── Schema bootstrap ──────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """Create database tables if they do not yet exist (fail-safe)."""
        try:
            engine = build_engine(self._settings)
            Base.metadata.create_all(engine)
        except Exception:
            pass  # degraded mode, Postgres not available yet

    # ── Public API ────────────────────────────────────────────────────────

    def check(self, prompt: str, session_id: str | None = None) -> CheckResponse:
        """
        Screen *prompt* and return a :class:`CheckResponse`.

        The full pipeline (cache → analyse → cache-write → audit-write) is
        executed with individual failure isolation at every I/O boundary.
        """
        # 1. Cache read (fail-open)
        try:
            cached = self._cache.get(prompt)
            if cached:
                cached["cached"] = True
                return CheckResponse(**cached)
        except Exception:
            pass

        # 2. Analyse
        result: AnalysisResult = self._analyzer.analyze(prompt)
        response = CheckResponse(
            verdict=result.verdict,
            sanitized_prompt=result.sanitized_prompt,
            maliciousness_score=result.maliciousness_score,
            pii_risk_score=result.pii_risk_score,
            pii_findings=result.pii_findings,
            reason=result.reason,
            cached=False,
        )

        # 3. Cache write (fail-open)
        try:
            self._cache.set(prompt, response.model_dump())
        except Exception:
            pass

        # 4. Audit (fail-open)
        self._persist(prompt, session_id, result)

        return response

    # ── Internal helpers ──────────────────────────────────────────────────

    def _persist(
        self,
        prompt: str,
        session_id: str | None,
        result: AnalysisResult,
    ) -> None:
        """Write an audit record to Postgres (silently skipped on failure)."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        try:
            with self._session_factory() as session:
                record = AuditRecord(
                    session_id=session_id,
                    prompt_hash=prompt_hash,
                    verdict=result.verdict.value,
                    maliciousness_score=result.maliciousness_score,
                    pii_count=len(result.pii_findings),
                    reason=result.reason,
                )
                session.add(record)
                session.commit()
        except Exception:
            pass
