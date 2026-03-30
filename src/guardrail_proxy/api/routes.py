"""FastAPI endpoint definitions for the Guardrail Proxy.

Endpoints
---------
POST /v1/guardrail/check   - Analyse a prompt and return a verdict.
POST /v1/guardrail/proxy   - Analyse and, if not blocked, forward to downstream LLM.
GET  /healthz              - Liveness + dependency health.
GET  /statusz              - Operator configuration snapshot.
GET  /metricsz             - Rolling request metrics (block-rate, latency p95/p99).

Per-request features
--------------------
* Auth       : ``X-API-Key`` header enforced on check/proxy when ``API_KEYS`` is set.
* Rate limit : Per-identity sliding-window limit; 429 when exceeded.
* Tenant     : Per-request threshold overrides via ``X-Maliciousness-Threshold``,
               ``X-PII-Block-Threshold``, ``X-PII-Sanitize-Threshold`` headers.
* Metrics    : Latency + verdict counters accumulated in ``MetricsStore``.
* Quarantine : Blocked-prompt metadata pushed to Redis list ``gp:quarantine``.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from guardrail_proxy.api.auth import require_api_key
from guardrail_proxy.api.dependencies import get_guardrail_service, get_settings
from guardrail_proxy.api.metrics import MetricsStore, get_metrics_store
from guardrail_proxy.api.ratelimit import check_rate_limit
from guardrail_proxy.config.settings import Settings
from guardrail_proxy.core.service import GuardrailService
from guardrail_proxy.integrations.downstream import DownstreamAdapter
from guardrail_proxy.models.contracts import (
    CheckRequest,
    CheckResponse,
    GuardrailVerdict,
    ProxyRequest,
    ProxyResponse,
)
from guardrail_proxy.storage.cache import AssessmentCache
from guardrail_proxy.storage.quarantine import QuarantineQueue

router = APIRouter()


# ── Per-tenant settings helper ────────────────────────────────────────────────

def _apply_tenant_overrides(settings: Settings, request: Request) -> Settings:
    """
    Return a copy of *settings* with per-request tenant threshold overrides.

    Supported headers (case-insensitive, as per HTTP spec):

    * ``X-Maliciousness-Threshold``  → maliciousness_threshold
    * ``X-PII-Block-Threshold``      → pii_block_threshold
    * ``X-PII-Sanitize-Threshold``   → pii_sanitize_threshold

    Returns *settings* **unchanged** (same object) when no override headers
    are present, so ``effective_settings is settings`` can be used to skip
    rebuilding the service for the common case.

    Invalid (non-numeric) header values are silently ignored.
    """
    overrides: dict = {}
    for header, field in (
        ("x-maliciousness-threshold", "maliciousness_threshold"),
        ("x-pii-block-threshold",     "pii_block_threshold"),
        ("x-pii-sanitize-threshold",  "pii_sanitize_threshold"),
    ):
        raw = request.headers.get(header)
        if raw is not None:
            try:
                overrides[field] = float(raw)
            except ValueError:
                pass
    return settings.model_copy(update=overrides) if overrides else settings


# ── Protected endpoints ───────────────────────────────────────────────────────

@router.post("/v1/guardrail/check", response_model=CheckResponse)
def check_prompt(
    req: CheckRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    service: GuardrailService = Depends(get_guardrail_service),
    _auth: str = Depends(require_api_key),
    _rate: None = Depends(check_rate_limit),
    metrics: MetricsStore = Depends(get_metrics_store),
) -> CheckResponse:
    """
    Analyse *prompt* and return a verdict without forwarding to any LLM.

    Per-tenant threshold overrides are applied via request headers.  The
    result is durably recorded in the metrics store, and blocked prompts
    are enqueued in the Redis quarantine queue (fail-open).
    """
    effective_settings = _apply_tenant_overrides(settings, request)
    if effective_settings is not settings:
        service = GuardrailService(effective_settings)

    t0 = time.perf_counter()
    result = service.check(req.prompt, req.session_id)
    metrics.record(result.verdict.value, (time.perf_counter() - t0) * 1000)

    if result.verdict == GuardrailVerdict.BLOCK:
        QuarantineQueue(effective_settings).push(req.prompt, result.reason)

    return result


@router.post("/v1/guardrail/proxy", response_model=ProxyResponse)
def proxy_prompt(
    req: ProxyRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    service: GuardrailService = Depends(get_guardrail_service),
    _auth: str = Depends(require_api_key),
    _rate: None = Depends(check_rate_limit),
    metrics: MetricsStore = Depends(get_metrics_store),
) -> ProxyResponse:
    """
    Analyse *prompt* and, if not blocked, forward to the downstream LLM.

    The effective prompt sent downstream is the sanitised version when the
    verdict is SANITIZE, or the original prompt when the verdict is ALLOW.
    Per-tenant threshold overrides are honoured.
    """
    effective_settings = _apply_tenant_overrides(settings, request)
    if effective_settings is not settings:
        service = GuardrailService(effective_settings)

    t0 = time.perf_counter()
    check_result = service.check(req.prompt, req.session_id)
    metrics.record(check_result.verdict.value, (time.perf_counter() - t0) * 1000)

    if check_result.verdict == GuardrailVerdict.BLOCK:
        QuarantineQueue(effective_settings).push(req.prompt, check_result.reason)

    llm_response: str | None = None
    if check_result.verdict != GuardrailVerdict.BLOCK:
        effective_prompt = check_result.sanitized_prompt or req.prompt
        adapter = DownstreamAdapter(effective_settings)
        llm_response = adapter.call(effective_prompt, req.model)

    return ProxyResponse(
        verdict=check_result.verdict,
        llm_response=llm_response,
        sanitized_prompt=check_result.sanitized_prompt,
        maliciousness_score=check_result.maliciousness_score,
        pii_findings=check_result.pii_findings,
        reason=check_result.reason,
    )


# ── Unauthenticated utility endpoints ─────────────────────────────────────────

@router.get("/healthz")
def health(settings: Settings = Depends(get_settings)) -> JSONResponse:
    """
    Liveness and dependency health check.

    Returns ``status: ok`` when Redis is reachable, ``status: degraded``
    when it is not.  The proxy continues to function in degraded mode.
    """
    cache = AssessmentCache(settings)
    redis_ok = cache.ping()
    return JSONResponse({
        "status": "ok" if redis_ok else "degraded",
        "redis": "up" if redis_ok else "down",
        "proxy_url":  f"http://{settings.host}:{settings.port}",
        "docs_url":   f"http://{settings.host}:{settings.port}/docs",
        "health_url": f"http://{settings.host}:{settings.port}/healthz",
    })


@router.get("/statusz")
def statusz(settings: Settings = Depends(get_settings)) -> JSONResponse:
    """
    Configuration snapshot for operators.

    Exposes service addresses, active thresholds, and feature flags.
    Never exposes raw credentials or API key values.
    """
    return JSONResponse({
        "proxy_url":          f"http://{settings.host}:{settings.port}",
        "docs_url":           f"http://{settings.host}:{settings.port}/docs",
        "health_url":         f"http://{settings.host}:{settings.port}/healthz",
        "classifier_source":  settings.maliciousness_model_path,
        "thresholds": {
            "maliciousness":  settings.maliciousness_threshold,
            "pii_block":      settings.pii_block_threshold,
            "pii_sanitize":   settings.pii_sanitize_threshold,
        },
        "auth_enabled": bool(settings.api_keys.strip()),
        "rate_limit": {
            "requests_per_window": settings.rate_limit_requests,
            "window_seconds":      settings.rate_limit_window_seconds,
        },
    })


@router.get("/metricsz")
def metricsz(metrics: MetricsStore = Depends(get_metrics_store)) -> JSONResponse:
    """
    Rolling in-process metrics snapshot.

    Returns request counters and latency percentiles accumulated since the
    last process start.  Values are approximate (in-memory only, not persisted).
    """
    return JSONResponse(metrics.snapshot())
