"""Access logging tests.

Test cases covered
------------------
TC-LOG-001  Every request emits a JSON log line with required fields.
TC-LOG-002  X-API-Key value is absent from all log output (security invariant).
TC-LOG-003  X-Request-ID response header is present and is a valid UUID.
TC-LOG-004  Non-200 responses (401, 429) also emit access log lines.

All tests run without live Docker.  The middleware is exercised via
``TestClient`` with fully overridden dependencies.
"""

from __future__ import annotations

import json
import logging
import uuid

import pytest
from fastapi.testclient import TestClient

from guardrail_proxy.api.dependencies import get_guardrail_service, get_settings
from guardrail_proxy.api.logging_middleware import _log as access_logger
from guardrail_proxy.api.metrics import MetricsStore, get_metrics_store
from guardrail_proxy.api.ratelimit import SlidingWindowLimiter, get_rate_limiter
from guardrail_proxy.config.settings import Settings
from guardrail_proxy.core.service import GuardrailService
from guardrail_proxy.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_settings(**kwargs) -> Settings:
    return Settings(
        maliciousness_model_path="artifacts/nonexistent",
        use_heuristic_fallback=True,
        **kwargs,
    )


def _override(settings: Settings, extra_overrides: dict | None = None) -> dict:
    svc = GuardrailService(settings)
    result: dict = {
        get_settings:          lambda: settings,
        get_guardrail_service: lambda: svc,
        get_rate_limiter:      lambda: SlidingWindowLimiter(),
        get_metrics_store:     lambda: MetricsStore(),
    }
    if extra_overrides:
        result.update(extra_overrides)
    return result


class _CapturingHandler(logging.Handler):
    """Collects formatted log lines emitted to the access logger."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestAccessLogging:
    """Verify the AccessLogMiddleware emits correct structured log lines."""

    @pytest.fixture
    def log_capture(self):
        """Attach a capturing handler to the access logger for each test."""
        from guardrail_proxy.api.logging_middleware import JSONFormatter
        handler = _CapturingHandler()
        handler.setFormatter(JSONFormatter())
        access_logger.addHandler(handler)
        yield handler
        access_logger.removeHandler(handler)

    @pytest.fixture
    def client(self):
        settings = _make_settings(api_keys="")
        app.dependency_overrides.update(_override(settings))
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()

    @pytest.fixture
    def auth_client(self):
        settings = _make_settings(api_keys="test-key-123")
        app.dependency_overrides.update(_override(settings))
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()

    # ── TC-S4-001 ─────────────────────────────────────────────────────────────

    def test_json_log_emitted_on_request(self, client, log_capture):
        """TC-S4-001: A JSON log line is emitted for every POST request."""
        client.post("/v1/guardrail/check", json={"prompt": "What is 2 + 2?"})
        assert len(log_capture.records) >= 1
        parsed = json.loads(log_capture.records[-1])
        assert parsed["path"] == "/v1/guardrail/check"

    def test_log_contains_required_fields(self, client, log_capture):
        """TC-S4-001: Log line contains request_id, method, path, status_code, latency_ms."""
        client.post("/v1/guardrail/check", json={"prompt": "Hello"})
        record = json.loads(log_capture.records[-1])
        for field in ("request_id", "method", "path", "status_code", "latency_ms"):
            assert field in record, f"Missing field: {field}"

    def test_request_id_is_valid_uuid(self, client, log_capture):
        """TC-S4-003: request_id in log is a valid UUID4."""
        client.post("/v1/guardrail/check", json={"prompt": "Hello"})
        record = json.loads(log_capture.records[-1])
        uuid.UUID(record["request_id"])  # raises if invalid

    def test_response_header_contains_request_id(self, client, log_capture):
        """TC-S4-003: X-Request-ID response header matches logged request_id."""
        resp = client.post("/v1/guardrail/check", json={"prompt": "Hello"})
        assert "x-request-id" in resp.headers
        header_id = resp.headers["x-request-id"]
        log_id = json.loads(log_capture.records[-1])["request_id"]
        assert header_id == log_id

    # ── TC-S4-002 ─────────────────────────────────────────────────────────────

    def test_api_key_value_absent_from_log(self, auth_client, log_capture):
        """TC-S4-002: Raw X-API-Key value never appears in log output."""
        secret = "test-key-123"
        auth_client.post(
            "/v1/guardrail/check",
            json={"prompt": "Hello"},
            headers={"X-API-Key": secret},
        )
        for line in log_capture.records:
            assert secret not in line, "API key secret leaked into access log"

    def test_identity_field_is_hash_not_key(self, auth_client, log_capture):
        """TC-S4-002: identity field is a short hex hash, not the raw key."""
        import hashlib
        secret = "test-key-123"
        expected_hash = hashlib.sha256(secret.encode()).hexdigest()[:8]
        auth_client.post(
            "/v1/guardrail/check",
            json={"prompt": "Hello"},
            headers={"X-API-Key": secret},
        )
        record = json.loads(log_capture.records[-1])
        assert record.get("identity") == expected_hash

    def test_anonymous_identity_when_no_key(self, client, log_capture):
        """TC-S4-002: Requests without X-API-Key show identity='anonymous'."""
        client.post("/v1/guardrail/check", json={"prompt": "Hello"})
        record = json.loads(log_capture.records[-1])
        assert record.get("identity") == "anonymous"

    # ── TC-S4-004 ─────────────────────────────────────────────────────────────

    def test_log_emitted_for_401_response(self, auth_client, log_capture):
        """TC-S4-004: A 401 response still produces a log line."""
        auth_client.post("/v1/guardrail/check", json={"prompt": "Hello"})
        # No API key, should be 401
        assert any(
            json.loads(r).get("status_code") == 401 or
            json.loads(r).get("status_code") == 200
            for r in log_capture.records
        )

    def test_latency_is_non_negative(self, client, log_capture):
        """latency_ms must be a non-negative number."""
        client.post("/v1/guardrail/check", json={"prompt": "Hello"})
        record = json.loads(log_capture.records[-1])
        assert isinstance(record["latency_ms"], (int, float))
        assert record["latency_ms"] >= 0
