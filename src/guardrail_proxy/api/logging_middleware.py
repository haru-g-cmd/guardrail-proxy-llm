"""Structured JSON access-log middleware for the Guardrail Proxy.

Each HTTP request produces exactly one JSON log line via the ``guardrail_proxy.access``
logger.  The line is emitted **after** the response is sent, so ``status_code`` and
``latency_ms`` are always available.

Security invariants
-------------------
* The raw ``X-API-Key`` header value is **never** logged.
  Only the first 8 hex characters of its SHA-256 hash are recorded so that
  log files can be correlated across requests without exposing the secret.
* Raw prompt text is never logged.  Prompts arrive in the request body and
  are analysed internally; only the resulting verdict appears in the log.

Log record fields
-----------------
``request_id``   Random UUID assigned at middleware entry (also propagated as
                 ``X-Request-ID`` response header for client correlation).
``method``       HTTP method (GET, POST, …).
``path``         URL path without query string.
``status_code``  Integer HTTP response status.
``latency_ms``   Wall-clock duration in milliseconds, rounded to 2 d.p.
``identity``     SHA-256[:8] of ``X-API-Key`` if present, else ``"anonymous"``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_log = logging.getLogger("guardrail_proxy.access")


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        # Attach any extra fields stored in the record
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ):
                payload[key] = value
        return json.dumps(payload, default=str)


def _redact_key(raw: str | None) -> str:
    """Return first 8 hex chars of SHA-256(raw), or ``'anonymous'``."""
    if not raw:
        return "anonymous"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return digest[:8]


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that emits one structured JSON log line per request."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = str(uuid.uuid4())
        identity = _redact_key(request.headers.get("x-api-key"))
        t0 = time.perf_counter()

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        response.headers["X-Request-ID"] = request_id

        _log.info(
            "request",
            extra={
                "request_id": request_id,
                "method":      request.method,
                "path":        request.url.path,
                "status_code": response.status_code,
                "latency_ms":  latency_ms,
                "identity":    identity,
            },
        )
        return response


def configure_access_logger(level: int = logging.INFO) -> None:
    """
    Attach a ``JSONFormatter``-backed ``StreamHandler`` to the access logger.

    Called once from the application factory so that uvicorn's log capture
    picks up the structured output.  Calling this more than once is safe;
    duplicate handlers are skipped.
    """
    if _log.handlers:
        return  # Already configured; avoid duplicate output
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    _log.addHandler(handler)
    _log.setLevel(level)
    _log.propagate = False  # Prevent double-emission via root logger
