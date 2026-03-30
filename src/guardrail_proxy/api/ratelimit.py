"""Sliding-window in-memory rate limiter for FastAPI.

Design
------
* Per-identity sliding window backed by an in-memory dict of ``deque`` timestamps.
* Identity = API key (when auth is enabled) or client IP address.
* Configuration via Settings::

    rate_limit_requests        - max requests per window (0 = disabled).
    rate_limit_window_seconds  - window size in seconds.

Test isolation
--------------
The limiter singleton is exposed via the ``get_rate_limiter`` FastAPI dependency
so tests can inject a fresh ``SlidingWindowLimiter`` instance per test via
``app.dependency_overrides[get_rate_limiter]``.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from fastapi import Depends, HTTPException, Request

from guardrail_proxy.api.auth import require_api_key
from guardrail_proxy.api.dependencies import get_settings
from guardrail_proxy.config.settings import Settings


class SlidingWindowLimiter:
    """Thread-safe per-identity sliding-window rate limiter."""

    def __init__(self) -> None:
        self._windows: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str, limit: int, window_secs: float) -> bool:
        """
        Return ``True`` and record the timestamp if the request is within limits.

        Returns ``False`` (without recording the timestamp) when throttled.
        """
        now = time.monotonic()
        cutoff = now - window_secs
        with self._lock:
            dq = self._windows.setdefault(key, deque())
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= limit:
                return False
            dq.append(now)
            return True

    def reset(self) -> None:
        """Clear all counters.  Useful in tests to ensure isolation."""
        with self._lock:
            self._windows.clear()


# Process-scoped singleton. Tests override via dependency injection.
_limiter = SlidingWindowLimiter()


def get_rate_limiter() -> SlidingWindowLimiter:
    """Return the process-wide rate limiter instance."""
    return _limiter


def check_rate_limit(
    request: Request,
    settings: Settings = Depends(get_settings),
    api_key: str = Depends(require_api_key),
    limiter: SlidingWindowLimiter = Depends(get_rate_limiter),
) -> None:
    """
    Enforce the per-identity sliding-window rate limit.

    Identity resolution order: API key → client IP → "unknown".
    Rate limiting is skipped entirely when ``Settings.rate_limit_requests == 0``.

    Raises
    ------
    HTTPException(429)
        When the caller has exceeded the configured request rate.
    """
    if settings.rate_limit_requests <= 0:
        return  # rate limiting disabled
    identity = api_key if api_key else (
        request.client.host if request.client else "unknown"
    )
    if not limiter.is_allowed(
        identity,
        settings.rate_limit_requests,
        settings.rate_limit_window_seconds,
    ):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
