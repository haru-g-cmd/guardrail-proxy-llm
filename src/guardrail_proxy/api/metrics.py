"""In-memory rolling metrics store for the Guardrail Proxy.

Counters and latency samples accumulate in memory only. Restarting the proxy
resets everything to zero.  No persistence layer is involved.

The store is exposed as a FastAPI dependency (``get_metrics_store``) so that
tests can inject a fresh ``MetricsStore`` per test via
``app.dependency_overrides[get_metrics_store]``.
"""

from __future__ import annotations

import threading
from collections import deque

_MAX_LATENCY_SAMPLES = 1_000


class MetricsStore:
    """
    Thread-safe in-memory accumulator for per-request metrics.

    Collected per request
    ---------------------
    * verdict: one of ``block``, ``sanitize``, ``allow``
    * end-to-end latency in milliseconds
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total: int = 0
        self._blocked: int = 0
        self._sanitized: int = 0
        self._allowed: int = 0
        self._latencies: deque[float] = deque(maxlen=_MAX_LATENCY_SAMPLES)

    def record(self, verdict: str, latency_ms: float) -> None:
        """Atomically record one completed request."""
        with self._lock:
            self._total += 1
            if verdict == "block":
                self._blocked += 1
            elif verdict == "sanitize":
                self._sanitized += 1
            else:
                self._allowed += 1
            self._latencies.append(latency_ms)

    def snapshot(self) -> dict:
        """Return a point-in-time metrics snapshot as a plain dict."""
        with self._lock:
            total = self._total
            lats = sorted(self._latencies)
            n = len(lats)

            def _pct(p: float) -> float:
                if not lats:
                    return 0.0
                idx = max(0, min(n - 1, int(n * p) - 1))
                return round(lats[idx], 2)

            return {
                "total_requests":  total,
                "total_blocked":   self._blocked,
                "total_sanitized": self._sanitized,
                "total_allowed":   self._allowed,
                "block_rate":      round(self._blocked / total, 4) if total else 0.0,
                "latency_p95_ms":  _pct(0.95),
                "latency_p99_ms":  _pct(0.99),
            }

    def reset(self) -> None:
        """Reset all counters and samples.  Useful in tests."""
        with self._lock:
            self._total = 0
            self._blocked = 0
            self._sanitized = 0
            self._allowed = 0
            self._latencies.clear()


# Process-scoped singleton. Tests override via dependency injection.
_metrics = MetricsStore()


def get_metrics_store() -> MetricsStore:
    """Return the process-wide metrics store instance."""
    return _metrics
