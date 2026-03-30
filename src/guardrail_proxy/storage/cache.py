"""Redis-backed cache for prompt assessment results.

Cache key   : ``gp:v1:<sha256(prompt + thresholds)>``  - keyed by content
              AND the active threshold configuration so that tenant overrides
              never collide with results cached at a different threshold.
TTL         : 5 minutes (configurable via ``_TTL_SECONDS``).
Serialisation: JSON (redis ``decode_responses=True`` handles str ↔ bytes).

Design decision: ``from_url()`` builds a connection pool lazily; it does not
connect immediately.  This means the cache can be constructed safely even when
Redis is not yet available, and ``ping()`` can be used for health checking.
"""

from __future__ import annotations

import hashlib
import json

import redis as redis_lib

from guardrail_proxy.config.settings import Settings

_TTL_SECONDS = 300  # 5 minutes


class AssessmentCache:
    """
    SHA-256-keyed JSON cache for :class:`~guardrail_proxy.models.contracts.CheckResponse`
    payloads.

    The cache key incorporates both the prompt content and the active threshold
    settings so that per-tenant threshold overrides always produce distinct cache
    entries and never serve a verdict cached under a different threshold.

    Parameters
    ----------
    settings : Settings
        Application settings.  ``redis_url`` and the three threshold fields are
        consumed here.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = redis_lib.from_url(settings.redis_url, decode_responses=True)
        # Threshold fingerprint is stable for the lifetime of this instance.
        self._threshold_tag = (
            f"{settings.maliciousness_threshold:.4f}"
            f":{settings.pii_block_threshold:.4f}"
            f":{settings.pii_sanitize_threshold:.4f}"
        )

    # ── Cache operations ──────────────────────────────────────────────────

    def _key(self, prompt: str) -> str:
        """Return the namespaced cache key for *prompt* under the active thresholds."""
        payload = f"{self._threshold_tag}:{prompt}"
        return "gp:v1:" + hashlib.sha256(payload.encode()).hexdigest()

    def get(self, prompt: str) -> dict | None:
        """
        Return the cached assessment dict for *prompt*, or ``None`` on a miss.

        Raises
        ------
        redis.exceptions.ConnectionError
            Propagated to caller who is expected to handle it gracefully.
        """
        raw = self._client.get(self._key(prompt))
        return json.loads(raw) if raw else None

    def set(self, prompt: str, payload: dict) -> None:
        """
        Cache *payload* for *prompt* with a fixed TTL.

        Raises
        ------
        redis.exceptions.ConnectionError
            Propagated to caller who is expected to handle it gracefully.
        """
        self._client.setex(self._key(prompt), _TTL_SECONDS, json.dumps(payload))

    # ── Health ────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """
        Return ``True`` if Redis responds to a PING command.

        Never raises; connection errors are caught and return ``False``.
        """
        try:
            return bool(self._client.ping())
        except Exception:
            return False
