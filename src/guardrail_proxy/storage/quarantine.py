"""Redis-backed quarantine queue for blocked prompts.

Each blocked prompt is stored as a JSON object containing:

* ``prompt_hash`` - SHA-256 hex digest of the blocked prompt (no raw text)
* ``reason``      - human-readable verdict reason (may be null)
* ``ts``          - ISO-8601 UTC timestamp

Entries are stored on a Redis list (newest first, via ``LPUSH``) and capped to
``_MAX_SIZE`` via ``LTRIM``.  All operations are **fail-open**: a Redis outage
silently drops the quarantine entry without affecting the guardrail verdict.

Queue key: ``gp:quarantine``
"""

from __future__ import annotations

import datetime
import hashlib
import json

import redis as redis_lib

from guardrail_proxy.config.settings import Settings

_QUEUE_KEY = "gp:quarantine"
_MAX_SIZE = 1_000


class QuarantineQueue:
    """Fail-open Redis-backed queue storing metadata for blocked prompts."""

    def __init__(self, settings: Settings) -> None:
        self._client = redis_lib.from_url(settings.redis_url, decode_responses=True)

    def push(self, prompt: str, reason: str | None) -> None:
        """
        Enqueue a blocked-prompt record.

        Silently discards the entry when Redis is unavailable.
        """
        entry = json.dumps({
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "reason": reason,
            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        })
        try:
            self._client.lpush(_QUEUE_KEY, entry)
            self._client.ltrim(_QUEUE_KEY, 0, _MAX_SIZE - 1)
        except Exception:
            pass

    def depth(self) -> int:
        """Return the current queue depth (0 on Redis error)."""
        try:
            return int(self._client.llen(_QUEUE_KEY))
        except Exception:
            return 0
