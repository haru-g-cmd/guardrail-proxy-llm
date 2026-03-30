"""FastAPI dependency injection for shared service instances.

Uses ``functools.lru_cache`` so that ``Settings`` and ``GuardrailService`` are
instantiated once per process (not once per request).  Tests override these
via ``app.dependency_overrides``.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from guardrail_proxy.config.settings import Settings
from guardrail_proxy.core.service import GuardrailService


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (loaded from env / .env file)."""
    return Settings()


def get_guardrail_service(
    settings: Settings = Depends(get_settings),
) -> GuardrailService:
    """Return a GuardrailService backed by the shared settings."""
    return GuardrailService(settings)
