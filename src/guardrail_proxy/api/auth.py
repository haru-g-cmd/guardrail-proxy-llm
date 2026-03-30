"""API key authentication dependency for the Guardrail Proxy.

When ``Settings.api_keys`` is empty (the default), authentication is **disabled**
and all requests are accepted.  This lets the service run without configuring
keys and keeps test setup simple.

When non-empty, every request to a protected endpoint must carry a valid
``X-API-Key`` header or a 401 Unauthorized response is returned.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from guardrail_proxy.api.dependencies import get_settings
from guardrail_proxy.config.settings import Settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    api_key: str | None = Security(_API_KEY_HEADER),
    settings: Settings = Depends(get_settings),
) -> str:
    """
    Validate the ``X-API-Key`` request header.

    Returns
    -------
    str
        The accepted API key, or an empty string when auth is disabled.

    Raises
    ------
    HTTPException(401)
        When auth is enabled and the key is missing or not in the allowed set.
    """
    valid = {k.strip() for k in settings.api_keys.split(",") if k.strip()}
    if not valid:
        return ""  # auth disabled, pass all requests
    if not api_key or api_key not in valid:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    return api_key
