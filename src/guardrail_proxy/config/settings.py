"""Application settings loaded from environment variables and .env file."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the Guardrail Proxy.

    All fields are overridable via environment variables or the .env file.
    Field names map directly to upper-cased env-var names
    (e.g. ``maliciousness_threshold`` → ``MALICIOUSNESS_THRESHOLD``).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── API ────────────────────────────────────────────────────────────────
    host: str = Field("127.0.0.1", description="Bind address for the proxy API.")
    port: int = Field(8010, description="Port for the proxy API (+10 offset).")

    # ── Postgres ───────────────────────────────────────────────────────────
    database_url: str = Field(
        "postgresql+psycopg://guardrail:guardrail@127.0.0.1:5442/guardrail",
        description="SQLAlchemy connection URL (Postgres on port 5442).",
    )

    # ── Redis ──────────────────────────────────────────────────────────────
    redis_url: str = Field(
        "redis://127.0.0.1:6389/0",
        description="Redis connection URL (port 6389).",
    )

    # ── Classifier ─────────────────────────────────────────────────────────
    maliciousness_model_path: str = Field(
        "artifacts/distilbert_guardrail",
        description=(
            "Path to a fine-tuned DistilBERT model directory. "
            "Leave as default to use the heuristic fallback."
        ),
    )
    maliciousness_threshold: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Maliciousness score at or above which a prompt is blocked.",
    )
    use_heuristic_fallback: bool = Field(
        True,
        description=(
            "When True and no trained model is found, score prompts with "
            "the built-in rule-based heuristic instead of raising an error."
        ),
    )

    # ── PII guard ──────────────────────────────────────────────────────────
    pii_block_threshold: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="PII risk score at or above which a prompt is blocked.",
    )
    pii_sanitize_threshold: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="PII risk score at or above which a prompt is sanitised.",
    )

    # ── Downstream LLM ─────────────────────────────────────────────────────
    downstream_url: str = Field(
        "http://127.0.0.1:11444/v1/chat/completions",
        description="Forwarding target for non-blocked prompts.",
    )
    # ── Auth ───────────────────────────────────────────────────────────────────
    api_keys: str = Field(
        "",
        description=(
            "Comma-separated API keys accepted in the X-API-Key header. "
            "Empty string disables authentication (default: disabled)."
        ),
    )

    # ── Rate limiting ──────────────────────────────────────────────────────────
    rate_limit_requests: int = Field(
        100,
        ge=0,
        description=(
            "Maximum requests per sliding window per identity (API key or IP). "
            "0 disables rate limiting."
        ),
    )
    rate_limit_window_seconds: float = Field(
        1.0,
        gt=0.0,
        description="Sliding window duration in seconds for rate limiting.",
    )