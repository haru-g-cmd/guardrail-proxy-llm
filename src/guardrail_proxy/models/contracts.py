"""Pydantic request/response contracts for the guardrail proxy API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class GuardrailVerdict(str, Enum):
    """The three possible outcomes of a prompt analysis."""

    ALLOW = "allow"
    SANITIZE = "sanitize"
    BLOCK = "block"


class CheckRequest(BaseModel):
    """Payload for ``POST /v1/guardrail/check``."""

    prompt: str = Field(..., min_length=1, max_length=32_768)
    session_id: str | None = Field(None, description="Optional caller-supplied session tag.")


class PIIFinding(BaseModel):
    """Details of a single PII span detected in a prompt."""

    entity_type: str
    start: int
    end: int
    replacement: str


class CheckResponse(BaseModel):
    """Structured analysis result returned to the caller."""

    verdict: GuardrailVerdict
    sanitized_prompt: str | None = None
    maliciousness_score: float
    pii_risk_score: float = 0.0
    pii_findings: list[PIIFinding]
    reason: str | None = None
    cached: bool = False


class ProxyRequest(BaseModel):
    """Payload for ``POST /v1/guardrail/proxy``: check and optionally forward."""

    prompt: str = Field(..., min_length=1, max_length=32_768)
    session_id: str | None = None
    model: str = "gpt-3.5-turbo"


class ProxyResponse(BaseModel):
    """Combined guardrail + downstream result."""

    verdict: GuardrailVerdict
    llm_response: str | None = None
    sanitized_prompt: str | None = None
    maliciousness_score: float
    pii_findings: list[PIIFinding]
    reason: str | None = None
