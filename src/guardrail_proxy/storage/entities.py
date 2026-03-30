"""SQLAlchemy ORM entity for persistent prompt audit records.

The ``audit_records`` table stores one row per prompt assessment.
Prompts are stored as SHA-256 hashes only, never as plain text, so that
the audit log cannot be used as a source of raw PII.
"""

from __future__ import annotations

import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from guardrail_proxy.storage.database import Base


class AuditRecord(Base):
    """One row per prompt assessment written by :class:`~guardrail_proxy.core.service.GuardrailService`."""

    __tablename__ = "audit_records"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    session_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True,
        comment="Caller-supplied session tag for grouping related requests.",
    )
    prompt_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True,
        comment="SHA-256 hex digest of the original prompt (no raw text stored).",
    )
    verdict: Mapped[str] = mapped_column(
        String(16), nullable=False,
        comment="One of: allow, sanitize, block.",
    )
    maliciousness_score: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Classifier score (0.0 – 1.0).",
    )
    pii_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="Number of PII spans detected.",
    )
    reason: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Human-readable explanation of the verdict.",
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
