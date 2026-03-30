"""SQLAlchemy engine and session factory for the audit database.

Connection uses psycopg3 (``psycopg`` package) in synchronous mode so that
FastAPI endpoints remain simple and no async session management is required.
``pool_pre_ping=True`` ensures stale connections are detected and recycled
automatically after a Docker restart.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from guardrail_proxy.config.settings import Settings


class Base(DeclarativeBase):
    """Shared declarative base for all ORM entities."""


def build_engine(settings: Settings) -> Engine:
    """
    Create a SQLAlchemy Engine for the configured Postgres URL.

    ``pool_pre_ping`` issues a lightweight ``SELECT 1`` before each checkout
    to avoid using a connection that has gone stale (e.g. after a container
    restart).
    """
    return create_engine(settings.database_url, pool_pre_ping=True)


def build_session_factory(settings: Settings) -> sessionmaker[Session]:
    """
    Create and return a :class:`~sqlalchemy.orm.sessionmaker` bound to the
    engine derived from *settings*.

    The returned factory is used as a context manager::

        with session_factory() as session:
            session.add(record)
            session.commit()
    """
    engine = build_engine(settings)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)
