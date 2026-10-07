"""Postgres engine and session helpers."""

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from mpc.config import get_settings


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or get_settings().database_url, pool_pre_ping=True, future=True)


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """A transaction: commit on success, roll back on error."""
    session = sessionmaker(bind=get_engine(url), expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping(url: str | None = None) -> bool:
    try:
        with get_engine(url).connect() as conn:
            conn.execute(text("select 1"))
        return True
    except Exception:
        return False
