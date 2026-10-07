"""Context Repo versions, the workspace turn counter, and post-commit hooks."""

from typing import Callable

from sqlalchemy import event, func, select, text
from sqlalchemy.orm import Session

from mpc.models import ContextRepoVersion

_VERSION_LOCK = 4242  # advisory lock id serialising version bumps


def current_version(session: Session) -> int:
    return session.scalar(select(func.coalesce(func.max(ContextRepoVersion.version), 0))) or 0


def bump_version(session: Session, reason: str, changed_keys: list[str]) -> int:
    """Create the next Context Repo version inside the caller's transaction."""
    session.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _VERSION_LOCK})
    version = current_version(session) + 1
    session.add(ContextRepoVersion(version=version, reason=reason, changed_keys=changed_keys))
    session.flush()
    return version


def current_turn(session: Session) -> int:
    """Global workspace turn: the number of messages ever written."""
    row = session.execute(text("SELECT last_value, is_called FROM workspace_turn_seq")).one()
    return int(row.last_value) if row.is_called else 0


# --------------------------------------------------------- commit hooks

_HOOKS = "mpc_after_commit"


def on_commit(session: Session, fn: Callable[[], None]) -> None:
    """Run fn after the session's transaction commits (dropped on rollback).

    Redis must only learn about versions that Postgres actually committed.
    """
    session.info.setdefault(_HOOKS, []).append(fn)


@event.listens_for(Session, "after_commit")
def _run_hooks(session: Session) -> None:
    for fn in session.info.pop(_HOOKS, []):
        fn()


@event.listens_for(Session, "after_rollback")
def _drop_hooks(session: Session) -> None:
    session.info.pop(_HOOKS, None)
