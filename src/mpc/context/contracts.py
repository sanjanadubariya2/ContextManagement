"""Contracts with version history. A new contract version is a Context Repo
change, so it bumps the repo version like an approval does."""

import redis
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mpc.context.cache import publish_version
from mpc.context.repo import bump_version, on_commit
from mpc.models import Contract


def latest(session: Session, path: str) -> Contract | None:
    return session.scalar(
        select(Contract).where(Contract.path == path).order_by(Contract.version.desc()).limit(1)
    )


def latest_all(session: Session) -> dict[str, Contract]:
    sub = (
        select(Contract.path, func.max(Contract.version).label("v"))
        .group_by(Contract.path)
        .subquery()
    )
    q = select(Contract).join(sub, (Contract.path == sub.c.path) & (Contract.version == sub.c.v))
    return {c.path: c for c in session.scalars(q)}


def publish(
    session: Session,
    path: str,
    content: str,
    *,
    created_by: str,
    linked_keys: list[str] | None = None,
    keywords: list[str] | None = None,
    r: redis.Redis | None = None,
    bump: bool = True,
) -> Contract:
    prev = latest(session, path)
    version = bump_version(session, f"contract {path}", [f"contract:{path}"]) if bump else None
    c = Contract(
        path=path,
        version=(prev.version + 1) if prev else 1,
        content=content,
        linked_keys=linked_keys if linked_keys is not None else (prev.linked_keys if prev else []),
        keywords=keywords if keywords is not None else (prev.keywords if prev else []),
        created_by=created_by,
        ctx_version=version,
    )
    session.add(c)
    session.flush()
    if r is not None and version is not None:
        on_commit(session, lambda: publish_version(r, version, [f"contract:{path}"]))
    return c
