"""Versioned artifact store in Postgres.

Instances write draft versions; promotion makes a draft the current version
of its path, supersedes the previous one, and re-indexes the file.
"""

import redis
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from mpc.context import cache
from mpc.context.indexer import IndexStats, index_artifact, team_for_path
from mpc.context.parsing import language_for
from mpc.context.repo import on_commit
from mpc.llm.base import Embedder
from mpc.models import Artifact


ARTIFACT_GEN = "artifact_gen"


class ArtifactError(Exception):
    pass


def _lock_path(session: Session, path: str) -> None:
    session.execute(text("SELECT pg_advisory_xact_lock(hashtext(:p))"), {"p": path})


def write_draft(
    session: Session, path: str, content: str, *, created_by: str, instance_id: str | None = None
) -> Artifact:
    _lock_path(session, path)
    last = session.scalar(
        select(func.coalesce(func.max(Artifact.version), 0)).where(Artifact.path == path)
    )
    art = Artifact(
        path=path,
        version=(last or 0) + 1,
        status="draft",
        content=content,
        language=language_for(path),
        team_id=team_for_path(path),
        created_by=created_by,
        instance_id=instance_id,
    )
    session.add(art)
    session.flush()
    return art


def promote(
    session: Session,
    artifact_id: int,
    embedder: Embedder,
    known_paths: set[str] | None = None,
    r: redis.Redis | None = None,
) -> IndexStats:
    art = session.get(Artifact, artifact_id)
    if art is None or art.status != "draft":
        raise ArtifactError(f"artifact {artifact_id} is not a draft")
    _lock_path(session, art.path)
    prev = current(session, art.path)
    if prev is not None:
        prev.status = "superseded"
        session.flush()  # free the one-current-per-path slot first
    art.status = "current"
    session.flush()
    stats = index_artifact(session, art, embedder, known_paths)
    if r is not None:
        path = art.path

        def _publish() -> None:
            r.incr(ARTIFACT_GEN)  # part of every package cache key
            cache.invalidate(r, [f"file:{path}"])

        on_commit(session, _publish)
    return stats


def current(session: Session, path: str) -> Artifact | None:
    return session.scalar(
        select(Artifact).where(Artifact.path == path, Artifact.status == "current")
    )


def current_versions(session: Session, paths: list[str] | None = None) -> dict[str, int]:
    q = select(Artifact.path, Artifact.version).where(Artifact.status == "current")
    if paths is not None:
        q = q.where(Artifact.path.in_(paths))
    return {p: v for p, v in session.execute(q)}
