"""Decision registry: typed keys and the proposal -> approved -> superseded
lifecycle. Each approval bumps the Context Repo version and, after commit,
publishes the version to Redis and invalidates cached packages that depend on
the changed key.

Who may approve, and how many approvals are needed (2 of 3, Conflict Cards,
votes), is governance and arrives in Increment 4. `approve` here is the
primitive governance will call once a decision has the votes it needs.
"""

from datetime import datetime, timezone
from typing import Any

import redis
from sqlalchemy import select
from sqlalchemy.orm import Session

from mpc.context.cache import publish_version
from mpc.context.indexer import index_decision, retire_decision
from mpc.context.repo import bump_version, current_turn, on_commit
from mpc.llm.base import Embedder
from mpc.models import Decision, DecisionApproval, DecisionKey

VALUE_TYPES = {"enum", "string", "int", "bool"}


class RegistryError(Exception):
    pass


def register_key(
    session: Session,
    key: str,
    value_type: str,
    *,
    allowed_values: dict[str, list[str]] | None = None,
    owner_team: str | None = None,
    keywords: list[str] | None = None,
    is_global_spec: bool = False,
    description: str = "",
) -> DecisionKey:
    if value_type not in VALUE_TYPES:
        raise RegistryError(f"unknown value type {value_type!r}")
    if value_type == "enum" and not allowed_values:
        raise RegistryError(f"enum key {key} needs allowed_values")
    dk = session.get(DecisionKey, key) or DecisionKey(key=key)
    dk.value_type = value_type
    dk.allowed_values = allowed_values
    dk.owner_team = owner_team
    dk.keywords = [str(k) for k in (keywords or [])]
    dk.is_global_spec = is_global_spec
    dk.description = description
    session.add(dk)
    session.flush()
    return dk


def validate_value(dk: DecisionKey, value: Any) -> Any:
    t = dk.value_type
    if t == "enum":
        if value not in (dk.allowed_values or {}):
            allowed = ", ".join(dk.allowed_values or {})
            raise RegistryError(f"{dk.key}: {value!r} is not one of: {allowed}")
        return value
    if t == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            raise RegistryError(f"{dk.key}: {value!r} is not an integer") from None
    if t == "bool":
        if isinstance(value, bool):
            return value
        if str(value).lower() in ("true", "yes", "1"):
            return True
        if str(value).lower() in ("false", "no", "0"):
            return False
        raise RegistryError(f"{dk.key}: {value!r} is not a boolean")
    return str(value)


def propose(
    session: Session,
    key: str,
    value: Any,
    *,
    proposed_by: str,
    embedder: Embedder,
    team_id: str | None = None,
    rationale: str = "",
) -> Decision:
    """Agents, Memorizer instances and members may only create proposals."""
    dk = session.get(DecisionKey, key)
    if dk is None:
        raise RegistryError(f"unknown decision key {key!r}")
    d = Decision(
        key=key,
        value=validate_value(dk, value),
        status="proposal",
        team_id=team_id,
        rationale=rationale,
        proposed_by=proposed_by,
        created_turn=current_turn(session),
    )
    session.add(d)
    session.flush()
    index_decision(session, d, embedder)
    return d


def approve(
    session: Session,
    decision_id: int,
    *,
    approvers: list[str],
    embedder: Embedder,
    r: redis.Redis | None = None,
) -> int:
    """Approve a proposal; supersede the key's previous approved value.

    Returns the new Context Repo version.
    """
    d = session.get(Decision, decision_id, with_for_update=True)
    if d is None:
        raise RegistryError(f"no decision {decision_id}")
    if d.status != "proposal":
        raise RegistryError(f"decision {decision_id} is {d.status}, not a proposal")

    version = bump_version(session, f"approve {d.key} = {d.value!r}", [d.key])
    prev = get_approved(session, d.key)
    if prev is not None:
        prev.status = "superseded"
        prev.superseded_in_version = version
        prev.superseded_by = d.id
        retire_decision(session, prev.id)
        session.flush()  # free the one-approved-per-key slot first

    d.status = "approved"
    d.approved_in_version = version
    d.approved_at = datetime.now(timezone.utc)
    for m in dict.fromkeys(approvers):
        if session.get(DecisionApproval, (d.id, m)) is None:
            session.add(DecisionApproval(decision_id=d.id, member_id=m))
    session.flush()
    index_decision(session, d, embedder)

    if r is not None:
        on_commit(session, lambda: publish_version(r, version, [f"key:{d.key}"]))
    return version


def reject(session: Session, decision_id: int) -> Decision:
    d = session.get(Decision, decision_id)
    if d is None or d.status != "proposal":
        raise RegistryError(f"decision {decision_id} is not an open proposal")
    d.status = "rejected"
    retire_decision(session, d.id)
    session.flush()
    return d


def get_approved(session: Session, key: str) -> Decision | None:
    return session.scalar(select(Decision).where(Decision.key == key, Decision.status == "approved"))


def approved_map(session: Session, keys: list[str] | None = None) -> dict[str, Decision]:
    q = select(Decision).where(Decision.status == "approved")
    if keys is not None:
        q = q.where(Decision.key.in_(keys))
    return {d.key: d for d in session.scalars(q.order_by(Decision.key))}


def proposals(session: Session, key: str | None = None) -> list[Decision]:
    q = select(Decision).where(Decision.status == "proposal")
    if key:
        q = q.where(Decision.key == key)
    return list(session.scalars(q.order_by(Decision.id)))


def history(session: Session, key: str) -> list[Decision]:
    return list(session.scalars(select(Decision).where(Decision.key == key).order_by(Decision.id)))


def snapshot(session: Session, version: int) -> dict[str, Any]:
    """Approved values as of a Context Repo version (time travel)."""
    q = select(Decision).where(
        Decision.approved_in_version.is_not(None),
        Decision.approved_in_version <= version,
        (Decision.superseded_in_version.is_(None)) | (Decision.superseded_in_version > version),
    )
    return {d.key: d.value for d in session.scalars(q)}
