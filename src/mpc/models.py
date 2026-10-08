"""ORM models. Migrations in mpc/migrations are the source of truth for the
schema; these mappings mirror them.

Increment 1 tables: teams, members, agents, decisions, context_repo_versions,
contracts, tasks, artifacts, context_traces, messages.
Increment 2 tables: decision_keys, decision_approvals, chunks.
"""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from mpc.config import EMBED_DIM


class Base(DeclarativeBase):
    pass


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------- workspace


class Team(Base):
    __tablename__ = "teams"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # frontend | backend | testing
    name: Mapped[str] = mapped_column(String)
    write_root: Mapped[str] = mapped_column(String)  # frontend/ | backend/ | tests/


class Member(Base):
    __tablename__ = "members"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # sneha
    display_name: Mapped[str] = mapped_column(String)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.id"))
    seat: Mapped[int] = mapped_column(Integer)  # 1..3, fixed membership
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = _now()


class Agent(Base):
    __tablename__ = "agents"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # agent_be_sneha
    member_id: Mapped[str] = mapped_column(ForeignKey("members.id"), unique=True)
    kind: Mapped[str] = mapped_column(String, default="personal")
    created_at: Mapped[datetime] = _now()


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    member_id: Mapped[str] = mapped_column(ForeignKey("members.id"))
    agent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    channel: Mapped[str] = mapped_column(String)  # personal:<member> | team:<team>
    role: Mapped[str] = mapped_column(String)  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    # Global workspace turn, from workspace_turn_seq.
    turn: Mapped[int] = mapped_column(
        Integer, server_default=text("nextval('workspace_turn_seq')")
    )
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = _now()

    __mapper_args__ = {"eager_defaults": True}


# ------------------------------------------------------------ context repo


class ContextRepoVersion(Base):
    __tablename__ = "context_repo_versions"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    reason: Mapped[str] = mapped_column(Text)
    changed_keys: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    created_at: Mapped[datetime] = _now()


class DecisionKey(Base):
    __tablename__ = "decision_keys"
    key: Mapped[str] = mapped_column(String, primary_key=True)  # auth.token_transport
    value_type: Mapped[str] = mapped_column(String)  # enum | string | int | bool
    # enum: {value: [alias phrases]} used for impact analysis and contradictions
    allowed_values: Mapped[dict[str, list[str]] | None] = mapped_column(JSONB, nullable=True)
    owner_team: Mapped[str | None] = mapped_column(ForeignKey("teams.id"), nullable=True)
    keywords: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    is_global_spec: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str] = mapped_column(Text, default="")


class Decision(Base):
    __tablename__ = "decisions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(ForeignKey("decision_keys.key"))
    value: Mapped[Any] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String)  # proposal | approved | superseded | rejected
    team_id: Mapped[str | None] = mapped_column(ForeignKey("teams.id"), nullable=True)
    rationale: Mapped[str] = mapped_column(Text, default="")
    proposed_by: Mapped[str] = mapped_column(String)  # member id or instance id
    created_turn: Mapped[int] = mapped_column(Integer, default=0)
    approved_in_version: Mapped[int | None] = mapped_column(
        ForeignKey("context_repo_versions.version"), nullable=True
    )
    superseded_in_version: Mapped[int | None] = mapped_column(
        ForeignKey("context_repo_versions.version"), nullable=True
    )
    superseded_by: Mapped[int | None] = mapped_column(ForeignKey("decisions.id"), nullable=True)
    created_at: Mapped[datetime] = _now()
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DecisionApproval(Base):
    __tablename__ = "decision_approvals"
    decision_id: Mapped[int] = mapped_column(ForeignKey("decisions.id"), primary_key=True)
    member_id: Mapped[str] = mapped_column(ForeignKey("members.id"), primary_key=True)
    created_at: Mapped[datetime] = _now()


class Contract(Base):
    __tablename__ = "contracts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String)  # openapi/auth.yaml
    version: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    linked_keys: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    keywords: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    created_by: Mapped[str] = mapped_column(String)
    ctx_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = _now()


# ------------------------------------------------------------------ work


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # t21
    title: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text, default="")
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.id"))
    status: Mapped[str] = mapped_column(String, default="open")  # open | claimed | done
    claimed_by: Mapped[str | None] = mapped_column(ForeignKey("members.id"), nullable=True)
    files: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    decision_keys: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    contracts: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    created_by: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = _now()


class Artifact(Base):
    """One version of one file. Exactly one `current` version per path."""

    __tablename__ = "artifacts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String)
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String)  # draft | current | superseded | discarded
    content: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String, default="text")
    team_id: Mapped[str | None] = mapped_column(ForeignKey("teams.id"), nullable=True)
    imports: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)  # raw specifiers
    resolved_imports: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)  # paths
    summary: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String)
    instance_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = _now()


class Chunk(Base):
    """A retrievable unit in pgvector: code, rationale, summary, requirement..."""

    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String)
    source_ref: Mapped[str] = mapped_column(String)  # artifact:12 | decision:5 | message:40
    path: Mapped[str | None] = mapped_column(String, nullable=True)
    artifact_id: Mapped[int | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    decision_id: Mapped[int | None] = mapped_column(ForeignKey("decisions.id"), nullable=True)
    symbol: Mapped[str | None] = mapped_column(String, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    tokens: Mapped[int] = mapped_column(Integer)
    visibility: Mapped[str] = mapped_column(String)  # frontend | backend | testing | shared
    status: Mapped[str] = mapped_column(String, default="active")  # active | superseded
    decision_status: Mapped[str | None] = mapped_column(String, nullable=True)
    is_global_spec: Mapped[bool] = mapped_column(Boolean, default=False)
    author: Mapped[str] = mapped_column(String)
    created_turn: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    created_at: Mapped[datetime] = _now()


class ContextTrace(Base):
    __tablename__ = "context_traces"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    member_id: Mapped[str | None] = mapped_column(String, nullable=True)
    instance_id: Mapped[str | None] = mapped_column(String, nullable=True)
    task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    task_text: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String)
    ticket: Mapped[dict] = mapped_column(JSONB)
    ctx_version: Mapped[int] = mapped_column(Integer)
    token_budget: Mapped[int] = mapped_column(Integer)
    total_tokens: Mapped[int] = mapped_column(Integer)
    pinned_tokens: Mapped[int] = mapped_column(Integer)
    items: Mapped[list] = mapped_column(JSONB)
    dropped: Mapped[list] = mapped_column(JSONB)
    dependencies: Mapped[dict] = mapped_column(JSONB)
    package_hash: Mapped[str] = mapped_column(String)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = _now()


# ------------------------------------------------------------ agents (Increment 3)


class InstanceRun(Base):
    """One run of one subagent instance (a re-run after staleness is a new attempt)."""

    __tablename__ = "instance_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instance_id: Mapped[str] = mapped_column(String)  # worker:be-sneha:t20
    template: Mapped[str] = mapped_column(String)  # backend_worker
    mode: Mapped[str] = mapped_column(String)
    member_id: Mapped[str] = mapped_column(ForeignKey("members.id"))
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    ticket: Mapped[dict] = mapped_column(JSONB)
    instructions: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String)  # running | completed | stale | invalid | failed
    trace_id: Mapped[int | None] = mapped_column(ForeignKey("context_traces.id"), nullable=True)
    ctx_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    files: Mapped[dict] = mapped_column(JSONB, default=dict)  # path -> promoted version
    proposals: Mapped[list[int]] = mapped_column(ARRAY(Integer), default=list)
    steps: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("instance_runs.id"))
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), nullable=True)
    requested_by: Mapped[str] = mapped_column(ForeignKey("members.id"))
    paths: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    verdict: Mapped[str] = mapped_column(String)  # approve | changes_requested | incomplete
    summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = _now()


class ReviewComment(Base):
    __tablename__ = "review_comments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_id: Mapped[int] = mapped_column(ForeignKey("reviews.id"))
    path: Mapped[str | None] = mapped_column(String, nullable=True)
    line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    body: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String)  # reviewer | precheck
    created_at: Mapped[datetime] = _now()


class TestRun(Base):
    __tablename__ = "test_runs"
    __test__ = False  # not a pytest class
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("instance_runs.id"), nullable=True)
    member_id: Mapped[str | None] = mapped_column(ForeignKey("members.id"), nullable=True)
    paths: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    passed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[int] = mapped_column(Integer, default=0)
    timed_out: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    output: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = _now()
