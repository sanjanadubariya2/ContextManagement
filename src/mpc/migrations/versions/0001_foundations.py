"""Increment 1: foundations schema.

teams, members, agents, decisions, context_repo_versions, contracts, tasks,
artifacts, context_traces, messages.

Revision ID: 0001
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _ts():
    return sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now())


def upgrade() -> None:
    op.create_table(
        "teams",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("write_root", sa.String, nullable=False),
    )
    op.create_table(
        "members",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("display_name", sa.String, nullable=False),
        sa.Column("team_id", sa.String, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("seat", sa.Integer, nullable=False),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        _ts(),
        sa.UniqueConstraint("team_id", "seat", name="uq_member_seat"),
        sa.CheckConstraint("seat between 1 and 3", name="ck_member_seat"),
    )
    op.create_table(
        "agents",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("member_id", sa.String, sa.ForeignKey("members.id"), nullable=False, unique=True),
        sa.Column("kind", sa.String, nullable=False, server_default="personal"),
        _ts(),
    )
    op.create_table(
        "context_repo_versions",
        sa.Column("version", sa.Integer, primary_key=True),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("changed_keys", ARRAY(sa.String), nullable=False, server_default="{}"),
        _ts(),
    )
    op.create_table(
        "decisions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("key", sa.String, nullable=False, index=True),
        sa.Column("value", JSONB, nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("team_id", sa.String, sa.ForeignKey("teams.id"), nullable=True),
        sa.Column("rationale", sa.Text, nullable=False, server_default=""),
        sa.Column("proposed_by", sa.String, nullable=False),
        sa.Column("created_turn", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "approved_in_version",
            sa.Integer,
            sa.ForeignKey("context_repo_versions.version"),
            nullable=True,
        ),
        sa.Column(
            "superseded_in_version",
            sa.Integer,
            sa.ForeignKey("context_repo_versions.version"),
            nullable=True,
        ),
        sa.Column("superseded_by", sa.Integer, sa.ForeignKey("decisions.id"), nullable=True),
        _ts(),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status in ('proposal','approved','superseded','rejected')", name="ck_decision_status"
        ),
    )
    # At most one approved decision per key.
    op.create_index(
        "uq_decision_approved_key",
        "decisions",
        ["key"],
        unique=True,
        postgresql_where=sa.text("status = 'approved'"),
    )
    op.create_table(
        "contracts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("path", sa.String, nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("linked_keys", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("keywords", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("created_by", sa.String, nullable=False),
        sa.Column("ctx_version", sa.Integer, nullable=True),
        _ts(),
        sa.UniqueConstraint("path", "version", name="uq_contract_version"),
    )
    op.create_table(
        "tasks",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("title", sa.String, nullable=False),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("team_id", sa.String, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("status", sa.String, nullable=False, server_default="open"),
        sa.Column("claimed_by", sa.String, sa.ForeignKey("members.id"), nullable=True),
        sa.Column("files", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("decision_keys", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("contracts", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("created_by", sa.String, nullable=False),
        _ts(),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("status in ('open','claimed','done')", name="ck_task_status"),
    )
    op.create_table(
        "artifacts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("path", sa.String, nullable=False, index=True),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("language", sa.String, nullable=False, server_default="text"),
        sa.Column("team_id", sa.String, sa.ForeignKey("teams.id"), nullable=True),
        sa.Column("imports", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("resolved_imports", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("summary", sa.Text, nullable=False, server_default=""),
        sa.Column("created_by", sa.String, nullable=False),
        sa.Column("instance_id", sa.String, nullable=True),
        _ts(),
        sa.UniqueConstraint("path", "version", name="uq_artifact_version"),
        sa.CheckConstraint(
            "status in ('draft','current','superseded')", name="ck_artifact_status"
        ),
    )
    op.create_index(
        "uq_artifact_current_path",
        "artifacts",
        ["path"],
        unique=True,
        postgresql_where=sa.text("status = 'current'"),
    )
    op.create_table(
        "context_traces",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("member_id", sa.String, nullable=True),
        sa.Column("instance_id", sa.String, nullable=True),
        sa.Column("task_id", sa.String, nullable=True),
        sa.Column("task_text", sa.Text, nullable=False),
        sa.Column("mode", sa.String, nullable=False),
        sa.Column("ticket", JSONB, nullable=False),
        sa.Column("ctx_version", sa.Integer, nullable=False),
        sa.Column("token_budget", sa.Integer, nullable=False),
        sa.Column("total_tokens", sa.Integer, nullable=False),
        sa.Column("pinned_tokens", sa.Integer, nullable=False),
        sa.Column("items", JSONB, nullable=False),
        sa.Column("dropped", JSONB, nullable=False),
        sa.Column("dependencies", JSONB, nullable=False),
        sa.Column("package_hash", sa.String, nullable=False),
        sa.Column("cache_hit", sa.Boolean, nullable=False, server_default=sa.false()),
        _ts(),
    )
    op.execute("CREATE SEQUENCE workspace_turn_seq START 1")
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("member_id", sa.String, sa.ForeignKey("members.id"), nullable=False),
        sa.Column("agent_id", sa.String, nullable=True),
        sa.Column("channel", sa.String, nullable=False),
        sa.Column("role", sa.String, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column(
            "turn",
            sa.Integer,
            nullable=False,
            server_default=sa.text("nextval('workspace_turn_seq')"),
        ),
        sa.Column("tokens", sa.Integer, nullable=False, server_default="0"),
        _ts(),
    )
    op.create_index("ix_messages_member", "messages", ["member_id", "turn"])


def downgrade() -> None:
    op.drop_table("messages")
    op.execute("DROP SEQUENCE IF EXISTS workspace_turn_seq")
    op.drop_table("context_traces")
    op.drop_table("artifacts")
    op.drop_table("tasks")
    op.drop_table("contracts")
    op.drop_table("decisions")
    op.drop_table("context_repo_versions")
    op.drop_table("agents")
    op.drop_table("members")
    op.drop_table("teams")
