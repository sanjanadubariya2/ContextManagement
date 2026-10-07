"""Increment 2: typed decision keys, approvals, and the pgvector chunk store.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

EMBED_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "decision_keys",
        sa.Column("key", sa.String, primary_key=True),
        sa.Column("value_type", sa.String, nullable=False),
        sa.Column("allowed_values", JSONB, nullable=True),
        sa.Column("owner_team", sa.String, sa.ForeignKey("teams.id"), nullable=True),
        sa.Column("keywords", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("is_global_spec", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.CheckConstraint(
            "value_type in ('enum','string','int','bool')", name="ck_decision_key_type"
        ),
    )
    op.create_foreign_key("fk_decision_key", "decisions", "decision_keys", ["key"], ["key"])
    op.create_table(
        "decision_approvals",
        sa.Column("decision_id", sa.Integer, sa.ForeignKey("decisions.id"), primary_key=True),
        sa.Column("member_id", sa.String, sa.ForeignKey("members.id"), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("kind", sa.String, nullable=False),
        sa.Column("source_ref", sa.String, nullable=False),
        sa.Column("path", sa.String, nullable=True),
        sa.Column("artifact_id", sa.Integer, sa.ForeignKey("artifacts.id"), nullable=True),
        sa.Column("decision_id", sa.Integer, sa.ForeignKey("decisions.id"), nullable=True),
        sa.Column("symbol", sa.String, nullable=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("tokens", sa.Integer, nullable=False),
        sa.Column("visibility", sa.String, nullable=False),
        sa.Column("status", sa.String, nullable=False, server_default="active"),
        sa.Column("decision_status", sa.String, nullable=True),
        sa.Column("is_global_spec", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("author", sa.String, nullable=False),
        sa.Column("created_turn", sa.Integer, nullable=False, server_default="0"),
        sa.Column("embedding", Vector(EMBED_DIM), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("status in ('active','superseded')", name="ck_chunk_status"),
        sa.CheckConstraint(
            "visibility in ('frontend','backend','testing','shared')", name="ck_chunk_visibility"
        ),
    )
    op.create_index("ix_chunks_filter", "chunks", ["status", "visibility", "kind"])
    op.create_index("ix_chunks_path", "chunks", ["path"])
    op.execute(
        "CREATE INDEX ix_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("decision_approvals")
    op.drop_constraint("fk_decision_key", "decisions", type_="foreignkey")
    op.drop_table("decision_keys")
