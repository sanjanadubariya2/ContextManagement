"""Increment 3: subagent runs, reviews, test runs; discarded drafts.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def _ts(name="created_at"):
    return sa.Column(name, sa.DateTime(timezone=True), server_default=sa.func.now())


def upgrade() -> None:
    op.drop_constraint("ck_artifact_status", "artifacts", type_="check")
    op.create_check_constraint(
        "ck_artifact_status", "artifacts", "status in ('draft','current','superseded','discarded')"
    )

    op.create_table(
        "instance_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("instance_id", sa.String, nullable=False, index=True),
        sa.Column("template", sa.String, nullable=False),
        sa.Column("mode", sa.String, nullable=False),
        sa.Column("member_id", sa.String, sa.ForeignKey("members.id"), nullable=False),
        sa.Column("task_id", sa.String, sa.ForeignKey("tasks.id"), nullable=True),
        sa.Column("attempt", sa.Integer, nullable=False, server_default="1"),
        sa.Column("ticket", JSONB, nullable=False),
        sa.Column("instructions", sa.Text, nullable=False, server_default=""),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("trace_id", sa.Integer, sa.ForeignKey("context_traces.id"), nullable=True),
        sa.Column("ctx_version", sa.Integer, nullable=True),
        sa.Column("summary", sa.Text, nullable=False, server_default=""),
        sa.Column("files", JSONB, nullable=False, server_default="{}"),
        sa.Column("proposals", ARRAY(sa.Integer), nullable=False, server_default="{}"),
        sa.Column("steps", sa.Integer, nullable=False, server_default="0"),
        sa.Column("input_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error", sa.Text, nullable=True),
        _ts("started_at"),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status in ('running','completed','stale','invalid','failed')", name="ck_run_status"
        ),
    )
    op.create_table(
        "reviews",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("instance_runs.id"), nullable=False),
        sa.Column("task_id", sa.String, sa.ForeignKey("tasks.id"), nullable=True),
        sa.Column("requested_by", sa.String, sa.ForeignKey("members.id"), nullable=False),
        sa.Column("paths", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("verdict", sa.String, nullable=False),
        sa.Column("summary", sa.Text, nullable=False, server_default=""),
        _ts(),
        sa.CheckConstraint(
            "verdict in ('approve','changes_requested','incomplete')", name="ck_review_verdict"
        ),
    )
    op.create_table(
        "review_comments",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("review_id", sa.Integer, sa.ForeignKey("reviews.id"), nullable=False),
        sa.Column("path", sa.String, nullable=True),
        sa.Column("line", sa.Integer, nullable=True),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("source", sa.String, nullable=False),  # reviewer | precheck
        _ts(),
    )
    op.create_table(
        "test_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("instance_runs.id"), nullable=True),
        sa.Column("member_id", sa.String, sa.ForeignKey("members.id"), nullable=True),
        sa.Column("paths", ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("exit_code", sa.Integer, nullable=True),
        sa.Column("passed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("failed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("errors", sa.Integer, nullable=False, server_default="0"),
        sa.Column("timed_out", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("duration_ms", sa.Integer, nullable=False, server_default="0"),
        sa.Column("output", sa.Text, nullable=False, server_default=""),
        _ts(),
    )


def downgrade() -> None:
    op.drop_table("test_runs")
    op.drop_table("review_comments")
    op.drop_table("reviews")
    op.drop_table("instance_runs")
    op.drop_constraint("ck_artifact_status", "artifacts", type_="check")
    op.create_check_constraint(
        "ck_artifact_status", "artifacts", "status in ('draft','current','superseded')"
    )
