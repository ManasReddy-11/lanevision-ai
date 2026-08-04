from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260804_initial_detection_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "detection_runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("media_type", sa.String(), nullable=False),
        sa.Column("algorithm", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("original_filename", sa.String(), nullable=False),
        sa.Column("original_media_path", sa.String(), nullable=False),
        sa.Column("processed_media_path", sa.String(), nullable=False),
        sa.Column("processing_time_ms", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("left_lane_detected", sa.Boolean(), nullable=True),
        sa.Column("right_lane_detected", sa.Boolean(), nullable=True),
        sa.Column("video_fps", sa.Float(), nullable=True),
        sa.Column("detection_success_rate", sa.Float(), nullable=True),
        sa.Column("both_lane_detection_rate", sa.Float(), nullable=True),
        sa.Column("metrics_json", sa.JSON(), nullable=True),
        sa.Column("warnings_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_detection_runs_created_at"), "detection_runs", ["created_at"], unique=False)
    op.create_index(op.f("ix_detection_runs_media_type"), "detection_runs", ["media_type"], unique=False)
    op.create_index(op.f("ix_detection_runs_algorithm"), "detection_runs", ["algorithm"], unique=False)
    op.create_index(op.f("ix_detection_runs_status"), "detection_runs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_detection_runs_status"), table_name="detection_runs")
    op.drop_index(op.f("ix_detection_runs_algorithm"), table_name="detection_runs")
    op.drop_index(op.f("ix_detection_runs_media_type"), table_name="detection_runs")
    op.drop_index(op.f("ix_detection_runs_created_at"), table_name="detection_runs")
    op.drop_table("detection_runs")
