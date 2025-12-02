"""create phrase_suggestions table

Revision ID: 20251114_1540_create_phrase_suggestions
Revises: 20251114_1530_add_trip_id_to_translations
Create Date: 2025-11-14 15:40:00
"""
from alembic import op
import sqlalchemy as sa

revision = "20251114_1540_create_phrase_suggestions"
down_revision = "20251114_1530_add_trip_id_to_translations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "phrase_suggestions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("phrase_id", sa.Integer(), nullable=False, index=True),
        sa.Column("user_id", sa.Integer(), nullable=True, index=True),
        sa.Column("context", sa.String(length=64), nullable=True),
        sa.Column("relevance", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_phrase_suggestions_phrase_id",
        "phrase_suggestions",
        ["phrase_id"],
        unique=False,
    )
    op.create_index(
        "ix_phrase_suggestions_user_id",
        "phrase_suggestions",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_phrase_suggestions_user_id", table_name="phrase_suggestions"
    )
    op.drop_index(
        "ix_phrase_suggestions_phrase_id", table_name="phrase_suggestions"
    )
    op.drop_table("phrase_suggestions")
