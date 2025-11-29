"""create phrases table

Revision ID: 20251114_1500_create_phrases
Revises: 20251114_1400_create_pois
Create Date: 2025-11-14 15:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '20251114_1500_create_phrases'
down_revision = '20251114_1400_create_pois'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'phrases',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('canonical_text', sa.Text(), nullable=False),
        sa.Column('translations', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('phonetic', sa.Text(), nullable=True),
        sa.Column('context_category', sa.String(64), nullable=False, index=True),
    )

def downgrade() -> None:
    op.drop_table('phrases')
