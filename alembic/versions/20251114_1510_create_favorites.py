"""create favorites table

Revision ID: 20251114_1510_create_favorites
Revises: 20251114_1500_create_phrases
Create Date: 2025-11-14 15:10:00
"""
from alembic import op
import sqlalchemy as sa

revision = '20251114_1510_create_favorites'
down_revision = '20251114_1500_create_phrases'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'favorites',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('target_type', sa.String(32), nullable=False),
        sa.Column('target_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

def downgrade() -> None:
    op.drop_table('favorites')
