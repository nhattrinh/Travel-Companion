"""create trips table

Revision ID: 20251114_1520_create_trips
Revises: 20251114_1510_create_favorites
Create Date: 2025-11-14 15:20:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = '20251114_1520_create_trips'
down_revision = '20251114_1510_create_favorites'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'trips',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('destination', sa.String(255), nullable=False),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(32), nullable=False, server_default='active', index=True),
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

def downgrade() -> None:
    op.drop_table('trips')
