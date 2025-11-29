"""add trip_id to translations

Revision ID: 20251114_1530_add_trip_id_to_translations
Revises: 20251114_1520_create_trips
Create Date: 2025-11-14 15:30:00
"""
from alembic import op
import sqlalchemy as sa

revision = '20251114_1530_add_trip_id_to_translations'
down_revision = '20251114_1520_create_trips'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('translations', sa.Column('trip_id', sa.Integer(), sa.ForeignKey('trips.id'), nullable=True, index=True))

def downgrade() -> None:
    op.drop_column('translations', 'trip_id')
