"""create pois table

Revision ID: 20251114_1400_create_pois
Revises: 20251114_1300_create_translations
Create Date: 2025-11-14 14:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = '20251114_1400_create_pois'
down_revision = '20251114_1300_create_translations'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'pois',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('category', sa.String(64), nullable=False),
        sa.Column('latitude', sa.Float(), nullable=False, index=True),
        sa.Column('longitude', sa.Float(), nullable=False, index=True),
        sa.Column('etiquette_notes', sa.Text(), nullable=True),
    )

def downgrade() -> None:
    op.drop_table('pois')
