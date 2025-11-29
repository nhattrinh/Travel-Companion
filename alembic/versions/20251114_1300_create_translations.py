"""create translations table

Revision ID: 20251114_1300_create_translations
Revises: 20251114_1200_create_users
Create Date: 2025-11-14 13:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = '20251114_1300_create_translations'
down_revision = '20251114_1200_create_users'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'translations',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=True, index=True),
        sa.Column('source_text', sa.Text(), nullable=False),
        sa.Column('target_text', sa.Text(), nullable=False),
        sa.Column('source_language', sa.String(8), nullable=False),
        sa.Column('target_language', sa.String(8), nullable=False),
        sa.Column('confidence', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_translations_user_id', 'translations', ['user_id'])

def downgrade() -> None:
    op.drop_index('ix_translations_user_id', table_name='translations')
    op.drop_table('translations')
