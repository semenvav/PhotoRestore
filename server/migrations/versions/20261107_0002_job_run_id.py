"""job run_id

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = '0002'
down_revision: str | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('job', schema=None) as batch_op:
        batch_op.add_column(sa.Column('run_id', sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f('ix_job_run_id'), ['run_id'], unique=False)
        batch_op.create_foreign_key(batch_op.f('fk_job_run_id_run'), 'run', ['run_id'], ['id'])


def downgrade() -> None:
    with op.batch_alter_table('job', schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f('fk_job_run_id_run'), type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_job_run_id'))
        batch_op.drop_column('run_id')
