"""Allow predictions without calibrated confidence and retain confidence metadata.

Revision ID: 20261125_0002
Revises: 20261125_0001
"""
from alembic import op
import sqlalchemy as sa

revision = "20261125_0002"
down_revision = "20261125_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # An uncalibrated LinearSVC returns a decision margin, not a probability.
    op.alter_column(
        "predictions",
        "confidence",
        existing_type=sa.Numeric(6, 5),
        nullable=True,
    )
    op.add_column(
        "predictions",
        sa.Column(
            "confidence_type",
            sa.String(length=32),
            nullable=False,
            server_default="model_score",
        ),
    )
    op.add_column(
        "predictions",
        sa.Column("probability_calibrated", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "predictions",
        sa.Column("decision_margin", sa.Numeric(12, 6), nullable=True),
    )
    op.add_column(
        "predictions",
        sa.Column("review_reasons", sa.JSON(), nullable=True),
    )
    op.create_check_constraint(
        "ck_predictions_confidence_type",
        "predictions",
        "confidence_type IN ('model_score', 'calibrated_probability', 'unavailable')",
    )


def downgrade() -> None:
    # Preserve the old NOT NULL invariant when rolling back to the prior schema.
    op.execute("UPDATE predictions SET confidence = 0 WHERE confidence IS NULL")
    op.drop_constraint(
        "ck_predictions_confidence_type",
        "predictions",
        type_="check",
    )
    op.drop_column("predictions", "review_reasons")
    op.drop_column("predictions", "decision_margin")
    op.drop_column("predictions", "probability_calibrated")
    op.drop_column("predictions", "confidence_type")
    op.alter_column(
        "predictions",
        "confidence",
        existing_type=sa.Numeric(6, 5),
        nullable=False,
    )
