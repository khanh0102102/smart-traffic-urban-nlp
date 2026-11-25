"""Create the initial 12-table ST-NLP schema.
Revision ID: 20261125_0001
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "20261125_0001"
down_revision = None
branch_labels = None
depends_on = None

UUID = sa.Uuid()
TS = sa.DateTime(timezone=True)
def audit_columns():
    return [
        sa.Column("created_at", TS, server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", TS, server_default=sa.func.now(), nullable=False),
    ]

def upgrade():
    op.create_table("users",
        sa.Column("id", UUID, primary_key=True), sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("full_name", sa.String(160), nullable=False), sa.Column("role", sa.String(20), nullable=False, server_default="CITIZEN"),
        sa.Column("google_sub", sa.String(255), unique=True), sa.Column("password_hash", sa.String(255)),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()), *audit_columns(),
        sa.CheckConstraint("role IN ('ADMIN','RECEIVER','CITIZEN')", name="ck_users_role"))
    op.create_index("ix_users_role_active", "users", ["role","is_active"])

    op.create_table("incident_categories",
        sa.Column("id", UUID, primary_key=True), sa.Column("code", sa.String(3), nullable=False, unique=True),
        sa.Column("name", sa.String(160), nullable=False, unique=True), sa.Column("slug", sa.String(180), nullable=False, unique=True),
        sa.Column("description", sa.Text()), sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()), *audit_columns(), sa.UniqueConstraint("sort_order", name="uq_incident_categories_sort_order"))

    op.create_table("model_versions",
        sa.Column("id", UUID, primary_key=True), sa.Column("name", sa.String(120), nullable=False),
        sa.Column("version", sa.String(60), nullable=False), sa.Column("algorithm", sa.String(30), nullable=False),
        sa.Column("representation", sa.String(30), nullable=False), sa.Column("artifact_path", sa.String(1000)),
        sa.Column("parameters", sa.JSON()), sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        *audit_columns(), sa.UniqueConstraint("name","version",name="uq_model_versions_name_version"),
        sa.CheckConstraint("algorithm IN ('NAIVE_BAYES','SVM','CNN','LSTM','OTHER')",name="ck_model_versions_algorithm"),
        sa.CheckConstraint("representation IN ('TF_IDF','WORD_EMBEDDING','OTHER')",name="ck_model_versions_representation"))

    op.create_table("datasets",
        sa.Column("id", UUID, primary_key=True), sa.Column("name", sa.String(160), nullable=False),
        sa.Column("version", sa.String(60), nullable=False), sa.Column("description", sa.Text()),
        sa.Column("source", sa.String(255)), sa.Column("created_by", UUID, sa.ForeignKey("users.id",ondelete="SET NULL")),
        *audit_columns(), sa.UniqueConstraint("name","version",name="uq_datasets_name_version"))

    op.create_table("reports",
        sa.Column("id", UUID, primary_key=True), sa.Column("tracking_code", sa.String(32), nullable=False, unique=True),
        sa.Column("user_id", UUID, sa.ForeignKey("users.id",ondelete="SET NULL")), sa.Column("raw_content", sa.Text(), nullable=False),
        sa.Column("address", sa.String(500)), sa.Column("latitude", sa.Numeric(10,7)), sa.Column("longitude", sa.Numeric(10,7)),
        sa.Column("final_category_id", UUID, sa.ForeignKey("incident_categories.id",ondelete="SET NULL")),
        sa.Column("status", sa.String(24), nullable=False, server_default="RECEIVED"),
        sa.Column("submitted_at", TS, server_default=sa.func.now(), nullable=False), *audit_columns(),
        sa.CheckConstraint("latitude IS NULL OR latitude BETWEEN -90 AND 90",name="ck_reports_latitude"),
        sa.CheckConstraint("longitude IS NULL OR longitude BETWEEN -180 AND 180",name="ck_reports_longitude"))
    op.create_index("ix_reports_submitted_at","reports",["submitted_at"])
    op.create_index("ix_reports_final_category","reports",["final_category_id"])

    op.create_table("report_attachments",
        sa.Column("id", UUID, primary_key=True), sa.Column("report_id", UUID, sa.ForeignKey("reports.id",ondelete="CASCADE"),nullable=False),
        sa.Column("original_filename",sa.String(255),nullable=False),sa.Column("storage_path",sa.String(1000),nullable=False),
        sa.Column("mime_type",sa.String(120),nullable=False),sa.Column("file_size",sa.Integer(),nullable=False),*audit_columns())
    op.create_index("ix_report_attachments_report_id","report_attachments",["report_id"])

    op.create_table("preprocessing_runs",
        sa.Column("id",UUID,primary_key=True),sa.Column("report_id",UUID,sa.ForeignKey("reports.id",ondelete="CASCADE"),nullable=False),
        sa.Column("pipeline_version",sa.String(80),nullable=False),sa.Column("cleaned_text",sa.Text(),nullable=False),
        sa.Column("normalized_text",sa.Text(),nullable=False),sa.Column("segmented_text",sa.Text(),nullable=False),
        sa.Column("stopwords_removed_text",sa.Text(),nullable=False),sa.Column("processing_time_ms",sa.Integer()),*audit_columns())
    op.create_index("ix_preprocessing_runs_report_created","preprocessing_runs",["report_id","created_at"])

    op.create_table("predictions",
        sa.Column("id",UUID,primary_key=True),sa.Column("report_id",UUID,sa.ForeignKey("reports.id",ondelete="CASCADE"),nullable=False),
        sa.Column("preprocessing_run_id",UUID,sa.ForeignKey("preprocessing_runs.id",ondelete="SET NULL")),
        sa.Column("model_version_id",UUID,sa.ForeignKey("model_versions.id",ondelete="RESTRICT"),nullable=False),
        sa.Column("predicted_category_id",UUID,sa.ForeignKey("incident_categories.id",ondelete="RESTRICT"),nullable=False),
        sa.Column("confidence",sa.Numeric(6,5),nullable=False),sa.Column("class_probabilities",sa.JSON()),
        sa.Column("inference_time_ms",sa.Integer()),sa.Column("needs_review",sa.Boolean(),nullable=False,server_default=sa.false()),*audit_columns(),
        sa.CheckConstraint("confidence BETWEEN 0 AND 1",name="ck_predictions_confidence"))
    op.create_index("ix_predictions_report_created","predictions",["report_id","created_at"])
    op.create_index("ix_predictions_needs_review","predictions",["needs_review"])

    op.create_table("prediction_reviews",
        sa.Column("id",UUID,primary_key=True),sa.Column("prediction_id",UUID,sa.ForeignKey("predictions.id",ondelete="CASCADE"),nullable=False),
        sa.Column("reviewer_user_id",UUID,sa.ForeignKey("users.id",ondelete="RESTRICT"),nullable=False),
        sa.Column("confirmed_category_id",UUID,sa.ForeignKey("incident_categories.id",ondelete="RESTRICT"),nullable=False),
        sa.Column("action",sa.String(20),nullable=False,server_default="CONFIRMED"),sa.Column("notes",sa.Text()),*audit_columns())
    op.create_index("ix_prediction_reviews_prediction_created","prediction_reviews",["prediction_id","created_at"])

    op.create_table("dataset_samples",
        sa.Column("id",UUID,primary_key=True),sa.Column("dataset_id",UUID,sa.ForeignKey("datasets.id",ondelete="CASCADE"),nullable=False),
        sa.Column("report_id",UUID,sa.ForeignKey("reports.id",ondelete="SET NULL")),sa.Column("raw_text",sa.Text(),nullable=False),
        sa.Column("processed_text",sa.Text()),sa.Column("category_id",UUID,sa.ForeignKey("incident_categories.id",ondelete="RESTRICT"),nullable=False),
        sa.Column("split",sa.String(12),nullable=False),sa.Column("is_verified",sa.Boolean(),nullable=False,server_default=sa.false()),*audit_columns(),
        sa.CheckConstraint("split IN ('TRAIN','VALIDATION','TEST')",name="ck_dataset_samples_split"))
    op.create_index("ix_dataset_samples_dataset_split","dataset_samples",["dataset_id","split"])

    op.create_table("training_runs",
        sa.Column("id",UUID,primary_key=True),sa.Column("dataset_id",UUID,sa.ForeignKey("datasets.id",ondelete="RESTRICT"),nullable=False),
        sa.Column("model_version_id",UUID,sa.ForeignKey("model_versions.id",ondelete="SET NULL")),
        sa.Column("algorithm",sa.String(30),nullable=False),sa.Column("representation",sa.String(30),nullable=False),
        sa.Column("hyperparameters",sa.JSON()),sa.Column("imbalance_method",sa.String(80)),
        sa.Column("status",sa.String(20),nullable=False,server_default="PENDING"),sa.Column("started_at",TS),sa.Column("finished_at",TS),
        sa.Column("artifact_path",sa.String(1000)),*audit_columns(),
        sa.CheckConstraint("status IN ('PENDING','RUNNING','COMPLETED','FAILED')",name="ck_training_runs_status"))
    op.create_index("ix_training_runs_dataset_created","training_runs",["dataset_id","created_at"])

    op.create_table("training_metrics",
        sa.Column("id",UUID,primary_key=True),sa.Column("training_run_id",UUID,sa.ForeignKey("training_runs.id",ondelete="CASCADE"),nullable=False),
        sa.Column("metric_name",sa.String(60),nullable=False),sa.Column("metric_value",sa.Numeric(8,6),nullable=False),
        sa.Column("class_label",sa.String(3),nullable=False,server_default="ALL"),sa.Column("details",sa.JSON()),*audit_columns(),
        sa.UniqueConstraint("training_run_id","metric_name","class_label",name="uq_training_metrics_run_name_class"),
        sa.CheckConstraint("metric_value BETWEEN 0 AND 1",name="ck_training_metrics_value"))

def downgrade():
    for name in ("training_metrics","training_runs","dataset_samples","prediction_reviews","predictions","preprocessing_runs","report_attachments","reports","datasets","model_versions","incident_categories","users"):
        op.drop_table(name)
