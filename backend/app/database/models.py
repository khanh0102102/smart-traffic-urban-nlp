"""The 12 SQLAlchemy models for the ST-NLP database."""
from __future__ import annotations
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any
from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, JSON, Numeric, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.base import Base, TimestampMixin

class User(TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("role IN ('ADMIN', 'RECEIVER', 'CITIZEN')", name="ck_users_role"), Index("ix_users_role_active", "role", "is_active"))
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="CITIZEN", server_default="CITIZEN")
    google_sub: Mapped[str | None] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    reports: Mapped[list[Report]] = relationship(back_populates="submitter", foreign_keys="Report.user_id")
    reviews: Mapped[list[PredictionReview]] = relationship(back_populates="reviewer")
    datasets_created: Mapped[list[Dataset]] = relationship(back_populates="creator")

class IncidentCategory(TimestampMixin, Base):
    __tablename__ = "incident_categories"
    __table_args__ = (UniqueConstraint("sort_order", name="uq_incident_categories_sort_order"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    code: Mapped[str] = mapped_column(String(3), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(180), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    final_reports: Mapped[list[Report]] = relationship(back_populates="final_category", foreign_keys="Report.final_category_id")
    predictions: Mapped[list[Prediction]] = relationship(back_populates="predicted_category")
    confirmed_reviews: Mapped[list[PredictionReview]] = relationship(back_populates="confirmed_category")
    samples: Mapped[list[DatasetSample]] = relationship(back_populates="category")

class Report(TimestampMixin, Base):
    __tablename__ = "reports"
    __table_args__ = (CheckConstraint("latitude IS NULL OR (latitude >= -90 AND latitude <= 90)", name="ck_reports_latitude"), CheckConstraint("longitude IS NULL OR (longitude >= -180 AND longitude <= 180)", name="ck_reports_longitude"), Index("ix_reports_submitted_at", "submitted_at"), Index("ix_reports_final_category", "final_category_id"))
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    tracking_code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"))
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
    address: Mapped[str | None] = mapped_column(String(500))
    latitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    longitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    final_category_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("incident_categories.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="RECEIVED", server_default="RECEIVED")
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    submitter: Mapped[User | None] = relationship(back_populates="reports", foreign_keys=[user_id])
    final_category: Mapped[IncidentCategory | None] = relationship(back_populates="final_reports", foreign_keys=[final_category_id])
    attachments: Mapped[list[ReportAttachment]] = relationship(back_populates="report", cascade="all, delete-orphan")
    preprocessing_runs: Mapped[list[PreprocessingRun]] = relationship(back_populates="report", cascade="all, delete-orphan")
    predictions: Mapped[list[Prediction]] = relationship(back_populates="report", cascade="all, delete-orphan")
    dataset_samples: Mapped[list[DatasetSample]] = relationship(back_populates="report")

class ReportAttachment(TimestampMixin, Base):
    __tablename__ = "report_attachments"
    __table_args__ = (Index("ix_report_attachments_report_id", "report_id"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    report_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("reports.id", ondelete="CASCADE"), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(120), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    report: Mapped[Report] = relationship(back_populates="attachments")

class PreprocessingRun(TimestampMixin, Base):
    __tablename__ = "preprocessing_runs"
    __table_args__ = (Index("ix_preprocessing_runs_report_created", "report_id", "created_at"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    report_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("reports.id", ondelete="CASCADE"), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(80), nullable=False)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    segmented_text: Mapped[str] = mapped_column(Text, nullable=False)
    stopwords_removed_text: Mapped[str] = mapped_column(Text, nullable=False)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer)
    report: Mapped[Report] = relationship(back_populates="preprocessing_runs")
    predictions: Mapped[list[Prediction]] = relationship(back_populates="preprocessing_run")

class ModelVersion(TimestampMixin, Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_versions_name_version"), CheckConstraint("algorithm IN ('NAIVE_BAYES', 'SVM', 'CNN', 'LSTM', 'OTHER')", name="ck_model_versions_algorithm"), CheckConstraint("representation IN ('TF_IDF', 'WORD_EMBEDDING', 'OTHER')", name="ck_model_versions_representation"))
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[str] = mapped_column(String(60), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(30), nullable=False)
    representation: Mapped[str] = mapped_column(String(30), nullable=False)
    artifact_path: Mapped[str | None] = mapped_column(String(1000))
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    predictions: Mapped[list[Prediction]] = relationship(back_populates="model_version")
    training_runs: Mapped[list[TrainingRun]] = relationship(back_populates="model_version")

class Prediction(TimestampMixin, Base):
    __tablename__ = "predictions"
    __table_args__ = (CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_predictions_confidence"), CheckConstraint("confidence_type IN ('model_score', 'calibrated_probability', 'unavailable')", name="ck_predictions_confidence_type"), Index("ix_predictions_report_created", "report_id", "created_at"), Index("ix_predictions_needs_review", "needs_review"))
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    report_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("reports.id", ondelete="CASCADE"), nullable=False)
    preprocessing_run_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("preprocessing_runs.id", ondelete="SET NULL"))
    model_version_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False)
    predicted_category_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("incident_categories.id", ondelete="RESTRICT"), nullable=False)
    # Uncalibrated SVM predictions legitimately have no probability/confidence value.
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(6, 5), nullable=True)
    confidence_type: Mapped[str] = mapped_column(String(32), nullable=False, default="model_score", server_default="model_score")
    probability_calibrated: Mapped[bool | None] = mapped_column(Boolean)
    decision_margin: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    class_probabilities: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    inference_time_ms: Mapped[int | None] = mapped_column(Integer)
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    review_reasons: Mapped[list[str] | None] = mapped_column(JSON)
    report: Mapped[Report] = relationship(back_populates="predictions")
    preprocessing_run: Mapped[PreprocessingRun | None] = relationship(back_populates="predictions")
    model_version: Mapped[ModelVersion] = relationship(back_populates="predictions")
    predicted_category: Mapped[IncidentCategory] = relationship(back_populates="predictions")
    reviews: Mapped[list[PredictionReview]] = relationship(back_populates="prediction", cascade="all, delete-orphan")

class PredictionReview(TimestampMixin, Base):
    __tablename__ = "prediction_reviews"
    __table_args__ = (Index("ix_prediction_reviews_prediction_created", "prediction_id", "created_at"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False)
    reviewer_user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    confirmed_category_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("incident_categories.id", ondelete="RESTRICT"), nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False, default="CONFIRMED", server_default="CONFIRMED")
    notes: Mapped[str | None] = mapped_column(Text)
    prediction: Mapped[Prediction] = relationship(back_populates="reviews")
    reviewer: Mapped[User] = relationship(back_populates="reviews")
    confirmed_category: Mapped[IncidentCategory] = relationship(back_populates="confirmed_reviews")

class Dataset(TimestampMixin, Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_datasets_name_version"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    version: Mapped[str] = mapped_column(String(60), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(255))
    created_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"))
    creator: Mapped[User | None] = relationship(back_populates="datasets_created")
    samples: Mapped[list[DatasetSample]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    training_runs: Mapped[list[TrainingRun]] = relationship(back_populates="dataset")

class DatasetSample(TimestampMixin, Base):
    __tablename__ = "dataset_samples"
    __table_args__ = (CheckConstraint("split IN ('TRAIN', 'VALIDATION', 'TEST')", name="ck_dataset_samples_split"), Index("ix_dataset_samples_dataset_split", "dataset_id", "split"))
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    report_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("reports.id", ondelete="SET NULL"))
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    processed_text: Mapped[str | None] = mapped_column(Text)
    category_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("incident_categories.id", ondelete="RESTRICT"), nullable=False)
    split: Mapped[str] = mapped_column(String(12), nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    dataset: Mapped[Dataset] = relationship(back_populates="samples")
    report: Mapped[Report | None] = relationship(back_populates="dataset_samples")
    category: Mapped[IncidentCategory] = relationship(back_populates="samples")

class TrainingRun(TimestampMixin, Base):
    __tablename__ = "training_runs"
    __table_args__ = (CheckConstraint("status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')", name="ck_training_runs_status"), Index("ix_training_runs_dataset_created", "dataset_id", "created_at"))
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("datasets.id", ondelete="RESTRICT"), nullable=False)
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("model_versions.id", ondelete="SET NULL"))
    algorithm: Mapped[str] = mapped_column(String(30), nullable=False)
    representation: Mapped[str] = mapped_column(String(30), nullable=False)
    hyperparameters: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    imbalance_method: Mapped[str | None] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING", server_default="PENDING")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    artifact_path: Mapped[str | None] = mapped_column(String(1000))
    dataset: Mapped[Dataset] = relationship(back_populates="training_runs")
    model_version: Mapped[ModelVersion | None] = relationship(back_populates="training_runs")
    metrics: Mapped[list[TrainingMetric]] = relationship(back_populates="training_run", cascade="all, delete-orphan")

class TrainingMetric(TimestampMixin, Base):
    __tablename__ = "training_metrics"
    __table_args__ = (UniqueConstraint("training_run_id", "metric_name", "class_label", name="uq_training_metrics_run_name_class"), CheckConstraint("metric_value >= 0 AND metric_value <= 1", name="ck_training_metrics_value"))
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    training_run_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(60), nullable=False)
    metric_value: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    class_label: Mapped[str] = mapped_column(String(3), nullable=False, default="ALL", server_default="ALL")
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    training_run: Mapped[TrainingRun] = relationship(back_populates="metrics")
