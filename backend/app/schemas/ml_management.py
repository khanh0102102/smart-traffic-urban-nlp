"""Pydantic schemas for dataset, sample, training run and metric management."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DatasetSplit = Literal["TRAIN", "VALIDATION", "TEST"]
TrainingAlgorithm = Literal["NAIVE_BAYES", "SVM", "CNN", "LSTM", "OTHER"]
TrainingRepresentation = Literal["TF_IDF", "WORD_EMBEDDING", "OTHER"]
TrainingStatus = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    version: str = Field(min_length=1, max_length=60)
    description: str | None = None
    source: str | None = Field(default=None, max_length=255)

    @field_validator("name", "version", "source")
    @classmethod
    def strip_required_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Value must not be empty.")
        return value


class DatasetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    version: str | None = Field(default=None, min_length=1, max_length=60)
    description: str | None = None
    source: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def require_update_field(self):
        if not self.model_fields_set:
            raise ValueError("At least one dataset field must be provided.")
        return self

    @field_validator("name", "version", "source")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Value must not be empty.")
        return value


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    version: str
    description: str | None
    source: str | None
    created_by: UUID | None
    created_at: datetime
    updated_at: datetime


class DatasetDetailRead(DatasetRead):
    sample_count: int
    train_sample_count: int
    validation_sample_count: int
    test_sample_count: int
    verified_sample_count: int


class DatasetPage(BaseModel):
    items: list[DatasetRead]
    total: int
    page: int
    page_size: int
    pages: int


class DatasetSampleCreate(BaseModel):
    raw_text: str = Field(min_length=1, max_length=20000)
    processed_text: str | None = Field(default=None, max_length=20000)
    category_code: str = Field(min_length=3, max_length=3)
    split: DatasetSplit
    is_verified: bool = False
    report_id: UUID | None = None

    @field_validator("raw_text", "processed_text")
    @classmethod
    def normalize_sample_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Sample text must not be empty.")
        return value

    @field_validator("category_code")
    @classmethod
    def normalize_category_code(cls, value: str) -> str:
        return value.strip().upper()


class DatasetSamplesBulkCreate(BaseModel):
    items: list[DatasetSampleCreate] = Field(min_length=1, max_length=500)


class DatasetSampleUpdate(BaseModel):
    raw_text: str | None = Field(default=None, min_length=1, max_length=20000)
    processed_text: str | None = Field(default=None, max_length=20000)
    category_code: str | None = Field(default=None, min_length=3, max_length=3)
    split: DatasetSplit | None = None
    is_verified: bool | None = None
    report_id: UUID | None = None

    @model_validator(mode="after")
    def validate_update(self):
        if not self.model_fields_set:
            raise ValueError("At least one sample field must be provided.")
        for key in ("raw_text", "category_code", "split", "is_verified"):
            if key in self.model_fields_set and getattr(self, key) is None:
                raise ValueError(f"{key} cannot be null.")
        return self

    @field_validator("raw_text", "processed_text")
    @classmethod
    def normalize_updated_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Sample text must not be empty.")
        return value

    @field_validator("category_code")
    @classmethod
    def normalize_updated_category_code(cls, value: str | None) -> str | None:
        return value.strip().upper() if value is not None else None


class DatasetSampleRead(BaseModel):
    id: UUID
    dataset_id: UUID
    report_id: UUID | None
    raw_text: str
    processed_text: str | None
    category_code: str
    category_name: str
    split: str
    is_verified: bool
    created_at: datetime
    updated_at: datetime


class DatasetSamplePage(BaseModel):
    items: list[DatasetSampleRead]
    total: int
    page: int
    page_size: int
    pages: int


class TrainingRunCreate(BaseModel):
    dataset_id: UUID
    algorithm: TrainingAlgorithm
    representation: TrainingRepresentation
    hyperparameters: dict[str, Any] | None = None
    imbalance_method: str | None = Field(default=None, max_length=80)


class TrainingRunStatusUpdate(BaseModel):
    status: Literal["RUNNING", "COMPLETED", "FAILED"]
    model_version_id: UUID | None = None
    artifact_path: str | None = Field(default=None, max_length=1000)


class TrainingMetricCreate(BaseModel):
    metric_name: str = Field(min_length=1, max_length=60)
    metric_value: Decimal = Field(ge=0, le=1, max_digits=8, decimal_places=6)
    class_label: str = Field(default="ALL", min_length=3, max_length=3)
    details: dict[str, Any] | None = None

    @field_validator("metric_name")
    @classmethod
    def normalize_metric_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Metric name must not be empty.")
        return value

    @field_validator("class_label")
    @classmethod
    def normalize_class_label(cls, value: str) -> str:
        value = value.strip().upper()
        if value != "ALL" and value not in {f"R{i:02d}" for i in range(1, 12)}:
            raise ValueError("class_label must be ALL or one of R01-R11.")
        return value


class TrainingMetricsBulkCreate(BaseModel):
    items: list[TrainingMetricCreate] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def reject_duplicate_keys(self):
        keys = [(item.metric_name, item.class_label) for item in self.items]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate (metric_name, class_label) pairs are not allowed.")
        return self


class TrainingMetricRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    training_run_id: UUID
    metric_name: str
    metric_value: Decimal
    class_label: str
    details: dict[str, Any] | None
    created_at: datetime


class TrainingRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    dataset_id: UUID
    model_version_id: UUID | None
    algorithm: str
    representation: str
    hyperparameters: dict[str, Any] | None
    imbalance_method: str | None
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    artifact_path: str | None
    created_at: datetime
    updated_at: datetime
    metrics: list[TrainingMetricRead] = []


class TrainingRunPage(BaseModel):
    items: list[TrainingRunRead]
    total: int
    page: int
    page_size: int
    pages: int
