"""Pydantic schemas for model predictions and preprocessing audit records."""
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PredictionCreate(BaseModel):
    report_id: UUID


class PredictionCategoryRead(BaseModel):
    code: str
    name: str


class PredictionPreprocessingRead(BaseModel):
    id: UUID
    pipeline_version: str
    cleaned_text: str
    normalized_text: str
    segmented_text: str
    stopwords_removed_text: str
    processing_time_ms: int | None


class PredictionModelRead(BaseModel):
    name: str
    version: str
    algorithm: str
    representation: str


class PredictionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    report_id: UUID
    preprocessing_run_id: UUID | None
    predicted_category: PredictionCategoryRead
    model_version: PredictionModelRead
    confidence: Decimal | None
    confidence_type: str
    probability_calibrated: bool | None
    decision_margin: Decimal | None
    class_probabilities: dict[str, float] | None
    needs_review: bool
    review_reasons: list[str] = Field(default_factory=list)
    inference_time_ms: int | None
    preprocessing: PredictionPreprocessingRead
    created_at: datetime
