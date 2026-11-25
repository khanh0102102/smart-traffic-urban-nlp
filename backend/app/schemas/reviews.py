"""Schemas for staff review decisions and immutable review history."""
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ReviewCreate(BaseModel):
    prediction_id: UUID
    confirmed_category_code: str = Field(min_length=3, max_length=3)
    notes: str | None = Field(default=None, max_length=4000)
    # Optional, explicit opt-in to materialize this verified decision as training data.
    dataset_id: UUID | None = None
    dataset_split: Literal["TRAIN", "VALIDATION", "TEST"] | None = None

    @model_validator(mode="after")
    def validate_dataset_target(self):
        if (self.dataset_id is None) != (self.dataset_split is None):
            raise ValueError("dataset_id and dataset_split must be provided together.")
        return self

    @field_validator("confirmed_category_code")
    @classmethod
    def normalize_category_code(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("notes")
    @classmethod
    def normalize_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ReviewCategoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    code: str
    name: str


class ReviewUserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    full_name: str
    role: str


class ReviewRead(BaseModel):
    id: UUID
    prediction_id: UUID
    report_id: UUID
    tracking_code: str
    predicted_category: ReviewCategoryRead
    confirmed_category: ReviewCategoryRead
    reviewer: ReviewUserRead
    action: str
    notes: str | None
    created_at: datetime
