"""Schemas for staff review decisions and immutable review history."""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReviewCreate(BaseModel):
    prediction_id: UUID
    confirmed_category_code: str = Field(min_length=3, max_length=3)
    notes: str | None = Field(default=None, max_length=4000)

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
