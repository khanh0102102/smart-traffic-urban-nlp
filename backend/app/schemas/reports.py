"""Pydantic schemas for citizen reports and staff management."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ReportStatus = Literal["RECEIVED", "PROCESSING", "RESOLVED", "REJECTED"]


class ReportCreate(BaseModel):
    raw_content: str = Field(min_length=5, max_length=20000)
    address: str | None = Field(default=None, max_length=500)
    latitude: Decimal | None = Field(default=None, ge=-90, le=90, max_digits=10, decimal_places=7)
    longitude: Decimal | None = Field(default=None, ge=-180, le=180, max_digits=10, decimal_places=7)

    @field_validator("raw_content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 5:
            raise ValueError("Report content must contain at least 5 non-whitespace characters.")
        return value

    @field_validator("address")
    @classmethod
    def normalize_address(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_coordinates(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Latitude and longitude must be provided together.")
        return self


class ReportStatusUpdate(BaseModel):
    status: ReportStatus
    # Omit to leave the category unchanged; send null to clear it.
    final_category_code: str | None = Field(default=None, min_length=3, max_length=3)


class ReportCategoryRead(BaseModel):
    code: str
    name: str
    description: str | None = None


class ReportAttachmentRead(BaseModel):
    id: UUID
    original_filename: str
    mime_type: str
    file_size: int
    file_url: str


class ReportRead(BaseModel):
    id: UUID
    tracking_code: str
    user_id: UUID | None
    raw_content: str
    address: str | None
    latitude: Decimal | None
    longitude: Decimal | None
    status: str
    category: ReportCategoryRead | None
    attachments: list[ReportAttachmentRead]
    submitted_at: datetime
    created_at: datetime
    updated_at: datetime


class ReportPage(BaseModel):
    items: list[ReportRead]
    total: int
    page: int
    page_size: int
    pages: int
