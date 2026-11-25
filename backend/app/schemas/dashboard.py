"""Response schemas for staff dashboard statistics and map data."""
from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DashboardCategoryStat(BaseModel):
    code: str
    name: str
    description: str | None = None
    is_active: bool
    report_count: int = Field(ge=0)
    percentage: float = Field(ge=0, le=100)


class DashboardStatusStat(BaseModel):
    status: str
    report_count: int = Field(ge=0)


class DashboardSummaryRead(BaseModel):
    generated_at: datetime
    date_from: date | None
    date_to: date | None
    total_reports: int
    categorized_reports: int
    unclassified_reports: int
    geolocated_reports: int
    pending_reports: int
    resolved_reports: int
    rejected_reports: int
    categories: list[DashboardCategoryStat]
    statuses: list[DashboardStatusStat]


class DashboardMapPoint(BaseModel):
    report_id: UUID
    tracking_code: str
    latitude: float
    longitude: float
    address: str | None
    status: str
    category_code: str | None
    category_name: str | None
    submitted_at: datetime
    content_preview: str


class DashboardMapRead(BaseModel):
    items: list[DashboardMapPoint]
    total: int
    limit: int
    offset: int
    has_more: bool
