"""Staff dashboard aggregate statistics and geolocated report API."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.api.dependencies import require_roles
from app.database.models import IncidentCategory, Prediction, Report, User
from app.database.session import get_db
from app.schemas.dashboard import (
    DashboardCategoryStat,
    DashboardMapPoint,
    DashboardMapRead,
    DashboardStatusStat,
    DashboardSummaryRead,
)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])
REPORT_STATUSES = ("RECEIVED", "PROCESSING", "RESOLVED", "REJECTED")


def _effective_category_id():
    """Use the latest staff-confirmed category, falling back to the newest AI prediction."""
    latest_prediction_category = (
        select(Prediction.predicted_category_id)
        .where(Prediction.report_id == Report.id)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .limit(1)
        .correlate(Report)
        .scalar_subquery()
    )
    return func.coalesce(Report.final_category_id, latest_prediction_category)


def _date_filters(date_from: date | None, date_to: date | None) -> list:
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must be on or before date_to.")

    filters = []
    if date_from:
        start = datetime.combine(date_from, time.min, tzinfo=timezone.utc)
        filters.append(Report.submitted_at >= start)
    if date_to:
        end_exclusive = datetime.combine(
            date_to + timedelta(days=1), time.min, tzinfo=timezone.utc
        )
        filters.append(Report.submitted_at < end_exclusive)
    return filters


@router.get("/summary", response_model=DashboardSummaryRead)
def get_dashboard_summary(
    date_from: date | None = Query(default=None, description="Inclusive start date (UTC)."),
    date_to: date | None = Query(default=None, description="Inclusive end date (UTC)."),
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
) -> DashboardSummaryRead:
    """Return overview counters and distribution across all configured incident categories."""
    filters = _date_filters(date_from, date_to)
    effective_category_id = _effective_category_id()

    total_reports = int(
        db.scalar(select(func.count(Report.id)).where(*filters)) or 0
    )
    categorized_reports = int(
        db.scalar(
            select(func.count(Report.id)).where(
                *filters, effective_category_id.is_not(None)
            )
        )
        or 0
    )
    geolocated_reports = int(
        db.scalar(
            select(func.count(Report.id)).where(
                *filters,
                Report.latitude.is_not(None),
                Report.longitude.is_not(None),
            )
        )
        or 0
    )

    category_join = and_(
        effective_category_id == IncidentCategory.id,
        *filters,
    )
    category_rows = db.execute(
        select(IncidentCategory, func.count(Report.id))
        .select_from(IncidentCategory)
        .outerjoin(Report, category_join)
        .group_by(IncidentCategory.id)
        .order_by(IncidentCategory.sort_order.asc(), IncidentCategory.code.asc())
    ).all()

    categories = [
        DashboardCategoryStat(
            code=category.code,
            name=category.name,
            description=category.description,
            is_active=category.is_active,
            report_count=int(report_count or 0),
            percentage=(
                round((int(report_count or 0) / categorized_reports) * 100, 2)
                if categorized_reports
                else 0.0
            ),
        )
        for category, report_count in category_rows
    ]

    status_rows = db.execute(
        select(Report.status, func.count(Report.id))
        .where(*filters)
        .group_by(Report.status)
    ).all()
    status_counts = {str(status): int(count) for status, count in status_rows}
    statuses = [
        DashboardStatusStat(status=status, report_count=status_counts.get(status, 0))
        for status in REPORT_STATUSES
    ]

    return DashboardSummaryRead(
        generated_at=datetime.now(timezone.utc),
        date_from=date_from,
        date_to=date_to,
        total_reports=total_reports,
        categorized_reports=categorized_reports,
        unclassified_reports=max(total_reports - categorized_reports, 0),
        geolocated_reports=geolocated_reports,
        pending_reports=status_counts.get("RECEIVED", 0)
        + status_counts.get("PROCESSING", 0),
        resolved_reports=status_counts.get("RESOLVED", 0),
        rejected_reports=status_counts.get("REJECTED", 0),
        categories=categories,
        statuses=statuses,
    )


@router.get("/map", response_model=DashboardMapRead)
def get_dashboard_map(
    date_from: date | None = Query(default=None, description="Inclusive start date (UTC)."),
    date_to: date | None = Query(default=None, description="Inclusive end date (UTC)."),
    category_code: str | None = Query(default=None, min_length=3, max_length=3),
    status: str | None = Query(default=None, pattern="^(RECEIVED|PROCESSING|RESOLVED|REJECTED)$"),
    limit: int = Query(default=500, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
) -> DashboardMapRead:
    """Return report locations for map markers, newest first, with optional filters."""
    filters = _date_filters(date_from, date_to)
    filters.extend(
        [
            Report.latitude.is_not(None),
            Report.longitude.is_not(None),
        ]
    )
    if status:
        filters.append(Report.status == status)
    effective_category_id = _effective_category_id()
    if category_code:
        normalized_code = category_code.strip().upper()
        category_id = db.scalar(
            select(IncidentCategory.id).where(IncidentCategory.code == normalized_code)
        )
        if category_id is None:
            raise HTTPException(status_code=422, detail="Unknown incident category code.")
        filters.append(effective_category_id == category_id)

    total = int(db.scalar(select(func.count(Report.id)).where(*filters)) or 0)
    rows = db.execute(
        select(Report, IncidentCategory)
        .select_from(Report)
        .outerjoin(IncidentCategory, effective_category_id == IncidentCategory.id)
        .where(*filters)
        .order_by(Report.submitted_at.desc(), Report.id.desc())
        .offset(offset)
        .limit(limit)
    ).all()

    items = [
        DashboardMapPoint(
            report_id=report.id,
            tracking_code=report.tracking_code,
            latitude=float(report.latitude),
            longitude=float(report.longitude),
            address=report.address,
            status=report.status,
            category_code=category.code if category else None,
            category_name=category.name if category else None,
            submitted_at=report.submitted_at,
            content_preview=report.raw_content.strip()[:240],
        )
        for report, category in rows
    ]
    return DashboardMapRead(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + len(items) < total,
    )
