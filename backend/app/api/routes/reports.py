"""Citizen report submission, attachment storage and staff management API."""
from __future__ import annotations

import math
import secrets
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.api.dependencies import get_current_user, require_roles
from app.core.config import settings
from app.database.models import IncidentCategory, Report, ReportAttachment, User
from app.database.session import get_db
from app.schemas.reports import (
    ReportAttachmentRead,
    ReportCategoryRead,
    ReportCreate,
    ReportPage,
    ReportRead,
    ReportStatusUpdate,
)

router = APIRouter(prefix="/reports", tags=["Reports"])
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
REPORT_STATUSES = {"RECEIVED", "PROCESSING", "RESOLVED", "REJECTED"}


def _new_tracking_code(db: Session) -> str:
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    for _ in range(5):
        code = f"RPT-{date_part}-{secrets.token_hex(4).upper()}"
        exists = db.scalar(select(Report.id).where(Report.tracking_code == code))
        if exists is None:
            return code
    raise HTTPException(status_code=503, detail="Could not allocate a report tracking code.")


def _report_response(report: Report) -> ReportRead:
    category = None
    if report.final_category is not None:
        category = ReportCategoryRead(
            code=report.final_category.code,
            name=report.final_category.name,
            description=report.final_category.description,
        )
    attachments = [
        ReportAttachmentRead(
            id=item.id,
            original_filename=item.original_filename,
            mime_type=item.mime_type,
            file_size=item.file_size,
            file_url=f"/api/v1/reports/{report.id}/attachments/{item.id}",
        )
        for item in report.attachments
    ]
    return ReportRead(
        id=report.id,
        tracking_code=report.tracking_code,
        user_id=report.user_id,
        raw_content=report.raw_content,
        address=report.address,
        latitude=report.latitude,
        longitude=report.longitude,
        status=report.status,
        category=category,
        attachments=attachments,
        submitted_at=report.submitted_at,
        created_at=report.created_at,
        updated_at=report.updated_at,
    )


def _report_or_404(db: Session, report_id: UUID) -> Report:
    report = db.scalar(
        select(Report)
        .where(Report.id == report_id)
        .options(joinedload(Report.final_category), selectinload(Report.attachments))
    )
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


def _assert_report_access(report: Report, user: User) -> None:
    if user.role in {"ADMIN", "RECEIVER"}:
        return
    if user.role != "CITIZEN" or report.user_id != user.id:
        # Do not disclose whether another citizen's report exists.
        raise HTTPException(status_code=404, detail="Report not found.")


def _page_response(
    db: Session,
    statement,
    count_statement,
    page: int,
    page_size: int,
) -> ReportPage:
    total = int(db.scalar(count_statement) or 0)
    reports = db.scalars(
        statement
        .options(joinedload(Report.final_category), selectinload(Report.attachments))
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return ReportPage(
        items=[_report_response(report) for report in reports],
        total=total,
        page=page,
        page_size=page_size,
        pages=math.ceil(total / page_size) if total else 0,
    )


@router.get("/categories", response_model=list[ReportCategoryRead])
def list_categories(
    _current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return active R01-R11 categories for display and staff classification."""
    rows = db.scalars(
        select(IncidentCategory)
        .where(IncidentCategory.is_active.is_(True))
        .order_by(IncidentCategory.sort_order)
    ).all()
    return [
        ReportCategoryRead(code=row.code, name=row.name, description=row.description)
        for row in rows
    ]


@router.post("", response_model=ReportRead, status_code=status.HTTP_201_CREATED)
def create_report(
    payload: ReportCreate,
    current_user: User = Depends(require_roles("CITIZEN")),
    db: Session = Depends(get_db),
):
    """Create a citizen report; image files are uploaded to the attachment endpoint."""
    report = Report(
        tracking_code=_new_tracking_code(db),
        user_id=current_user.id,
        raw_content=payload.raw_content,
        address=payload.address,
        latitude=payload.latitude,
        longitude=payload.longitude,
        status="RECEIVED",
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return _report_response(report)


@router.get("/mine", response_model=ReportPage)
def list_my_reports(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(require_roles("CITIZEN")),
    db: Session = Depends(get_db),
):
    """List only the authenticated citizen's own reports."""
    base = select(Report).where(Report.user_id == current_user.id).order_by(Report.submitted_at.desc())
    count = select(func.count(Report.id)).where(Report.user_id == current_user.id)
    return _page_response(db, base, count, page, page_size)


@router.get("", response_model=ReportPage)
def list_all_reports(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    report_status: str | None = Query(default=None, alias="status"),
    category_code: str | None = Query(default=None, min_length=3, max_length=3),
    search: str | None = Query(default=None, min_length=1, max_length=200),
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Staff-only paginated report queue with optional status/category/text filters."""
    conditions = []
    if report_status is not None:
        if report_status not in REPORT_STATUSES:
            raise HTTPException(status_code=422, detail="Unsupported report status.")
        conditions.append(Report.status == report_status)
    if category_code is not None:
        category = db.scalar(
            select(IncidentCategory).where(
                IncidentCategory.code == category_code,
                IncidentCategory.is_active.is_(True),
            )
        )
        if category is None:
            raise HTTPException(status_code=422, detail="Unknown or inactive category code.")
        conditions.append(Report.final_category_id == category.id)
    if search:
        conditions.append(Report.raw_content.ilike(f"%{search.strip()}%"))
    base = select(Report).where(*conditions).order_by(Report.submitted_at.desc())
    count = select(func.count(Report.id)).where(*conditions)
    return _page_response(db, base, count, page, page_size)


@router.get("/tracking/{tracking_code}", response_model=ReportRead)
def get_report_by_tracking_code(
    tracking_code: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    report = db.scalar(
        select(Report)
        .where(Report.tracking_code == tracking_code)
        .options(joinedload(Report.final_category), selectinload(Report.attachments))
    )
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    _assert_report_access(report, current_user)
    return _report_response(report)


@router.get("/{report_id}", response_model=ReportRead)
def get_report(
    report_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    report = _report_or_404(db, report_id)
    _assert_report_access(report, current_user)
    return _report_response(report)


@router.patch("/{report_id}/status", response_model=ReportRead)
def update_report_status(
    report_id: UUID,
    payload: ReportStatusUpdate,
    current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Allow staff to update workflow status and optionally assign a final category."""
    report = _report_or_404(db, report_id)
    report.status = payload.status
    if "final_category_code" in payload.model_fields_set:
        if payload.final_category_code is None:
            report.final_category_id = None
        else:
            category = db.scalar(
                select(IncidentCategory).where(
                    IncidentCategory.code == payload.final_category_code,
                    IncidentCategory.is_active.is_(True),
                )
            )
            if category is None:
                raise HTTPException(status_code=422, detail="Unknown or inactive category code.")
            report.final_category_id = category.id
    db.commit()
    return _report_response(_report_or_404(db, report_id))


def _detect_image_type(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


@router.post("/{report_id}/attachments", response_model=list[ReportAttachmentRead], status_code=status.HTTP_201_CREATED)
async def upload_report_attachments(
    report_id: UUID,
    files: list[UploadFile] = File(..., description="JPEG, PNG or WebP images; up to configured per-report limit."),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload validated images; only the report owner or staff may attach files."""
    report = _report_or_404(db, report_id)
    _assert_report_access(report, current_user)
    if not files:
        raise HTTPException(status_code=422, detail="At least one image file is required.")
    if len(report.attachments) + len(files) > settings.report_max_attachments:
        raise HTTPException(
            status_code=413,
            detail=f"A report may contain at most {settings.report_max_attachments} images.",
        )

    validated: list[tuple[UploadFile, bytes, str, str]] = []
    for upload in files:
        content = await upload.read(settings.report_max_upload_size_bytes + 1)
        if not content:
            raise HTTPException(status_code=422, detail="Empty image files are not allowed.")
        if len(content) > settings.report_max_upload_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Each image must be at most {settings.report_max_upload_size_bytes} bytes.",
            )
        detected_type = _detect_image_type(content)
        if detected_type is None:
            raise HTTPException(status_code=415, detail="Only valid JPEG, PNG and WebP images are accepted.")
        if upload.content_type not in {None, "", "application/octet-stream", detected_type}:
            raise HTTPException(status_code=415, detail="Image content does not match its declared MIME type.")
        original_name = (upload.filename or "image").replace("\\", "/").split("/")[-1]
        original_name = original_name.replace("\x00", "").strip()[:255] or "image"
        validated.append((upload, content, detected_type, original_name))

    storage_root = settings.report_storage_root
    report_folder = storage_root / "reports" / str(report.id)
    report_folder.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []
    created_records: list[ReportAttachment] = []
    try:
        for _upload, content, mime_type, original_name in validated:
            extension = ALLOWED_IMAGE_TYPES[mime_type]
            stored_name = f"{uuid4().hex}{extension}"
            destination = report_folder / stored_name
            temporary = report_folder / f".{stored_name}.tmp"
            temporary.write_bytes(content)
            temporary.replace(destination)
            created_paths.append(destination)
            relative_path = destination.relative_to(storage_root).as_posix()
            record = ReportAttachment(
                report_id=report.id,
                original_filename=original_name,
                storage_path=relative_path,
                mime_type=mime_type,
                file_size=len(content),
            )
            db.add(record)
            created_records.append(record)
        db.commit()
    except Exception:
        db.rollback()
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise

    for record in created_records:
        db.refresh(record)
    return [
        ReportAttachmentRead(
            id=record.id,
            original_filename=record.original_filename,
            mime_type=record.mime_type,
            file_size=record.file_size,
            file_url=f"/api/v1/reports/{report.id}/attachments/{record.id}",
        )
        for record in created_records
    ]


@router.get("/{report_id}/attachments/{attachment_id}")
def download_report_attachment(
    report_id: UUID,
    attachment_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Serve an attachment only to the report owner or authorized staff."""
    report = _report_or_404(db, report_id)
    _assert_report_access(report, current_user)
    attachment = db.scalar(
        select(ReportAttachment).where(
            ReportAttachment.id == attachment_id,
            ReportAttachment.report_id == report.id,
        )
    )
    if attachment is None:
        raise HTTPException(status_code=404, detail="Attachment not found.")

    storage_root = settings.report_storage_root
    file_path = (storage_root / attachment.storage_path).resolve()
    if storage_root not in file_path.parents or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Attachment file is unavailable.")
    return FileResponse(
        file_path,
        media_type=attachment.mime_type,
        headers={
            "Content-Disposition": "inline",
            "X-Content-Type-Options": "nosniff",
        },
    )
