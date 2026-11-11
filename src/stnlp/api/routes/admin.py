"""Admin review queue, moderation history and summary statistics."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from stnlp.api.dependencies import require_admin
from stnlp.api.schemas import ModerationRequest
from stnlp.database.models import Category, ModerationLog, Prediction, Report
from stnlp.database.session import get_db

router = APIRouter(prefix="/admin", tags=["admin"])


def _latest_prediction(report: Report) -> Prediction | None:
    if not report.predictions:
        return None
    return max(report.predictions, key=lambda item: item.id)


def _serialize_report(report: Report) -> dict:
    latest = _latest_prediction(report)
    return {
        "id": report.id,
        "report_code": report.report_code,
        "text": report.text,
        "latitude": report.latitude,
        "longitude": report.longitude,
        "image_url": report.image_url,
        "status": report.status,
        "created_at": report.created_at.isoformat() if report.created_at else None,
        "predicted_category": (
            {"code": latest.category.code, "name": latest.category.name} if latest else None
        ),
        "confidence": latest.confidence if latest else None,
        "needs_review": latest.needs_review if latest else None,
        "model_name": latest.model_name if latest else None,
        "model_version": latest.model_version if latest else None,
        "confirmed_category": (
            {
                "code": report.confirmed_category.code,
                "name": report.confirmed_category.name,
            }
            if report.confirmed_category else None
        ),
    }


@router.get("/reports")
def list_reports(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    needs_review: bool | None = None,
    _admin: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    latest_prediction_id = (
        db.query(func.max(Prediction.id))
        .filter(Prediction.report_id == Report.id)
        .correlate(Report)
        .scalar_subquery()
    )
    query = db.query(Report).outerjoin(
        Prediction, Prediction.id == latest_prediction_id
    )
    if needs_review is not None:
        query = query.filter(Prediction.needs_review.is_(needs_review))
    total = query.count()
    reports = (
        query.order_by(Report.created_at.desc(), Report.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return {"total": total, "skip": skip, "limit": limit,
            "items": [_serialize_report(report) for report in reports]}


@router.get("/reports/{report_id}")
def get_report(
    report_id: int,
    _admin: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    report = db.query(Report).filter(Report.id == report_id).first()
    if report is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy phản ánh.")
    result = _serialize_report(report)
    result["moderation_history"] = [
        {
            "id": log.id,
            "old_category_id": log.old_category_id,
            "new_category_id": log.new_category_id,
            "admin_identifier": log.admin_identifier,
            "note": log.note,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in sorted(report.moderation_logs, key=lambda item: item.id, reverse=True)
    ]
    return result


@router.patch("/reports/{report_id}/moderation")
def moderate_report(
    report_id: int,
    payload: ModerationRequest,
    admin_identifier: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    report = db.query(Report).filter(Report.id == report_id).first()
    if report is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy phản ánh.")
    category = db.query(Category).filter(Category.code == payload.category_code).first()
    if category is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy danh mục.")

    latest = _latest_prediction(report)
    log = ModerationLog(
        report_id=report.id,
        admin_identifier=admin_identifier,
        old_category_id=latest.category_id if latest else report.confirmed_category_id,
        new_category_id=category.id,
        note=payload.note,
    )
    db.add(log)
    report.confirmed_category_id = category.id
    report.status = "reviewed"
    db.commit()
    db.refresh(report)
    return {
        "message": "Đã lưu kết quả kiểm duyệt.",
        "report": _serialize_report(report),
    }


@router.get("/stats")
def admin_stats(
    _admin: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    total_reports = db.query(func.count(Report.id)).scalar() or 0
    needs_review_count = (
        db.query(func.count(Prediction.id))
        .filter(Prediction.needs_review.is_(True))
        .scalar() or 0
    )
    by_category = (
        db.query(Category.code, Category.name, func.count(Prediction.id))
        .outerjoin(Prediction, Prediction.category_id == Category.id)
        .group_by(Category.id, Category.code, Category.name)
        .order_by(Category.code)
        .all()
    )
    return {
        "total_reports": int(total_reports),
        "needs_review_predictions": int(needs_review_count),
        "predictions_by_category": [
            {"code": code, "name": name, "count": int(count)}
            for code, name, count in by_category
        ],
    }
