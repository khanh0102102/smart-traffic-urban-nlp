"""Citizen report submission endpoint."""

from datetime import datetime, timezone
import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from stnlp.api.schemas import CategoryOut, ReportCreateOut, ReportCreateRequest
from stnlp.config import settings
from stnlp.database.models import Category, Prediction as PredictionRow, Report
from stnlp.database.session import get_db
from stnlp.inference.predictor import ModelNotReadyError, get_predictor
from stnlp.labels import label_name

router = APIRouter(tags=["reports"])


@router.post("/reports", response_model=ReportCreateOut, status_code=status.HTTP_201_CREATED)
def create_report(payload: ReportCreateRequest, db: Session = Depends(get_db)):
    try:
        result = get_predictor().predict(
            payload.text, include_probabilities=payload.include_probabilities
        )
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Mô hình chưa sẵn sàng: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    category = db.query(Category).filter(Category.code == result.predicted_code).first()
    if category is None:
        raise HTTPException(status_code=500, detail="Danh mục dự đoán chưa được khởi tạo.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    report = Report(
        report_code=f"STNLP-{timestamp}-{secrets.token_hex(3).upper()}",
        text=payload.text,
        latitude=payload.latitude,
        longitude=payload.longitude,
        image_url=payload.image_url,
        status="needs_review" if result.confidence <= settings.confidence_threshold else "new",
    )
    db.add(report)
    db.flush()

    prediction = PredictionRow(
        report_id=report.id,
        category_id=category.id,
        confidence=result.confidence,
        probabilities=result.probabilities or None,
        model_name=result.model_name,
        model_version=result.model_version,
        needs_review=result.confidence <= settings.confidence_threshold,
    )
    db.add(prediction)
    db.commit()
    db.refresh(report)

    return ReportCreateOut(
        report_id=report.id,
        report_code=report.report_code,
        status=report.status,
        predicted_category=CategoryOut(code=category.code, name=category.name),
        confidence=result.confidence,
        needs_review=result.confidence <= settings.confidence_threshold,
        model_name=result.model_name,
        model_version=result.model_version,
        probabilities=result.probabilities or None,
    )
