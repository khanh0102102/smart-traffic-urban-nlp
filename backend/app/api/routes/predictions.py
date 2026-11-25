"""Authenticated AI classification API for citizen reports."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.api.dependencies import get_current_user
from app.database.models import Prediction, Report, User
from app.database.session import get_db
from app.schemas.predictions import PredictionCreate, PredictionRead
from app.services.prediction_service import execute_prediction, get_model_predictor

router = APIRouter(prefix="/predictions", tags=["AI Prediction"])


def _authorized_report(db: Session, report_id: UUID, user: User) -> Report:
    report = db.scalar(select(Report).where(Report.id == report_id))
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    if user.role not in {"ADMIN", "RECEIVER"}:
        if user.role != "CITIZEN" or report.user_id != user.id:
            raise HTTPException(status_code=404, detail="Report not found.")
    return report


def _prediction_query(db: Session, prediction_id: UUID, user: User) -> Prediction:
    prediction = db.scalar(
        select(Prediction)
        .where(Prediction.id == prediction_id)
        .options(
            joinedload(Prediction.predicted_category),
            joinedload(Prediction.model_version),
            joinedload(Prediction.preprocessing_run),
        )
    )
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    _authorized_report(db, prediction.report_id, user)
    return prediction


@router.post("", response_model=PredictionRead, status_code=status.HTTP_201_CREATED)
def create_prediction(
    payload: PredictionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    predictor=Depends(get_model_predictor),
):
    """Run the packaged NLP classifier for an existing report and save its audit trail."""
    report = _authorized_report(db, payload.report_id, current_user)
    return execute_prediction(db, report, predictor)


@router.get("/{prediction_id}", response_model=PredictionRead)
def get_prediction(
    prediction_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve one prior prediction, subject to the source report's access rules."""
    return _prediction_query(db, prediction_id, current_user)


@router.get("/report/{report_id}", response_model=list[PredictionRead])
def list_report_predictions(
    report_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve the prediction history for an owned report or any report visible to staff."""
    _authorized_report(db, report_id, current_user)
    rows = db.scalars(
        select(Prediction)
        .where(Prediction.report_id == report_id)
        .options(
            joinedload(Prediction.predicted_category),
            joinedload(Prediction.model_version),
            joinedload(Prediction.preprocessing_run),
        )
        .order_by(Prediction.created_at.desc())
    ).unique().all()
    return rows
