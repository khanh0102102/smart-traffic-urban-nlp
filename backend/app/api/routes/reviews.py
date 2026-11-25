"""Staff review API for confirming or correcting AI prediction labels."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.api.dependencies import get_current_user, require_roles
from app.database.models import (
    Dataset,
    DatasetSample,
    IncidentCategory,
    Prediction,
    PredictionReview,
    Report,
    User,
)
from app.database.session import get_db
from app.schemas.reviews import ReviewCategoryRead, ReviewCreate, ReviewRead, ReviewUserRead

router = APIRouter(prefix="/reviews", tags=["Prediction Review"])


def _authorized_report(report: Report | None, user: User) -> Report:
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    if user.role in {"ADMIN", "RECEIVER"}:
        return report
    if user.role != "CITIZEN" or report.user_id != user.id:
        raise HTTPException(status_code=404, detail="Review not found.")
    return report


def _prediction_for_review(db: Session, prediction_id: UUID, user: User) -> Prediction:
    prediction = db.scalar(
        select(Prediction)
        .where(Prediction.id == prediction_id)
        .options(joinedload(Prediction.report), joinedload(Prediction.predicted_category))
    )
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    _authorized_report(prediction.report, user)
    return prediction


def _review_response(review: PredictionReview) -> ReviewRead:
    prediction = review.prediction
    return ReviewRead(
        id=review.id,
        prediction_id=review.prediction_id,
        report_id=prediction.report_id,
        tracking_code=prediction.report.tracking_code,
        predicted_category=ReviewCategoryRead(
            code=prediction.predicted_category.code,
            name=prediction.predicted_category.name,
        ),
        confirmed_category=ReviewCategoryRead(
            code=review.confirmed_category.code,
            name=review.confirmed_category.name,
        ),
        reviewer=ReviewUserRead(
            id=review.reviewer.id,
            full_name=review.reviewer.full_name,
            role=review.reviewer.role,
        ),
        action=review.action,
        notes=review.notes,
        created_at=review.created_at,
    )


def _review_query(db: Session, review_id: UUID) -> PredictionReview:
    review = db.scalar(
        select(PredictionReview)
        .where(PredictionReview.id == review_id)
        .options(
            joinedload(PredictionReview.prediction).joinedload(Prediction.report),
            joinedload(PredictionReview.prediction).joinedload(Prediction.predicted_category),
            joinedload(PredictionReview.confirmed_category),
            joinedload(PredictionReview.reviewer),
        )
    )
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found.")
    return review


@router.post("", response_model=ReviewRead, status_code=status.HTTP_201_CREATED)
def create_review(
    payload: ReviewCreate,
    current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Confirm or correct an AI label; every decision appends a review-history row."""
    prediction = db.scalar(
        select(Prediction)
        .where(Prediction.id == payload.prediction_id)
        .options(
            joinedload(Prediction.report),
            joinedload(Prediction.predicted_category),
            joinedload(Prediction.preprocessing_run),
        )
    )
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    confirmed_category = db.scalar(
        select(IncidentCategory).where(
            IncidentCategory.code == payload.confirmed_category_code,
            IncidentCategory.is_active.is_(True),
        )
    )
    if confirmed_category is None:
        raise HTTPException(status_code=422, detail="Unknown or inactive R01-R11 category code.")

    target_dataset = None
    if payload.dataset_id is not None:
        target_dataset = db.get(Dataset, payload.dataset_id)
        if target_dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")

    action = "CONFIRMED" if confirmed_category.id == prediction.predicted_category_id else "CORRECTED"
    review = PredictionReview(
        prediction_id=prediction.id,
        reviewer_user_id=current_user.id,
        confirmed_category_id=confirmed_category.id,
        action=action,
        notes=payload.notes,
        created_at=datetime.now(timezone.utc),
    )
    try:
        # The report keeps the latest staff-confirmed label; review rows remain append-only.
        prediction.report.final_category = confirmed_category
        db.add(review)

        # A staff decision is verified ground truth. Refresh any dataset rows already
        # materialized from this report so later corrections do not leave stale labels.
        processed_text = (
            prediction.preprocessing_run.stopwords_removed_text
            if prediction.preprocessing_run is not None
            else None
        )
        samples = db.scalars(
            select(DatasetSample).where(DatasetSample.report_id == prediction.report_id)
        ).all()
        target_sample = None
        for sample in samples:
            sample.raw_text = prediction.report.raw_content
            if processed_text is not None:
                sample.processed_text = processed_text
            sample.category_id = confirmed_category.id
            sample.is_verified = True
            if target_dataset is not None and sample.dataset_id == target_dataset.id:
                target_sample = sample

        if target_dataset is not None and target_sample is None:
            db.add(
                DatasetSample(
                    dataset_id=target_dataset.id,
                    report_id=prediction.report_id,
                    raw_text=prediction.report.raw_content,
                    processed_text=processed_text,
                    category_id=confirmed_category.id,
                    split=payload.dataset_split,
                    is_verified=True,
                )
            )
        elif target_sample is not None:
            target_sample.split = payload.dataset_split

        db.flush()
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="The review decision could not be saved.") from exc

    return _review_response(_review_query(db, review.id))


@router.get("/prediction/{prediction_id}", response_model=list[ReviewRead])
def list_prediction_reviews(
    prediction_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return review decisions for a prediction, newest first."""
    _prediction_for_review(db, prediction_id, current_user)
    rows = db.scalars(
        select(PredictionReview)
        .where(PredictionReview.prediction_id == prediction_id)
        .options(
            joinedload(PredictionReview.prediction).joinedload(Prediction.report),
            joinedload(PredictionReview.prediction).joinedload(Prediction.predicted_category),
            joinedload(PredictionReview.confirmed_category),
            joinedload(PredictionReview.reviewer),
        )
        .order_by(PredictionReview.created_at.desc(), PredictionReview.id.desc())
    ).unique().all()
    return [_review_response(row) for row in rows]


@router.get("/report/{report_id}", response_model=list[ReviewRead])
def list_report_reviews(
    report_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the review audit trail for a report, newest first."""
    report = db.scalar(select(Report).where(Report.id == report_id))
    _authorized_report(report, current_user)
    rows = db.scalars(
        select(PredictionReview)
        .join(Prediction, Prediction.id == PredictionReview.prediction_id)
        .where(Prediction.report_id == report_id)
        .options(
            joinedload(PredictionReview.prediction).joinedload(Prediction.report),
            joinedload(PredictionReview.prediction).joinedload(Prediction.predicted_category),
            joinedload(PredictionReview.confirmed_category),
            joinedload(PredictionReview.reviewer),
        )
        .order_by(PredictionReview.created_at.desc(), PredictionReview.id.desc())
    ).unique().all()
    return [_review_response(row) for row in rows]


@router.get("/{review_id}", response_model=ReviewRead)
def get_review(
    review_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve one review decision subject to the source report's access rules."""
    review = _review_query(db, review_id)
    _authorized_report(review.prediction.report, current_user)
    return _review_response(review)
