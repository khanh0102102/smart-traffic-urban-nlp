"""Staff ML management API for datasets, samples, training runs and metrics."""
from datetime import datetime, timezone
from math import ceil
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from app.api.dependencies import require_roles
from app.database.models import (
    Dataset,
    DatasetSample,
    IncidentCategory,
    ModelVersion,
    Report,
    TrainingMetric,
    TrainingRun,
    User,
)
from app.database.session import get_db
from app.schemas.ml_management import (
    DatasetCreate,
    DatasetDetailRead,
    DatasetPage,
    DatasetRead,
    DatasetSampleCreate,
    DatasetSamplePage,
    DatasetSampleRead,
    DatasetSampleUpdate,
    DatasetSamplesBulkCreate,
    DatasetUpdate,
    TrainingMetricRead,
    TrainingMetricsBulkCreate,
    TrainingRunCreate,
    TrainingRunPage,
    TrainingRunRead,
    TrainingRunStatusUpdate,
)

router = APIRouter(prefix="/ml", tags=["ML Management"])


def _page_count(total: int, page_size: int) -> int:
    return ceil(total / page_size) if total else 0


def _get_dataset_or_404(db: Session, dataset_id: UUID) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return dataset


def _get_sample_or_404(db: Session, sample_id: UUID) -> DatasetSample:
    sample = db.get(DatasetSample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Dataset sample not found.")
    return sample


def _get_run_or_404(db: Session, run_id: UUID, *, with_metrics: bool = False) -> TrainingRun:
    statement = select(TrainingRun).where(TrainingRun.id == run_id)
    if with_metrics:
        statement = statement.options(selectinload(TrainingRun.metrics))
    run = db.scalar(statement)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return run


def _active_category_map(db: Session, codes: set[str]) -> dict[str, IncidentCategory]:
    if not codes:
        return {}
    rows = db.scalars(
        select(IncidentCategory).where(
            IncidentCategory.code.in_(codes),
            IncidentCategory.is_active.is_(True),
        )
    ).all()
    result = {row.code: row for row in rows}
    missing = sorted(codes - result.keys())
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown or inactive category code(s): {', '.join(missing)}.",
        )
    return result


def _dataset_detail(db: Session, dataset: Dataset) -> DatasetDetailRead:
    total = int(
        db.scalar(
            select(func.count(DatasetSample.id)).where(
                DatasetSample.dataset_id == dataset.id
            )
        )
        or 0
    )
    split_counts = dict(
        db.execute(
            select(DatasetSample.split, func.count(DatasetSample.id))
            .where(DatasetSample.dataset_id == dataset.id)
            .group_by(DatasetSample.split)
        ).all()
    )
    verified = int(
        db.scalar(
            select(func.count(DatasetSample.id)).where(
                DatasetSample.dataset_id == dataset.id,
                DatasetSample.is_verified.is_(True),
            )
        )
        or 0
    )
    return DatasetDetailRead(
        **DatasetRead.model_validate(dataset).model_dump(),
        sample_count=total,
        train_sample_count=int(split_counts.get("TRAIN", 0)),
        validation_sample_count=int(split_counts.get("VALIDATION", 0)),
        test_sample_count=int(split_counts.get("TEST", 0)),
        verified_sample_count=verified,
    )


def _sample_read(sample: DatasetSample, category: IncidentCategory) -> DatasetSampleRead:
    return DatasetSampleRead(
        id=sample.id,
        dataset_id=sample.dataset_id,
        report_id=sample.report_id,
        raw_text=sample.raw_text,
        processed_text=sample.processed_text,
        category_code=category.code,
        category_name=category.name,
        split=sample.split,
        is_verified=sample.is_verified,
        created_at=sample.created_at,
        updated_at=sample.updated_at,
    )


@router.post("/datasets", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
def create_dataset(
    payload: DatasetCreate,
    current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Create a versioned dataset; name and version must be unique together."""
    dataset = Dataset(
        name=payload.name,
        version=payload.version,
        description=payload.description,
        source=payload.source,
        created_by=current_user.id,
    )
    db.add(dataset)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="A dataset with this name and version already exists.",
        ) from exc
    db.refresh(dataset)
    return DatasetRead.model_validate(dataset)


@router.get("/datasets", response_model=DatasetPage)
def list_datasets(
    q: str | None = Query(default=None, max_length=160),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Search and paginate datasets by name, version, description or source."""
    statement = select(Dataset)
    count_statement = select(func.count(Dataset.id))
    if q and q.strip():
        term = f"%{q.strip()}%"
        predicate = or_(
            Dataset.name.ilike(term),
            Dataset.version.ilike(term),
            Dataset.description.ilike(term),
            Dataset.source.ilike(term),
        )
        statement = statement.where(predicate)
        count_statement = count_statement.where(predicate)
    total = int(db.scalar(count_statement) or 0)
    rows = db.scalars(
        statement.order_by(Dataset.created_at.desc(), Dataset.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return DatasetPage(
        items=[DatasetRead.model_validate(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
        pages=_page_count(total, page_size),
    )


@router.get("/datasets/{dataset_id}", response_model=DatasetDetailRead)
def get_dataset(
    dataset_id: UUID,
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Return dataset metadata and sample counts grouped by split."""
    return _dataset_detail(db, _get_dataset_or_404(db, dataset_id))


@router.patch("/datasets/{dataset_id}", response_model=DatasetRead)
def update_dataset(
    dataset_id: UUID,
    payload: DatasetUpdate,
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Update dataset metadata without changing its samples or training history."""
    dataset = _get_dataset_or_404(db, dataset_id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(dataset, field, value)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="A dataset with this name and version already exists.",
        ) from exc
    db.refresh(dataset)
    return DatasetRead.model_validate(dataset)


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: UUID,
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Delete a dataset and its samples only when no training run references it."""
    dataset = _get_dataset_or_404(db, dataset_id)
    run_count = int(
        db.scalar(
            select(func.count(TrainingRun.id)).where(
                TrainingRun.dataset_id == dataset.id
            )
        )
        or 0
    )
    if run_count:
        raise HTTPException(
            status_code=409,
            detail="Dataset cannot be deleted because training runs reference it.",
        )
    db.delete(dataset)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/datasets/{dataset_id}/samples",
    response_model=list[DatasetSampleRead],
    status_code=status.HTTP_201_CREATED,
)
def create_dataset_samples(
    dataset_id: UUID,
    payload: DatasetSamplesBulkCreate,
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Add up to 500 labeled samples atomically to a dataset."""
    _get_dataset_or_404(db, dataset_id)
    categories = _active_category_map(
        db, {item.category_code for item in payload.items}
    )
    report_ids = {item.report_id for item in payload.items if item.report_id is not None}
    if report_ids:
        existing_report_ids = set(
            db.scalars(select(Report.id).where(Report.id.in_(report_ids))).all()
        )
        missing_reports = report_ids - existing_report_ids
        if missing_reports:
            raise HTTPException(status_code=422, detail="One or more report_id values do not exist.")

    samples = [
        DatasetSample(
            dataset_id=dataset_id,
            report_id=item.report_id,
            raw_text=item.raw_text,
            processed_text=item.processed_text,
            category_id=categories[item.category_code].id,
            split=item.split,
            is_verified=item.is_verified,
        )
        for item in payload.items
    ]
    db.add_all(samples)
    db.commit()
    for sample in samples:
        db.refresh(sample)
    return [
        _sample_read(sample, categories[item.category_code])
        for sample, item in zip(samples, payload.items, strict=True)
    ]


@router.get("/datasets/{dataset_id}/samples", response_model=DatasetSamplePage)
def list_dataset_samples(
    dataset_id: UUID,
    split: Literal["TRAIN", "VALIDATION", "TEST"] | None = None,
    category_code: str | None = Query(default=None, min_length=3, max_length=3),
    is_verified: bool | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Filter and paginate labeled samples within one dataset."""
    _get_dataset_or_404(db, dataset_id)
    statement = (
        select(DatasetSample, IncidentCategory)
        .join(IncidentCategory, IncidentCategory.id == DatasetSample.category_id)
        .where(DatasetSample.dataset_id == dataset_id)
    )
    count_statement = select(func.count(DatasetSample.id)).where(
        DatasetSample.dataset_id == dataset_id
    )
    if split:
        statement = statement.where(DatasetSample.split == split)
        count_statement = count_statement.where(DatasetSample.split == split)
    if category_code:
        normalized_code = category_code.strip().upper()
        category = db.scalar(
            select(IncidentCategory).where(IncidentCategory.code == normalized_code)
        )
        if category is None:
            raise HTTPException(status_code=422, detail="Unknown incident category code.")
        statement = statement.where(DatasetSample.category_id == category.id)
        count_statement = count_statement.where(DatasetSample.category_id == category.id)
    if is_verified is not None:
        statement = statement.where(DatasetSample.is_verified.is_(is_verified))
        count_statement = count_statement.where(DatasetSample.is_verified.is_(is_verified))

    total = int(db.scalar(count_statement) or 0)
    rows = db.execute(
        statement.order_by(DatasetSample.created_at.desc(), DatasetSample.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return DatasetSamplePage(
        items=[_sample_read(sample, category) for sample, category in rows],
        total=total,
        page=page,
        page_size=page_size,
        pages=_page_count(total, page_size),
    )


@router.patch("/dataset-samples/{sample_id}", response_model=DatasetSampleRead)
def update_dataset_sample(
    sample_id: UUID,
    payload: DatasetSampleUpdate = Body(...),
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Correct sample text, label, split, verification flag or linked report."""
    sample = _get_sample_or_404(db, sample_id)
    values = payload.model_dump(exclude_unset=True)

    if "category_code" in values:
        category = _active_category_map(db, {values.pop("category_code")})[
            payload.category_code
        ]
        sample.category_id = category.id
    else:
        category = db.get(IncidentCategory, sample.category_id)

    if "report_id" in values and values["report_id"] is not None:
        report_exists = db.scalar(
            select(Report.id).where(Report.id == values["report_id"])
        )
        if report_exists is None:
            raise HTTPException(status_code=422, detail="report_id does not exist.")

    for field, value in values.items():
        setattr(sample, field, value)

    db.commit()
    db.refresh(sample)
    if "category_code" not in payload.model_fields_set:
        category = db.get(IncidentCategory, sample.category_id)
    return _sample_read(sample, category)


@router.delete("/dataset-samples/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset_sample(
    sample_id: UUID,
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    sample = _get_sample_or_404(db, sample_id)
    db.delete(sample)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/training-runs", response_model=TrainingRunRead, status_code=status.HTTP_201_CREATED)
def create_training_run(
    payload: TrainingRunCreate,
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Register a pending training job for a dataset with at least one TRAIN sample."""
    _get_dataset_or_404(db, payload.dataset_id)
    train_count = int(
        db.scalar(
            select(func.count(DatasetSample.id)).where(
                DatasetSample.dataset_id == payload.dataset_id,
                DatasetSample.split == "TRAIN",
            )
        )
        or 0
    )
    if train_count == 0:
        raise HTTPException(
            status_code=422,
            detail="The dataset must contain at least one TRAIN sample before training.",
        )
    run = TrainingRun(
        dataset_id=payload.dataset_id,
        algorithm=payload.algorithm,
        representation=payload.representation,
        hyperparameters=payload.hyperparameters,
        imbalance_method=payload.imbalance_method,
        status="PENDING",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return TrainingRunRead.model_validate(run)


@router.get("/training-runs", response_model=TrainingRunPage)
def list_training_runs(
    dataset_id: UUID | None = None,
    run_status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"] | None = Query(
        default=None, alias="status"
    ),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """List and filter registered training runs."""
    statement = select(TrainingRun).options(selectinload(TrainingRun.metrics))
    count_statement = select(func.count(TrainingRun.id))
    if dataset_id is not None:
        statement = statement.where(TrainingRun.dataset_id == dataset_id)
        count_statement = count_statement.where(TrainingRun.dataset_id == dataset_id)
    if run_status is not None:
        statement = statement.where(TrainingRun.status == run_status)
        count_statement = count_statement.where(TrainingRun.status == run_status)
    total = int(db.scalar(count_statement) or 0)
    rows = db.scalars(
        statement.order_by(TrainingRun.created_at.desc(), TrainingRun.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return TrainingRunPage(
        items=[TrainingRunRead.model_validate(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
        pages=_page_count(total, page_size),
    )


@router.get("/training-runs/{run_id}", response_model=TrainingRunRead)
def get_training_run(
    run_id: UUID,
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    run = _get_run_or_404(db, run_id, with_metrics=True)
    return TrainingRunRead.model_validate(run)


@router.patch("/training-runs/{run_id}/status", response_model=TrainingRunRead)
def update_training_run_status(
    run_id: UUID,
    payload: TrainingRunStatusUpdate,
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Advance a training run through PENDING -> RUNNING -> COMPLETED/FAILED."""
    run = _get_run_or_404(db, run_id, with_metrics=True)
    allowed_transitions = {
        "PENDING": {"RUNNING", "FAILED"},
        "RUNNING": {"COMPLETED", "FAILED"},
        "COMPLETED": set(),
        "FAILED": set(),
    }
    if payload.status not in allowed_transitions[run.status]:
        raise HTTPException(
            status_code=409,
            detail=f"Invalid training status transition: {run.status} -> {payload.status}.",
        )

    values = payload.model_dump(exclude_unset=True)
    if "model_version_id" in values and values["model_version_id"] is not None:
        if db.get(ModelVersion, values["model_version_id"]) is None:
            raise HTTPException(status_code=422, detail="model_version_id does not exist.")
    run.status = payload.status
    if payload.status == "RUNNING" and run.started_at is None:
        run.started_at = datetime.now(timezone.utc)
    if payload.status in {"COMPLETED", "FAILED"}:
        run.finished_at = datetime.now(timezone.utc)
    if "model_version_id" in values:
        run.model_version_id = values["model_version_id"]
    if "artifact_path" in values:
        run.artifact_path = values["artifact_path"]

    db.commit()
    return TrainingRunRead.model_validate(_get_run_or_404(db, run.id, with_metrics=True))


@router.post(
    "/training-runs/{run_id}/metrics",
    response_model=list[TrainingMetricRead],
    status_code=status.HTTP_201_CREATED,
)
def create_training_metrics(
    run_id: UUID,
    payload: TrainingMetricsBulkCreate,
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """Persist metric values (0..1) after the training worker has started."""
    run = _get_run_or_404(db, run_id)
    if run.status not in {"RUNNING", "COMPLETED"}:
        raise HTTPException(
            status_code=409,
            detail="Metrics can only be recorded for RUNNING or COMPLETED training runs.",
        )

    keys = {(item.metric_name, item.class_label) for item in payload.items}
    existing = db.scalars(
        select(TrainingMetric).where(
            TrainingMetric.training_run_id == run_id,
            or_(
                *[
                    (TrainingMetric.metric_name == name)
                    & (TrainingMetric.class_label == label)
                    for name, label in keys
                ]
            ),
        )
    ).all()
    if existing:
        raise HTTPException(
            status_code=409,
            detail="One or more metrics already exist for this run and class label.",
        )

    metrics = [
        TrainingMetric(
            training_run_id=run_id,
            metric_name=item.metric_name,
            metric_value=item.metric_value,
            class_label=item.class_label,
            details=item.details,
        )
        for item in payload.items
    ]
    db.add_all(metrics)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Duplicate training metric.") from exc
    for metric in metrics:
        db.refresh(metric)
    return [TrainingMetricRead.model_validate(metric) for metric in metrics]


@router.get("/training-runs/{run_id}/metrics", response_model=list[TrainingMetricRead])
def list_training_metrics(
    run_id: UUID,
    _current_user: User = Depends(require_roles("ADMIN", "RECEIVER")),
    db: Session = Depends(get_db),
):
    """List run-level and per-category metrics in stable order."""
    _get_run_or_404(db, run_id)
    rows = db.scalars(
        select(TrainingMetric)
        .where(TrainingMetric.training_run_id == run_id)
        .order_by(TrainingMetric.metric_name.asc(), TrainingMetric.class_label.asc())
    ).all()
    return [TrainingMetricRead.model_validate(row) for row in rows]
