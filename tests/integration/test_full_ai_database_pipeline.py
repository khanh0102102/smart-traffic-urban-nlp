"""Cross-module integration test for AI prediction → review → dataset → training."""
from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.security import create_access_token
from app.database import models  # noqa: F401
from app.database.base import Base
from app.database.models import (
    DatasetSample, IncidentCategory, ModelVersion, Prediction, PredictionReview,
    PreprocessingRun, Report, TrainingMetric, TrainingRun, User,
)
from app.database.seed import CATEGORIES
from app.database.session import get_db
from app.main import app
from app.services.prediction_service import get_model_predictor


class DeterministicPredictor:
    """Predictable model boundary; real service, ORM and API persistence stay enabled."""
    def __init__(self, path: Path):
        self.path = path
        self.path.write_bytes(b"phase24-deterministic-model-bundle")
        self.key = "naive_bayes"
        self.meta = {
            "selected_model": "naive_bayes",
            "selected_model_name": "Naive Bayes",
            "selection_metric": "validation_macro_f1",
            "selection_value": 0.84,
            "confidence_threshold": 0.60,
        }
        self.bundle = {
            "_inference": self.meta,
            "preprocessing": {"remove_stopwords": True},
            "probability_calibrated": None,
        }
    def predict(self, texts, *, already_processed=False):
        assert already_processed is True
        assert len(texts) == 1
        return [{
            "raw_content": "",
            "processed_text": texts[0],
            "model_key": "naive_bayes",
            "model_name": "Naive Bayes",
            "predicted_label_code": "R01",
            "predicted_label_name": "Đường/Cầu/Hạ tầng hư hỏng",
            "confidence": 0.91,
            "confidence_type": "model_score",
            "probability_calibrated": None,
            "decision_margin": None,
            "class_probabilities": {"R01": 0.91, "R02": 0.09},
            "has_known_features": True,
            "needs_manual_review": False,
            "warning_reasons": [],
        }]


@pytest.fixture
def database():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    categories = [
        IncidentCategory(
            code=code, name=name, slug=slug, description=description,
            sort_order=order, is_active=True,
        )
        for code, name, slug, description, order in CATEGORIES
    ]
    citizen = User(
        email=f"pipeline-citizen-{uuid4().hex[:8]}@example.com",
        full_name="Pipeline Citizen", role="CITIZEN",
        google_sub=f"google-{uuid4()}",
    )
    receiver = User(
        email=f"pipeline-receiver-{uuid4().hex[:8]}@example.com",
        full_name="Pipeline Receiver", role="RECEIVER",
    )
    session.add_all([*categories, citizen, receiver])
    session.commit()
    try:
        yield session, citizen, receiver
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def pipeline_client(database, tmp_path, monkeypatch):
    session, citizen, receiver = database
    monkeypatch.setattr(settings, "jwt_secret_key", "phase24-pipeline-secret-" * 2)
    monkeypatch.setattr(settings, "jwt_algorithm", "HS256")
    monkeypatch.setattr(settings, "jwt_issuer", "st-nlp-api")
    monkeypatch.setattr(settings, "jwt_audience", "st-nlp-client")
    monkeypatch.setattr(settings, "jwt_access_token_expire_minutes", 30)
    def override_db():
        yield session
    predictor = DeterministicPredictor(tmp_path / "deterministic-bundle.joblib")
    app.dependency_overrides.clear()
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_model_predictor] = lambda: predictor
    from src.nlp_preprocessing import preprocessor
    monkeypatch.setattr(
        preprocessor, "segment_words",
        lambda _value, tokenizer=None: "duong o ga mat duong hong nguy hiem",
    )
    with TestClient(app) as client:
        yield client, session, citizen, receiver
    app.dependency_overrides.clear()


def token_headers(user):
    token, _ = create_access_token(user.id, user.role)
    return {"Authorization": f"Bearer {token}"}


def test_full_ai_review_dataset_and_training_lineage(pipeline_client):
    client, db, citizen, receiver = pipeline_client
    created_report = client.post(
        "/api/v1/reports",
        headers=token_headers(citizen),
        json={
            "raw_content": "Đường Nguyễn Huệ có ổ gà sâu, gây nguy hiểm.",
            "address": "Đường Nguyễn Huệ",
        },
    )
    assert created_report.status_code == 201, created_report.text
    report_id = created_report.json()["id"]

    prediction_response = client.post(
        "/api/v1/predictions",
        headers=token_headers(citizen),
        json={"report_id": report_id},
    )
    assert prediction_response.status_code == 201, prediction_response.text
    prediction_payload = prediction_response.json()
    assert prediction_payload["predicted_category"]["code"] == "R01"
    assert prediction_payload["preprocessing"]["stopwords_removed_text"]
    assert prediction_payload["model_version"]["algorithm"] == "NAIVE_BAYES"

    model_version = db.scalar(select(ModelVersion))
    prediction = db.scalar(select(Prediction))
    preprocessing = db.scalar(select(PreprocessingRun))
    assert model_version is not None and prediction is not None and preprocessing is not None
    assert prediction.preprocessing_run_id == preprocessing.id
    assert prediction.model_version_id == model_version.id

    dataset_response = client.post(
        "/api/v1/ml/datasets",
        headers=token_headers(receiver),
        json={
            "name": "Phase 24 reviewed reports",
            "version": "v1",
            "source": "staff-reviewed-reports",
        },
    )
    assert dataset_response.status_code == 201, dataset_response.text
    dataset_id = dataset_response.json()["id"]

    review_response = client.post(
        "/api/v1/reviews",
        headers=token_headers(receiver),
        json={
            "prediction_id": str(prediction.id),
            "confirmed_category_code": "R01",
            "notes": "Đã đối chiếu nội dung phản ánh.",
            "dataset_id": dataset_id,
            "dataset_split": "TRAIN",
        },
    )
    assert review_response.status_code == 201, review_response.text

    sample = db.scalar(select(DatasetSample).where(DatasetSample.dataset_id == UUID(dataset_id)))
    review = db.scalar(select(PredictionReview))
    report = db.get(Report, UUID(report_id))
    assert sample is not None and sample.is_verified is True
    assert sample.raw_text == report.raw_content
    assert sample.processed_text == preprocessing.stopwords_removed_text
    assert sample.category_id == report.final_category_id
    assert review is not None and review.action == "CONFIRMED"

    training_response = client.post(
        "/api/v1/ml/training-runs",
        headers=token_headers(receiver),
        json={
            "dataset_id": dataset_id,
            "algorithm": "NAIVE_BAYES",
            "representation": "TF_IDF",
        },
    )
    assert training_response.status_code == 201, training_response.text
    run_id = training_response.json()["id"]
    running = client.patch(
        f"/api/v1/ml/training-runs/{run_id}/status",
        headers=token_headers(receiver),
        json={"status": "RUNNING"},
    )
    assert running.status_code == 200

    metrics_response = client.post(
        f"/api/v1/ml/training-runs/{run_id}/metrics",
        headers=token_headers(receiver),
        json={
            "items": [
                {"metric_name": "accuracy", "metric_value": "0.91"},
                {"metric_name": "macro_f1", "metric_value": "0.84"},
            ]
        },
    )
    assert metrics_response.status_code == 201, metrics_response.text

    completed = client.patch(
        f"/api/v1/ml/training-runs/{run_id}/status",
        headers=token_headers(receiver),
        json={
            "status": "COMPLETED",
            "model_version_id": str(model_version.id),
            "artifact_path": model_version.artifact_path,
        },
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["status"] == "COMPLETED"
    assert completed.json()["model_version_id"] == str(model_version.id)
    assert db.scalar(select(TrainingRun).where(TrainingRun.id == run_id)) is not None
    assert len(db.scalars(select(TrainingMetric)).all()) == 2

    review_history = client.get(
        f"/api/v1/reviews/report/{report_id}",
        headers=token_headers(citizen),
    )
    prediction_history = client.get(
        f"/api/v1/predictions/report/{report_id}",
        headers=token_headers(citizen),
    )
    assert review_history.status_code == 200 and len(review_history.json()) == 1
    assert prediction_history.status_code == 200 and len(prediction_history.json()) == 1
