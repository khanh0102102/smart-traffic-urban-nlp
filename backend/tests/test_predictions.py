"""Prediction API tests using deterministic estimators with the real Phase 5 preprocessor."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

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
    IncidentCategory,
    ModelVersion,
    Prediction,
    PreprocessingRun,
    Report,
    User,
)
from app.database.session import get_db
from app.main import app
from app.services.prediction_service import get_model_predictor


class FakePredictor:
    def __init__(self, path: Path, model_key: str, output: dict):
        self.path = path
        self.path.write_bytes(b"deterministic-test-inference-bundle")
        self.key = model_key
        self.meta = {
            "selected_model": model_key,
            "selected_model_name": {"svm": "SVM", "naive_bayes": "Naive Bayes"}[model_key],
            "selection_metric": "validation_macro_f1",
            "selection_value": 0.83,
            "confidence_threshold": 0.60,
        }
        self.bundle = {
            "_inference": self.meta,
            "preprocessing": {
                "pipeline": "src.nlp_preprocessing.preprocess_vietnamese_text",
                "remove_stopwords": True,
            },
            "probability_calibrated": False if model_key == "svm" else None,
        }
        self.output = output

    def predict(self, texts, *, already_processed=False):
        assert already_processed is True
        assert len(texts) == 1
        return [dict(self.output, processed_text=texts[0])]


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def client(db_session, monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret_key", "c" * 64)
    monkeypatch.setattr(settings, "jwt_algorithm", "HS256")
    monkeypatch.setattr(settings, "jwt_issuer", "st-nlp-api")
    monkeypatch.setattr(settings, "jwt_audience", "st-nlp-client")
    monkeypatch.setattr(settings, "jwt_access_token_expire_minutes", 30)

    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db

    # Exercise the actual Phase 5 pipeline deterministically without depending
    # on Underthesea's native NLP model/data files in CI.
    from src.nlp_preprocessing import preprocessor
    monkeypatch.setattr(
        preprocessor,
        "segment_words",
        lambda value, tokenizer=None: "mặt đường có ổ_gà sâu gây nguy hiểm",
    )

    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def create_citizen_report(db_session, email="citizen@example.com"):
    citizen = User(
        email=email,
        full_name="Citizen",
        role="CITIZEN",
        google_sub=f"google-{uuid4()}",
    )
    category = IncidentCategory(
        code="R01",
        name="Đường/Cầu/Hạ tầng hư hỏng",
        slug=f"duong-ha-tang-{uuid4().hex[:8]}",
        sort_order=1,
        is_active=True,
    )
    db_session.add_all([citizen, category])
    db_session.commit()
    report = Report(
        tracking_code=f"RPT-TEST-{uuid4().hex[:8]}",
        user_id=citizen.id,
        raw_content="Mặt đường có ổ gà sâu, gây nguy hiểm cho người đi xe máy.",
        address="Đường Nguyễn Huệ",
        latitude=12.6816000,
        longitude=108.0372000,
        status="RECEIVED",
    )
    db_session.add(report)
    db_session.commit()
    db_session.refresh(report)
    return citizen, report


def auth_headers(user: User) -> dict[str, str]:
    token, _ = create_access_token(user.id, user.role)
    return {"Authorization": f"Bearer {token}"}


def naive_bayes_output():
    return {
        "raw_content": "",
        "model_key": "naive_bayes",
        "model_name": "Naive Bayes",
        "predicted_label_code": "R01",
        "predicted_label_name": "Đường/Cầu/Hạ tầng hư hỏng",
        "confidence": 0.87,
        "confidence_type": "model_score",
        "probability_calibrated": None,
        "decision_margin": None,
        "class_probabilities": {"R01": 0.87, "R02": 0.13},
        "has_known_features": True,
        "needs_manual_review": False,
        "warning_reasons": [],
    }


def override_predictor(tmp_path, model_key, output):
    fake = FakePredictor(
        tmp_path / f"{model_key}-bundle.joblib",
        model_key,
        output,
    )
    app.dependency_overrides[get_model_predictor] = lambda: fake
    return fake


def test_prediction_persists_preprocessing_and_model_output(client, db_session, tmp_path):
    citizen, report = create_citizen_report(db_session)
    override_predictor(tmp_path, "naive_bayes", naive_bayes_output())

    response = client.post(
        "/api/v1/predictions",
        headers=auth_headers(citizen),
        json={"report_id": str(report.id)},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["predicted_category"]["code"] == "R01"
    assert payload["confidence"] == "0.87000"
    assert payload["confidence_type"] == "model_score"
    assert payload["needs_review"] is False
    assert payload["preprocessing"]["pipeline_version"] == "phase5-v1"
    assert payload["preprocessing"]["cleaned_text"] == report.raw_content
    assert payload["preprocessing"]["segmented_text"]
    assert payload["preprocessing"]["stopwords_removed_text"]

    preprocessing = db_session.scalar(select(PreprocessingRun))
    prediction = db_session.scalar(select(Prediction))
    model_version = db_session.scalar(select(ModelVersion))
    assert preprocessing is not None
    assert prediction is not None
    assert prediction.preprocessing_run_id == preprocessing.id
    assert prediction.class_probabilities == {"R01": 0.87, "R02": 0.13}
    assert model_version is not None
    assert model_version.algorithm == "NAIVE_BAYES"
    assert model_version.representation == "TF_IDF"


def test_uncalibrated_svm_is_saved_without_fabricating_confidence(
    client, db_session, tmp_path
):
    citizen, report = create_citizen_report(db_session, "svm@example.com")
    output = {
        "raw_content": "",
        "model_key": "svm",
        "model_name": "SVM",
        "predicted_label_code": "R01",
        "predicted_label_name": "Đường/Cầu/Hạ tầng hư hỏng",
        "confidence": None,
        "confidence_type": "unavailable",
        "probability_calibrated": False,
        "decision_margin": 1.245,
        "class_probabilities": None,
        "has_known_features": True,
        "needs_manual_review": True,
        "warning_reasons": ["SVM calibration unavailable."],
    }
    override_predictor(tmp_path, "svm", output)

    response = client.post(
        "/api/v1/predictions",
        headers=auth_headers(citizen),
        json={"report_id": str(report.id)},
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["confidence"] is None
    assert result["confidence_type"] == "unavailable"
    assert result["decision_margin"] == "1.245000"
    assert result["probability_calibrated"] is False
    assert result["needs_review"] is True

    saved = db_session.scalar(select(Prediction))
    assert saved is not None
    assert saved.confidence is None
    assert saved.decision_margin is not None


def test_citizen_cannot_predict_another_users_report(client, db_session, tmp_path):
    owner, report = create_citizen_report(db_session, "owner@example.com")
    intruder = User(
        email="intruder@example.com",
        full_name="Intruder",
        role="CITIZEN",
        google_sub=f"google-{uuid4()}",
    )
    db_session.add(intruder)
    db_session.commit()
    override_predictor(tmp_path, "naive_bayes", naive_bayes_output())

    denied = client.post(
        "/api/v1/predictions",
        headers=auth_headers(intruder),
        json={"report_id": str(report.id)},
    )
    assert denied.status_code == 404


def test_prediction_history_is_available_to_owner_and_staff(client, db_session, tmp_path):
    citizen, report = create_citizen_report(db_session, "history@example.com")
    override_predictor(tmp_path, "naive_bayes", naive_bayes_output())

    created = client.post(
        "/api/v1/predictions",
        headers=auth_headers(citizen),
        json={"report_id": str(report.id)},
    )
    assert created.status_code == 201, created.text

    history = client.get(
        f"/api/v1/predictions/report/{report.id}",
        headers=auth_headers(citizen),
    )
    assert history.status_code == 200
    assert len(history.json()) == 1

    receiver = User(email="receiver@example.com", full_name="Receiver", role="RECEIVER")
    db_session.add(receiver)
    db_session.commit()
    staff_history = client.get(
        f"/api/v1/predictions/report/{report.id}",
        headers=auth_headers(receiver),
    )
    assert staff_history.status_code == 200
    assert len(staff_history.json()) == 1


def test_prediction_does_not_overwrite_staff_final_category(client, db_session, tmp_path):
    citizen, report = create_citizen_report(db_session, "final@example.com")
    category = db_session.scalar(select(IncidentCategory).where(IncidentCategory.code == "R01"))
    report.final_category = category
    db_session.commit()
    override_predictor(tmp_path, "naive_bayes", naive_bayes_output())

    response = client.post(
        "/api/v1/predictions",
        headers=auth_headers(citizen),
        json={"report_id": str(report.id)},
    )
    assert response.status_code == 201, response.text
    db_session.refresh(report)
    assert report.final_category_id == category.id


def test_prediction_endpoint_requires_authentication(client, db_session):
    citizen, report = create_citizen_report(db_session, "anonymous@example.com")
    response = client.post("/api/v1/predictions", json={"report_id": str(report.id)})
    assert response.status_code == 401
