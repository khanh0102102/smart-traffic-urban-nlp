"""Prediction review API tests for label decisions, permissions and history."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.security import create_access_token
from app.database import models  # noqa: F401
from app.database.base import Base
from app.database.models import (
    Dataset,
    DatasetSample,
    IncidentCategory,
    ModelVersion,
    Prediction,
    PredictionReview,
    PreprocessingRun,
    Report,
    User,
)
from app.database.session import get_db
from app.main import app


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
    monkeypatch.setattr(settings, "jwt_secret_key", "d" * 64)
    monkeypatch.setattr(settings, "jwt_algorithm", "HS256")
    monkeypatch.setattr(settings, "jwt_issuer", "st-nlp-api")
    monkeypatch.setattr(settings, "jwt_audience", "st-nlp-client")
    monkeypatch.setattr(settings, "jwt_access_token_expire_minutes", 30)

    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def auth_headers(user: User) -> dict[str, str]:
    token, _ = create_access_token(user.id, user.role)
    return {"Authorization": f"Bearer {token}"}


def setup_prediction(db_session):
    owner = User(
        email=f"owner-{uuid4().hex[:8]}@example.com",
        full_name="Report Owner",
        role="CITIZEN",
        google_sub=f"google-{uuid4()}",
    )
    receiver = User(
        email=f"receiver-{uuid4().hex[:8]}@example.com",
        full_name="Receiving Officer",
        role="RECEIVER",
    )
    admin = User(
        email=f"admin-{uuid4().hex[:8]}@example.com",
        full_name="Administrator",
        role="ADMIN",
    )
    intruder = User(
        email=f"intruder-{uuid4().hex[:8]}@example.com",
        full_name="Other Citizen",
        role="CITIZEN",
        google_sub=f"google-{uuid4()}",
    )
    category_original = IncidentCategory(
        code="R01",
        name="Đường/Cầu/Hạ tầng hư hỏng",
        slug=f"duong-{uuid4().hex[:8]}",
        sort_order=1,
        is_active=True,
    )
    category_corrected = IncidentCategory(
        code="R02",
        name="Điện lực/Chiếu sáng",
        slug=f"dien-{uuid4().hex[:8]}",
        sort_order=2,
        is_active=True,
    )
    inactive_category = IncidentCategory(
        code="R03",
        name="Cống/Hố ga/Thoát nước",
        slug=f"cong-{uuid4().hex[:8]}",
        sort_order=3,
        is_active=False,
    )
    model_version = ModelVersion(
        name="ST-NLP Naive Bayes",
        version=f"test-{uuid4().hex[:8]}",
        algorithm="NAIVE_BAYES",
        representation="TF_IDF",
        artifact_path="test/bundle.joblib",
        parameters={"fixture": True},
        is_active=True,
    )
    db_session.add_all([
        owner,
        receiver,
        admin,
        intruder,
        category_original,
        category_corrected,
        inactive_category,
        model_version,
    ])
    db_session.commit()

    report = Report(
        tracking_code=f"RPT-TEST-{uuid4().hex[:8]}",
        user_id=owner.id,
        raw_content="Mặt đường có ổ gà sâu gây nguy hiểm.",
        status="RECEIVED",
    )
    db_session.add(report)
    db_session.commit()

    prediction = Prediction(
        report_id=report.id,
        model_version_id=model_version.id,
        predicted_category_id=category_original.id,
        confidence=0.87,
        confidence_type="model_score",
        probability_calibrated=None,
        needs_review=False,
    )
    db_session.add(prediction)
    db_session.commit()
    db_session.refresh(prediction)
    return {
        "owner": owner,
        "receiver": receiver,
        "admin": admin,
        "intruder": intruder,
        "category_original": category_original,
        "category_corrected": category_corrected,
        "inactive_category": inactive_category,
        "report": report,
        "prediction": prediction,
    }


def test_staff_confirms_predicted_label_and_persists_review(client, db_session):
    data = setup_prediction(db_session)
    response = client.post(
        "/api/v1/reviews",
        headers=auth_headers(data["receiver"]),
        json={
            "prediction_id": str(data["prediction"].id),
            "confirmed_category_code": "r01",
            "notes": "  Nội dung phản ánh phù hợp với nhãn dự đoán.  ",
        },
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["action"] == "CONFIRMED"
    assert result["predicted_category"]["code"] == "R01"
    assert result["confirmed_category"]["code"] == "R01"
    assert result["reviewer"]["id"] == str(data["receiver"].id)
    assert result["notes"] == "Nội dung phản ánh phù hợp với nhãn dự đoán."

    saved = db_session.scalar(select(PredictionReview))
    db_session.refresh(data["report"])
    assert saved is not None
    assert saved.reviewer_user_id == data["receiver"].id
    assert saved.confirmed_category_id == data["category_original"].id
    assert data["report"].final_category_id == data["category_original"].id


def test_staff_corrects_label_without_mutating_original_prediction(client, db_session):
    data = setup_prediction(db_session)
    response = client.post(
        "/api/v1/reviews",
        headers=auth_headers(data["admin"]),
        json={
            "prediction_id": str(data["prediction"].id),
            "confirmed_category_code": "R02",
        },
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["action"] == "CORRECTED"
    assert result["predicted_category"]["code"] == "R01"
    assert result["confirmed_category"]["code"] == "R02"
    assert result["notes"] is None

    db_session.refresh(data["prediction"])
    db_session.refresh(data["report"])
    assert data["prediction"].predicted_category_id == data["category_original"].id
    assert data["report"].final_category_id == data["category_corrected"].id


def test_each_review_is_appended_and_latest_decision_sets_final_category(client, db_session):
    data = setup_prediction(db_session)
    first = client.post(
        "/api/v1/reviews",
        headers=auth_headers(data["receiver"]),
        json={
            "prediction_id": str(data["prediction"].id),
            "confirmed_category_code": "R01",
        },
    )
    second = client.post(
        "/api/v1/reviews",
        headers=auth_headers(data["admin"]),
        json={
            "prediction_id": str(data["prediction"].id),
            "confirmed_category_code": "R02",
            "notes": "Sau khi đối chiếu, điều chỉnh sang nhóm điện chiếu sáng.",
        },
    )
    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text

    rows = db_session.scalars(
        select(PredictionReview).where(
            PredictionReview.prediction_id == data["prediction"].id
        )
    ).all()
    assert len(rows) == 2
    db_session.refresh(data["report"])
    assert data["report"].final_category_id == data["category_corrected"].id

    history = client.get(
        f"/api/v1/reviews/report/{data['report'].id}",
        headers=auth_headers(data["owner"]),
    )
    assert history.status_code == 200
    assert len(history.json()) == 2
    assert {row["confirmed_category"]["code"] for row in history.json()} == {"R01", "R02"}


def test_citizen_cannot_create_review_but_owner_can_read_history(client, db_session):
    data = setup_prediction(db_session)
    denied = client.post(
        "/api/v1/reviews",
        headers=auth_headers(data["owner"]),
        json={
            "prediction_id": str(data["prediction"].id),
            "confirmed_category_code": "R01",
        },
    )
    assert denied.status_code == 403

    review = PredictionReview(
        prediction_id=data["prediction"].id,
        reviewer_user_id=data["receiver"].id,
        confirmed_category_id=data["category_original"].id,
        action="CONFIRMED",
    )
    db_session.add(review)
    db_session.commit()

    visible = client.get(
        f"/api/v1/reviews/prediction/{data['prediction'].id}",
        headers=auth_headers(data["owner"]),
    )
    assert visible.status_code == 200
    assert len(visible.json()) == 1


def test_citizen_cannot_read_another_users_review(client, db_session):
    data = setup_prediction(db_session)
    review = PredictionReview(
        prediction_id=data["prediction"].id,
        reviewer_user_id=data["receiver"].id,
        confirmed_category_id=data["category_original"].id,
        action="CONFIRMED",
    )
    db_session.add(review)
    db_session.commit()

    response = client.get(
        f"/api/v1/reviews/{review.id}",
        headers=auth_headers(data["intruder"]),
    )
    assert response.status_code == 404


def test_inactive_or_unknown_categories_are_rejected(client, db_session):
    data = setup_prediction(db_session)
    for code in ("R03", "R99"):
        response = client.post(
            "/api/v1/reviews",
            headers=auth_headers(data["receiver"]),
            json={
                "prediction_id": str(data["prediction"].id),
                "confirmed_category_code": code,
            },
        )
        assert response.status_code == 422

    assert db_session.scalar(select(PredictionReview)) is None



def test_review_materializes_verified_dataset_sample_and_keeps_it_current(client, db_session):
    data = setup_prediction(db_session)
    dataset = Dataset(
        name="Reviewed urban reports",
        version="v1",
        description="Ground-truth reports accepted by staff.",
        source="staff-review",
        created_by=data["receiver"].id,
    )
    preprocessing = PreprocessingRun(
        report_id=data["report"].id,
        pipeline_version="phase5-v1",
        cleaned_text=data["report"].raw_content,
        normalized_text="mat duong co o ga sau gay nguy hiem",
        segmented_text="mặt đường có ổ_gà sâu gây nguy hiểm",
        stopwords_removed_text="mặt đường ổ_gà sâu gây nguy hiểm",
        processing_time_ms=8,
    )
    db_session.add_all([dataset, preprocessing])
    db_session.flush()
    data["prediction"].preprocessing_run_id = preprocessing.id
    db_session.commit()

    first = client.post(
        "/api/v1/reviews",
        headers=auth_headers(data["receiver"]),
        json={
            "prediction_id": str(data["prediction"].id),
            "confirmed_category_code": "R01",
            "dataset_id": str(dataset.id),
            "dataset_split": "TRAIN",
        },
    )
    assert first.status_code == 201, first.text
    sample = db_session.scalar(
        select(DatasetSample).where(
            DatasetSample.dataset_id == dataset.id,
            DatasetSample.report_id == data["report"].id,
        )
    )
    assert sample is not None
    assert sample.category_id == data["category_original"].id
    assert sample.is_verified is True
    assert sample.split == "TRAIN"
    assert sample.processed_text == preprocessing.stopwords_removed_text

    second = client.post(
        "/api/v1/reviews",
        headers=auth_headers(data["admin"]),
        json={
            "prediction_id": str(data["prediction"].id),
            "confirmed_category_code": "R02",
        },
    )
    assert second.status_code == 201, second.text
    db_session.refresh(sample)
    assert sample.category_id == data["category_corrected"].id
    assert sample.is_verified is True
    assert sample.split == "TRAIN"
    assert db_session.scalar(
        select(func.count(DatasetSample.id)).where(
            DatasetSample.dataset_id == dataset.id,
            DatasetSample.report_id == data["report"].id,
        )
    ) == 1

    history = db_session.scalars(
        select(PredictionReview).where(
            PredictionReview.prediction_id == data["prediction"].id
        )
    ).all()
    assert len(history) == 2


def test_review_dataset_id_and_split_must_be_provided_together(client, db_session):
    data = setup_prediction(db_session)
    response = client.post(
        "/api/v1/reviews",
        headers=auth_headers(data["receiver"]),
        json={
            "prediction_id": str(data["prediction"].id),
            "confirmed_category_code": "R01",
            "dataset_id": str(uuid4()),
        },
    )
    assert response.status_code == 422
    assert db_session.scalar(select(PredictionReview)) is None


def test_review_endpoint_requires_authentication(client, db_session):
    data = setup_prediction(db_session)
    response = client.post(
        "/api/v1/reviews",
        json={
            "prediction_id": str(data["prediction"].id),
            "confirmed_category_code": "R01",
        },
    )
    assert response.status_code == 401


def test_review_requires_valid_payload(client, db_session):
    data = setup_prediction(db_session)
    response = client.post(
        "/api/v1/reviews",
        headers=auth_headers(data["receiver"]),
        json={
            "prediction_id": str(data["prediction"].id),
            "confirmed_category_code": "TOOLONG",
        },
    )
    assert response.status_code == 422


