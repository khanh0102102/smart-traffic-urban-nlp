"""ML management API tests for datasets, samples, training runs and metrics."""
from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.security import create_access_token
from app.database import models  # noqa: F401
from app.database.base import Base
from app.database.models import IncidentCategory, ModelVersion, User
from app.database.seed import CATEGORIES
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
    monkeypatch.setattr(settings, "jwt_secret_key", "f" * 64)
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


def setup_users_and_categories(db_session):
    staff = User(
        email=f"staff-{uuid4().hex[:8]}@example.com",
        full_name="ML Manager",
        role="ADMIN",
    )
    receiver = User(
        email=f"receiver-{uuid4().hex[:8]}@example.com",
        full_name="Data Receiver",
        role="RECEIVER",
    )
    citizen = User(
        email=f"citizen-{uuid4().hex[:8]}@example.com",
        full_name="Citizen",
        role="CITIZEN",
        google_sub=f"google-{uuid4()}",
    )
    categories = [
        IncidentCategory(
            code=code,
            name=name,
            slug=slug,
            description=description,
            sort_order=order,
            is_active=True,
        )
        for code, name, slug, description, order in CATEGORIES
    ]
    inactive = IncidentCategory(
        code="R99",
        name="Inactive test category",
        slug=f"inactive-{uuid4().hex[:8]}",
        sort_order=99,
        is_active=False,
    )
    db_session.add_all([staff, receiver, citizen, *categories, inactive])
    db_session.commit()
    return {
        "staff": staff,
        "receiver": receiver,
        "citizen": citizen,
        "categories": {row.code: row for row in categories},
    }


def create_dataset(client, headers, *, name="Urban reports", version="v1"):
    return client.post(
        "/api/v1/ml/datasets",
        headers=headers,
        json={
            "name": name,
            "version": version,
            "description": "Dataset for urban incident classification.",
            "source": "staff-reviewed-reports",
        },
    )


def add_samples(client, headers, dataset_id):
    return client.post(
        f"/api/v1/ml/datasets/{dataset_id}/samples",
        headers=headers,
        json={
            "items": [
                {
                    "raw_text": "Mặt đường có ổ gà sâu.",
                    "processed_text": "mặt đường ổ gà sâu",
                    "category_code": "r01",
                    "split": "TRAIN",
                    "is_verified": True,
                },
                {
                    "raw_text": "Đèn đường bị mất điện.",
                    "category_code": "R02",
                    "split": "VALIDATION",
                },
                {
                    "raw_text": "Nắp cống bị vỡ.",
                    "category_code": "R03",
                    "split": "TEST",
                },
            ]
        },
    )


def test_dataset_crud_search_and_sample_statistics(client, db_session):
    data = setup_users_and_categories(db_session)
    headers = auth_headers(data["staff"])

    created = create_dataset(client, headers)
    assert created.status_code == 201, created.text
    dataset = created.json()
    assert dataset["created_by"] == str(data["staff"].id)

    duplicate = create_dataset(client, headers)
    assert duplicate.status_code == 409

    added = add_samples(client, headers, dataset["id"])
    assert added.status_code == 201, added.text
    assert len(added.json()) == 3
    assert added.json()[0]["category_code"] == "R01"

    detail = client.get(f"/api/v1/ml/datasets/{dataset['id']}", headers=headers)
    assert detail.status_code == 200
    detail_data = detail.json()
    assert detail_data["sample_count"] == 3
    assert detail_data["train_sample_count"] == 1
    assert detail_data["validation_sample_count"] == 1
    assert detail_data["test_sample_count"] == 1
    assert detail_data["verified_sample_count"] == 1

    searched = client.get(
        "/api/v1/ml/datasets?q=Urban&page=1&page_size=10",
        headers=headers,
    )
    assert searched.status_code == 200
    assert searched.json()["total"] == 1

    updated = client.patch(
        f"/api/v1/ml/datasets/{dataset['id']}",
        headers=headers,
        json={"description": "Updated dataset description."},
    )
    assert updated.status_code == 200
    assert updated.json()["description"] == "Updated dataset description."


def test_dataset_samples_filter_update_and_delete(client, db_session):
    data = setup_users_and_categories(db_session)
    headers = auth_headers(data["staff"])
    dataset = create_dataset(client, headers).json()
    added = add_samples(client, headers, dataset["id"])
    sample_id = added.json()[0]["id"]

    filtered = client.get(
        f"/api/v1/ml/datasets/{dataset['id']}/samples?split=TRAIN&is_verified=true",
        headers=headers,
    )
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1
    assert filtered.json()["items"][0]["category_code"] == "R01"

    corrected = client.patch(
        f"/api/v1/ml/dataset-samples/{sample_id}",
        headers=headers,
        json={
            "raw_text": "Đường có nhiều ổ gà gây nguy hiểm.",
            "category_code": "R02",
            "split": "VALIDATION",
            "is_verified": True,
        },
    )
    assert corrected.status_code == 200, corrected.text
    assert corrected.json()["category_code"] == "R02"
    assert corrected.json()["split"] == "VALIDATION"
    assert corrected.json()["is_verified"] is True

    deleted = client.delete(
        f"/api/v1/ml/dataset-samples/{sample_id}",
        headers=headers,
    )
    assert deleted.status_code == 204


def test_bulk_samples_reject_unknown_or_inactive_categories_atomically(client, db_session):
    data = setup_users_and_categories(db_session)
    headers = auth_headers(data["staff"])
    dataset = create_dataset(client, headers).json()

    response = client.post(
        f"/api/v1/ml/datasets/{dataset['id']}/samples",
        headers=headers,
        json={
            "items": [
                {"raw_text": "Mặt đường có ổ gà.", "category_code": "R01", "split": "TRAIN"},
                {"raw_text": "Nhãn không hoạt động.", "category_code": "R99", "split": "TRAIN"},
            ]
        },
    )
    assert response.status_code == 422
    listing = client.get(
        f"/api/v1/ml/datasets/{dataset['id']}/samples",
        headers=headers,
    )
    assert listing.status_code == 200
    assert listing.json()["total"] == 0


def test_training_run_lifecycle_and_metric_management(client, db_session):
    data = setup_users_and_categories(db_session)
    headers = auth_headers(data["staff"])
    dataset = create_dataset(client, headers).json()
    added = add_samples(client, headers, dataset["id"])
    assert added.status_code == 201

    created = client.post(
        "/api/v1/ml/training-runs",
        headers=headers,
        json={
            "dataset_id": dataset["id"],
            "algorithm": "SVM",
            "representation": "TF_IDF",
            "hyperparameters": {"C": 1.0, "kernel": "linear"},
            "imbalance_method": "class_weight",
        },
    )
    assert created.status_code == 201, created.text
    run_id = created.json()["id"]
    assert created.json()["status"] == "PENDING"

    pending_metrics = client.post(
        f"/api/v1/ml/training-runs/{run_id}/metrics",
        headers=headers,
        json={"items": [{"metric_name": "accuracy", "metric_value": "0.9"}]},
    )
    assert pending_metrics.status_code == 409

    started = client.patch(
        f"/api/v1/ml/training-runs/{run_id}/status",
        headers=headers,
        json={"status": "RUNNING"},
    )
    assert started.status_code == 200
    assert started.json()["started_at"] is not None

    metrics = client.post(
        f"/api/v1/ml/training-runs/{run_id}/metrics",
        headers=headers,
        json={
            "items": [
                {"metric_name": "accuracy", "metric_value": "0.930000"},
                {"metric_name": "macro_f1", "metric_value": "0.870000"},
                {"metric_name": "precision", "class_label": "r01", "metric_value": "0.91"},
            ]
        },
    )
    assert metrics.status_code == 201, metrics.text
    assert len(metrics.json()) == 3

    duplicate = client.post(
        f"/api/v1/ml/training-runs/{run_id}/metrics",
        headers=headers,
        json={"items": [{"metric_name": "accuracy", "metric_value": "0.95"}]},
    )
    assert duplicate.status_code == 409

    metric_list = client.get(
        f"/api/v1/ml/training-runs/{run_id}/metrics",
        headers=headers,
    )
    assert metric_list.status_code == 200
    assert len(metric_list.json()) == 3

    completed = client.patch(
        f"/api/v1/ml/training-runs/{run_id}/status",
        headers=headers,
        json={"status": "COMPLETED", "artifact_path": "ml/artifacts/svm/model.joblib"},
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "COMPLETED"
    assert completed.json()["finished_at"] is not None
    assert completed.json()["artifact_path"] == "ml/artifacts/svm/model.joblib"
    assert len(completed.json()["metrics"]) == 3

    invalid_transition = client.patch(
        f"/api/v1/ml/training-runs/{run_id}/status",
        headers=headers,
        json={"status": "RUNNING"},
    )
    assert invalid_transition.status_code == 409

    listed = client.get(
        "/api/v1/ml/training-runs?status=COMPLETED",
        headers=headers,
    )
    assert listed.status_code == 200
    assert listed.json()["total"] == 1



def test_training_run_requires_verified_train_sample(client, db_session):
    data = setup_users_and_categories(db_session)
    headers = auth_headers(data["staff"])
    dataset = create_dataset(client, headers).json()

    unverified = client.post(
        f"/api/v1/ml/datasets/{dataset['id']}/samples",
        headers=headers,
        json={
            "items": [
                {
                    "raw_text": "Phản ánh chưa được nhân viên xác minh.",
                    "category_code": "R01",
                    "split": "TRAIN",
                    "is_verified": False,
                }
            ]
        },
    )
    assert unverified.status_code == 201
    rejected = client.post(
        "/api/v1/ml/training-runs",
        headers=headers,
        json={
            "dataset_id": dataset["id"],
            "algorithm": "NAIVE_BAYES",
            "representation": "TF_IDF",
        },
    )
    assert rejected.status_code == 422
    assert "verified TRAIN" in rejected.json()["detail"]

    verified = client.post(
        f"/api/v1/ml/datasets/{dataset['id']}/samples",
        headers=headers,
        json={
            "items": [
                {
                    "raw_text": "Phản ánh đã được nhân viên xác minh.",
                    "category_code": "R01",
                    "split": "TRAIN",
                    "is_verified": True,
                }
            ]
        },
    )
    assert verified.status_code == 201
    accepted = client.post(
        "/api/v1/ml/training-runs",
        headers=headers,
        json={
            "dataset_id": dataset["id"],
            "algorithm": "NAIVE_BAYES",
            "representation": "TF_IDF",
        },
    )
    assert accepted.status_code == 201, accepted.text


def test_training_requires_train_samples_and_dataset_delete_is_protected(client, db_session):
    data = setup_users_and_categories(db_session)
    headers = auth_headers(data["staff"])
    dataset = create_dataset(client, headers).json()

    no_train = client.post(
        "/api/v1/ml/training-runs",
        headers=headers,
        json={
            "dataset_id": dataset["id"],
            "algorithm": "NAIVE_BAYES",
            "representation": "TF_IDF",
        },
    )
    assert no_train.status_code == 422

    add_samples(client, headers, dataset["id"])
    run = client.post(
        "/api/v1/ml/training-runs",
        headers=headers,
        json={
            "dataset_id": dataset["id"],
            "algorithm": "NAIVE_BAYES",
            "representation": "TF_IDF",
        },
    )
    assert run.status_code == 201

    blocked = client.delete(
        f"/api/v1/ml/datasets/{dataset['id']}",
        headers=headers,
    )
    assert blocked.status_code == 409


def test_training_metrics_validate_class_label_and_range(client, db_session):
    data = setup_users_and_categories(db_session)
    headers = auth_headers(data["staff"])
    dataset = create_dataset(client, headers).json()
    add_samples(client, headers, dataset["id"])
    run = client.post(
        "/api/v1/ml/training-runs",
        headers=headers,
        json={
            "dataset_id": dataset["id"],
            "algorithm": "CNN",
            "representation": "WORD_EMBEDDING",
        },
    ).json()
    client.patch(
        f"/api/v1/ml/training-runs/{run['id']}/status",
        headers=headers,
        json={"status": "RUNNING"},
    )

    invalid_class = client.post(
        f"/api/v1/ml/training-runs/{run['id']}/metrics",
        headers=headers,
        json={"items": [{"metric_name": "precision", "class_label": "R99", "metric_value": "0.9"}]},
    )
    assert invalid_class.status_code == 422

    invalid_value = client.post(
        f"/api/v1/ml/training-runs/{run['id']}/metrics",
        headers=headers,
        json={"items": [{"metric_name": "accuracy", "metric_value": "1.5"}]},
    )
    assert invalid_value.status_code == 422


def test_ml_management_is_staff_only_and_requires_authentication(client, db_session):
    data = setup_users_and_categories(db_session)
    citizen_headers = auth_headers(data["citizen"])
    denied = client.get("/api/v1/ml/datasets", headers=citizen_headers)
    assert denied.status_code == 403
    assert client.get("/api/v1/ml/datasets").status_code == 401

    preflight = client.options(
        "/api/v1/ml/datasets/00000000-0000-0000-0000-000000000000",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "DELETE",
        },
    )
    assert preflight.status_code == 200
    assert "DELETE" in preflight.headers["access-control-allow-methods"]
