"""API contract tests for payload validation and role boundaries."""
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
from app.database.models import IncidentCategory, User
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
    session.add_all([
        IncidentCategory(
            code=code, name=name, slug=slug, description=description,
            sort_order=order, is_active=True,
        )
        for code, name, slug, description, order in CATEGORIES
    ])
    citizen = User(
        email=f"api-citizen-{uuid4().hex[:8]}@example.com",
        full_name="API Citizen", role="CITIZEN", google_sub=f"google-{uuid4()}",
    )
    staff = User(
        email=f"api-staff-{uuid4().hex[:8]}@example.com",
        full_name="API Receiver", role="RECEIVER",
    )
    session.add_all([citizen, staff])
    session.commit()
    try:
        yield session, citizen, staff
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def api(db_session, monkeypatch):
    session, citizen, staff = db_session
    monkeypatch.setattr(settings, "jwt_secret_key", "phase24-api-secret-" * 3)
    monkeypatch.setattr(settings, "jwt_algorithm", "HS256")
    monkeypatch.setattr(settings, "jwt_issuer", "st-nlp-api")
    monkeypatch.setattr(settings, "jwt_audience", "st-nlp-client")
    monkeypatch.setattr(settings, "jwt_access_token_expire_minutes", 30)
    def override_db():
        yield session
    app.dependency_overrides.clear()
    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as client:
        yield client, citizen, staff
    app.dependency_overrides.clear()


def headers(user):
    token, _ = create_access_token(user.id, user.role)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize(
    "payload",
    [
        {"raw_content": "tiny"},
        {"raw_content": "valid report", "latitude": 12.5},
        {"raw_content": "valid report", "longitude": 108.1},
        {"raw_content": "valid report", "latitude": 91, "longitude": 108.1},
    ],
)
def test_report_api_rejects_invalid_content_or_coordinate_pairs(api, payload):
    client, citizen, _staff = api
    response = client.post("/api/v1/reports", headers=headers(citizen), json=payload)
    assert response.status_code == 422


def test_staff_dashboard_and_ml_routes_reject_citizen_role(api):
    client, citizen, staff = api
    dashboard_denied = client.get("/api/v1/dashboard/summary", headers=headers(citizen))
    datasets_denied = client.get("/api/v1/ml/datasets", headers=headers(citizen))
    staff_dashboard = client.get("/api/v1/dashboard/summary", headers=headers(staff))
    assert dashboard_denied.status_code == 403
    assert datasets_denied.status_code == 403
    assert staff_dashboard.status_code == 200
    assert len(staff_dashboard.json()["categories"]) == 11


def test_protected_api_routes_reject_missing_bearer_token(api):
    client, _citizen, _staff = api
    assert client.get("/api/v1/reports/mine").status_code == 401
    assert client.get("/api/v1/dashboard/summary").status_code == 401
    assert client.get("/api/v1/ml/datasets").status_code == 401


def test_prediction_api_rejects_invalid_report_identifier(api):
    client, citizen, _staff = api
    response = client.get(
        "/api/v1/reports/not-a-uuid",
        headers=headers(citizen),
    )
    assert response.status_code == 422
