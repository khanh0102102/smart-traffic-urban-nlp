"""Dashboard aggregate, category resolution, map and authorization tests."""
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
from app.database.models import IncidentCategory, ModelVersion, Prediction, Report, User
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
    monkeypatch.setattr(settings, "jwt_secret_key", "e" * 64)
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


def setup_dashboard_data(db_session):
    staff = User(
        email=f"staff-{uuid4().hex[:8]}@example.com",
        full_name="Dashboard Staff",
        role="RECEIVER",
    )
    citizen = User(
        email=f"citizen-{uuid4().hex[:8]}@example.com",
        full_name="Dashboard Citizen",
        role="CITIZEN",
        google_sub=f"google-{uuid4()}",
    )
    categories = []
    for code, name, slug, description, order in CATEGORIES:
        category = IncidentCategory(
            code=code,
            name=name,
            slug=slug,
            description=description,
            sort_order=order,
            is_active=True,
        )
        categories.append(category)
    model_version = ModelVersion(
        name="Dashboard test model",
        version=f"test-{uuid4().hex[:8]}",
        algorithm="NAIVE_BAYES",
        representation="TF_IDF",
        artifact_path="test/dashboard.joblib",
        parameters={"fixture": True},
        is_active=True,
    )
    db_session.add_all([staff, citizen, *categories, model_version])
    db_session.commit()

    by_code = {category.code: category for category in categories}
    first = Report(
        tracking_code=f"RPT-{uuid4().hex[:10]}",
        user_id=citizen.id,
        raw_content="Mặt đường có ổ gà sâu trước cổng trường.",
        address="12 Nguyễn Huệ",
        latitude=12.6816,
        longitude=108.0372,
        final_category_id=by_code["R02"].id,
        status="RECEIVED",
    )
    second = Report(
        tracking_code=f"RPT-{uuid4().hex[:10]}",
        user_id=citizen.id,
        raw_content="Dây điện bị đứt và có nguy cơ mất an toàn.",
        address="20 Trần Phú",
        latitude=12.6820,
        longitude=108.0380,
        status="RESOLVED",
    )
    third = Report(
        tracking_code=f"RPT-{uuid4().hex[:10]}",
        user_id=citizen.id,
        raw_content="Một phản ánh chưa được phân loại và không có vị trí.",
        status="REJECTED",
    )
    db_session.add_all([first, second, third])
    db_session.commit()

    db_session.add_all(
        [
            Prediction(
                report_id=first.id,
                model_version_id=model_version.id,
                predicted_category_id=by_code["R01"].id,
                confidence=0.81,
                confidence_type="model_score",
                needs_review=False,
            ),
            Prediction(
                report_id=second.id,
                model_version_id=model_version.id,
                predicted_category_id=by_code["R02"].id,
                confidence=0.91,
                confidence_type="model_score",
                needs_review=False,
            ),
        ]
    )
    db_session.commit()
    return {
        "staff": staff,
        "citizen": citizen,
        "categories": by_code,
        "reports": [first, second, third],
    }


def test_dashboard_summary_counts_all_eleven_categories_and_effective_labels(client, db_session):
    data = setup_dashboard_data(db_session)
    response = client.get(
        "/api/v1/dashboard/summary",
        headers=auth_headers(data["staff"]),
    )
    assert response.status_code == 200, response.text
    result = response.json()

    assert result["total_reports"] == 3
    assert result["categorized_reports"] == 2
    assert result["unclassified_reports"] == 1
    assert result["geolocated_reports"] == 2
    assert result["pending_reports"] == 1
    assert result["resolved_reports"] == 1
    assert result["rejected_reports"] == 1
    assert len(result["categories"]) == 11

    counts = {item["code"]: item["report_count"] for item in result["categories"]}
    percentages = {item["code"]: item["percentage"] for item in result["categories"]}
    # The confirmed R02 label overrides the original R01 AI prediction.
    assert counts["R01"] == 0
    assert counts["R02"] == 2
    assert counts["R11"] == 0
    assert percentages["R02"] == 100.0
    assert len(result["statuses"]) == 4


def test_dashboard_map_returns_only_geolocated_points_and_category_filter(client, db_session):
    data = setup_dashboard_data(db_session)
    headers = auth_headers(data["staff"])

    all_points = client.get("/api/v1/dashboard/map", headers=headers)
    assert all_points.status_code == 200, all_points.text
    result = all_points.json()
    assert result["total"] == 2
    assert len(result["items"]) == 2
    assert all(item["latitude"] is not None and item["longitude"] is not None for item in result["items"])

    filtered = client.get(
        "/api/v1/dashboard/map?category_code=r02&status=RESOLVED",
        headers=headers,
    )
    assert filtered.status_code == 200, filtered.text
    filtered_result = filtered.json()
    assert filtered_result["total"] == 1
    assert filtered_result["items"][0]["category_code"] == "R02"
    assert filtered_result["items"][0]["status"] == "RESOLVED"


def test_dashboard_map_paginates_and_rejects_unknown_category(client, db_session):
    data = setup_dashboard_data(db_session)
    headers = auth_headers(data["staff"])
    page = client.get("/api/v1/dashboard/map?limit=1&offset=0", headers=headers)
    assert page.status_code == 200
    assert page.json()["total"] == 2
    assert len(page.json()["items"]) == 1
    assert page.json()["has_more"] is True

    unknown = client.get("/api/v1/dashboard/map?category_code=R99", headers=headers)
    assert unknown.status_code == 422


def test_dashboard_date_range_and_invalid_range(client, db_session):
    data = setup_dashboard_data(db_session)
    headers = auth_headers(data["staff"])
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    included = client.get(
        f"/api/v1/dashboard/summary?date_from={today}&date_to={today}",
        headers=headers,
    )
    assert included.status_code == 200
    assert included.json()["total_reports"] == 3

    invalid = client.get(
        "/api/v1/dashboard/summary?date_from=2026-02-02&date_to=2026-02-01",
        headers=headers,
    )
    assert invalid.status_code == 422


def test_dashboard_is_staff_only_and_requires_authentication(client, db_session):
    data = setup_dashboard_data(db_session)
    citizen_response = client.get(
        "/api/v1/dashboard/summary",
        headers=auth_headers(data["citizen"]),
    )
    assert citizen_response.status_code == 403
    assert client.get("/api/v1/dashboard/map").status_code == 401
