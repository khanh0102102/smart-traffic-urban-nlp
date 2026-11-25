"""Report creation, ownership, staff workflow and image upload tests."""
from __future__ import annotations

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
def client(db_session, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "jwt_secret_key", "b" * 64)
    monkeypatch.setattr(settings, "jwt_algorithm", "HS256")
    monkeypatch.setattr(settings, "jwt_issuer", "st-nlp-api")
    monkeypatch.setattr(settings, "jwt_audience", "st-nlp-client")
    monkeypatch.setattr(settings, "jwt_access_token_expire_minutes", 30)
    monkeypatch.setattr(settings, "report_upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "report_max_upload_size_bytes", 1024)
    monkeypatch.setattr(settings, "report_max_attachments", 5)

    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def create_user(db_session, email: str, role: str) -> User:
    user = User(
        email=email,
        full_name=email.split("@")[0],
        role=role,
        google_sub=f"google-{email}" if role == "CITIZEN" else None,
        password_hash=None,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def auth_headers(user: User) -> dict[str, str]:
    token, _ = create_access_token(user.id, user.role)
    return {"Authorization": f"Bearer {token}"}


def test_citizen_creates_report_and_coordinates_are_returned(client, db_session):
    citizen = create_user(db_session, "one@example.com", "CITIZEN")
    response = client.post(
        "/api/v1/reports",
        headers=auth_headers(citizen),
        json={
            "raw_content": "Mặt đường có ổ gà lớn trước nhà dân.",
            "address": "20 Nguyễn Huệ",
            "latitude": 12.6816,
            "longitude": 108.0372,
        },
    )
    assert response.status_code == 201
    report = response.json()
    assert report["status"] == "RECEIVED"
    assert report["user_id"] == str(citizen.id)
    assert report["latitude"] == "12.6816000"
    assert report["longitude"] == "108.0372000"
    assert report["tracking_code"].startswith("RPT-")
    assert report["attachments"] == []


def test_create_report_requires_citizen_role_and_paired_coordinates(client, db_session):
    staff = create_user(db_session, "staff@example.com", "RECEIVER")
    assert client.post(
        "/api/v1/reports",
        headers=auth_headers(staff),
        json={"raw_content": "Đèn tín hiệu giao thông bị hỏng."},
    ).status_code == 403

    citizen = create_user(db_session, "two@example.com", "CITIZEN")
    response = client.post(
        "/api/v1/reports",
        headers=auth_headers(citizen),
        json={"raw_content": "Đèn tín hiệu giao thông bị hỏng.", "latitude": 10.0},
    )
    assert response.status_code == 422


def test_citizens_only_see_their_own_reports(client, db_session):
    first = create_user(db_session, "first@example.com", "CITIZEN")
    second = create_user(db_session, "second@example.com", "CITIZEN")
    created = client.post(
        "/api/v1/reports",
        headers=auth_headers(first),
        json={"raw_content": "Nắp cống bị vỡ trên vỉa hè."},
    )
    report_id = created.json()["id"]

    own = client.get("/api/v1/reports/mine", headers=auth_headers(first))
    assert own.status_code == 200
    assert own.json()["total"] == 1
    assert own.json()["items"][0]["id"] == report_id

    hidden = client.get(f"/api/v1/reports/{report_id}", headers=auth_headers(second))
    assert hidden.status_code == 404
    assert client.get("/api/v1/reports", headers=auth_headers(first)).status_code == 403


def test_staff_lists_reports_and_updates_status_and_category(client, db_session):
    citizen = create_user(db_session, "reporter@example.com", "CITIZEN")
    staff = create_user(db_session, "receiver@example.com", "RECEIVER")
    category = IncidentCategory(
        code="R01",
        name="Đường/Cầu/Hạ tầng hư hỏng",
        slug="duong-cau-ha-tang-hu-hong",
        sort_order=1,
        is_active=True,
    )
    db_session.add(category)
    db_session.commit()

    created = client.post(
        "/api/v1/reports",
        headers=auth_headers(citizen),
        json={"raw_content": "Mặt đường có ổ gà sâu."},
    )
    report_id = created.json()["id"]
    denied = client.patch(
        f"/api/v1/reports/{report_id}/status",
        headers=auth_headers(citizen),
        json={"status": "PROCESSING"},
    )
    assert denied.status_code == 403

    listed = client.get("/api/v1/reports?status=RECEIVED", headers=auth_headers(staff))
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    updated = client.patch(
        f"/api/v1/reports/{report_id}/status",
        headers=auth_headers(staff),
        json={"status": "PROCESSING", "final_category_code": "R01"},
    )
    assert updated.status_code == 200
    assert updated.json()["status"] == "PROCESSING"
    assert updated.json()["category"]["code"] == "R01"

    filtered = client.get(
        "/api/v1/reports?category_code=R01",
        headers=auth_headers(staff),
    )
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1

    cleared = client.patch(
        f"/api/v1/reports/{report_id}/status",
        headers=auth_headers(staff),
        json={"status": "PROCESSING", "final_category_code": None},
    )
    assert cleared.status_code == 200
    assert cleared.json()["category"] is None


def test_upload_image_and_download_is_owner_scoped(client, db_session):
    citizen = create_user(db_session, "image-owner@example.com", "CITIZEN")
    other = create_user(db_session, "other-owner@example.com", "CITIZEN")
    created = client.post(
        "/api/v1/reports",
        headers=auth_headers(citizen),
        json={"raw_content": "Cây xanh gãy đổ chắn lối đi."},
    )
    report_id = created.json()["id"]
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"valid-test-image"
    uploaded = client.post(
        f"/api/v1/reports/{report_id}/attachments",
        headers=auth_headers(citizen),
        files=[("files", ("tree.png", image_bytes, "image/png"))],
    )
    assert uploaded.status_code == 201
    attachment = uploaded.json()[0]
    assert attachment["original_filename"] == "tree.png"
    assert attachment["mime_type"] == "image/png"

    downloaded = client.get(attachment["file_url"], headers=auth_headers(citizen))
    assert downloaded.status_code == 200
    assert downloaded.content == image_bytes
    assert client.get(attachment["file_url"], headers=auth_headers(other)).status_code == 404


def test_upload_rejects_non_images_and_oversized_files(client, db_session):
    citizen = create_user(db_session, "upload@example.com", "CITIZEN")
    report = client.post(
        "/api/v1/reports",
        headers=auth_headers(citizen),
        json={"raw_content": "Biển báo giao thông bị nghiêng."},
    ).json()
    url = f"/api/v1/reports/{report['id']}/attachments"

    invalid = client.post(
        url,
        headers=auth_headers(citizen),
        files=[("files", ("note.txt", b"not-an-image", "text/plain"))],
    )
    assert invalid.status_code == 415

    oversized = b"\x89PNG\r\n\x1a\n" + (b"x" * 1100)
    large = client.post(
        url,
        headers=auth_headers(citizen),
        files=[("files", ("large.png", oversized, "image/png"))],
    )
    assert large.status_code == 413


def test_report_detail_requires_authentication(client, db_session):
    citizen = create_user(db_session, "private@example.com", "CITIZEN")
    report = client.post(
        "/api/v1/reports",
        headers=auth_headers(citizen),
        json={"raw_content": "Ngập nước sau cơn mưa lớn."},
    ).json()
    assert client.get(f"/api/v1/reports/{report['id']}").status_code == 401
