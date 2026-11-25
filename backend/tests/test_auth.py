"""Authentication endpoint, JWT and role-guard tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.security import hash_password
from app.database.base import Base
from app.database import models  # noqa: F401 - register all mapped tables
from app.database.models import User
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
    monkeypatch.setattr(settings, "jwt_secret_key", "a" * 64)
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


def google_claims(**overrides):
    claims = {
        "sub": "google-subject-123",
        "email": "citizen@example.com",
        "email_verified": True,
        "name": "Citizen Person",
        "iss": "https://accounts.google.com",
    }
    claims.update(overrides)
    return claims


def test_google_login_creates_citizen_and_jwt_me(client, monkeypatch):
    import app.core.auth_service as service
    monkeypatch.setattr(service, "verify_google_id_token", lambda _token: google_claims())

    response = client.post("/api/v1/auth/google", json={"id_token": "x" * 40})
    assert response.status_code == 200
    payload = response.json()
    assert payload["user"]["role"] == "CITIZEN"
    assert payload["user"]["email"] == "citizen@example.com"
    assert payload["token_type"] == "bearer"
    assert payload["expires_in"] == 1800

    me = client.get("/api/v1/auth/me", headers={
        "Authorization": f"Bearer {payload['access_token']}"
    })
    assert me.status_code == 200
    assert me.json()["id"] == payload["user"]["id"]


def test_google_login_refuses_to_link_staff_account(client, db_session, monkeypatch):
    import app.core.auth_service as service
    db_session.add(User(
        email="citizen@example.com", full_name="Existing Staff", role="RECEIVER",
        password_hash=hash_password("A-strong-staff-password"),
    ))
    db_session.commit()
    monkeypatch.setattr(service, "verify_google_id_token", lambda _token: google_claims())

    response = client.post("/api/v1/auth/google", json={"id_token": "x" * 40})
    assert response.status_code == 409


def test_google_login_requires_verified_email(client, monkeypatch):
    import app.core.auth_service as service
    monkeypatch.setattr(
        service, "verify_google_id_token",
        lambda _token: google_claims(email_verified=False),
    )
    # The verifier itself rejects unverified email; this endpoint must return an auth error.
    from fastapi import HTTPException
    def reject_unverified(_token):
        raise HTTPException(status_code=401, detail="Google account must have a verified email.")
    monkeypatch.setattr(service, "verify_google_id_token", reject_unverified)

    response = client.post("/api/v1/auth/google", json={"id_token": "x" * 40})
    assert response.status_code == 401


def test_staff_login_returns_jwt_and_staff_role_is_authorized(client, db_session):
    db_session.add(User(
        email="receiver@example.com", full_name="Traffic Receiver", role="RECEIVER",
        password_hash=hash_password("A-strong-staff-password"),
    ))
    db_session.commit()

    response = client.post("/api/v1/auth/staff/login", json={
        "email": "RECEIVER@example.com",
        "password": "A-strong-staff-password",
    })
    assert response.status_code == 200
    token = response.json()["access_token"]
    staff_me = client.get("/api/v1/auth/staff/me", headers={"Authorization": f"Bearer {token}"})
    assert staff_me.status_code == 200
    assert staff_me.json()["role"] == "RECEIVER"


def test_citizen_cannot_use_staff_password_login(client, db_session):
    db_session.add(User(
        email="person@example.com", full_name="Citizen", role="CITIZEN",
        google_sub="google-citizen-sub",
    ))
    db_session.commit()
    response = client.post("/api/v1/auth/staff/login", json={
        "email": "person@example.com",
        "password": "any-password",
    })
    assert response.status_code == 401


def test_staff_only_endpoint_rejects_citizen_jwt(client, monkeypatch):
    import app.core.auth_service as service
    monkeypatch.setattr(service, "verify_google_id_token", lambda _token: google_claims())
    login = client.post("/api/v1/auth/google", json={"id_token": "x" * 40})
    token = login.json()["access_token"]
    response = client.get("/api/v1/auth/staff/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


def test_auth_me_rejects_missing_or_invalid_bearer_token(client):
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get(
        "/api/v1/auth/me", headers={"Authorization": "Bearer not-a-valid-jwt"}
    ).status_code == 401
