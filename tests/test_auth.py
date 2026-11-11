from stnlp.auth.security import create_access_token, decode_access_token, hash_password, verify_password


def test_password_hash_verifies_without_storing_plaintext():
    password = "A-secure-demo-password!"
    encoded = hash_password(password)
    assert password not in encoded
    assert verify_password(password, encoded) is True
    assert verify_password("incorrect-password", encoded) is False


def test_jwt_token_contains_admin_subject():
    token = create_access_token("stnlp-admin")
    payload = decode_access_token(token)
    assert payload["sub"] == "stnlp-admin"
    assert payload["role"] == "admin"


def test_admin_login_and_bearer_access(monkeypatch):
    from fastapi.testclient import TestClient

    from stnlp.api.main import app
    from stnlp.config import settings
    from stnlp.database.models import AdminUser
    from stnlp.database.session import SessionLocal

    monkeypatch.setattr(settings, "admin_password_hash", "configured-for-test")
    password = "Long-test-admin-password!"
    with SessionLocal() as db:
        db.query(AdminUser).filter(AdminUser.username == "auth-test-admin").delete()
        db.add(AdminUser(
            username="auth-test-admin",
            password_hash=hash_password(password),
            role="admin",
            is_active=True,
        ))
        db.commit()

    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/login",
            json={"username": "auth-test-admin", "password": password},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]
        profile = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert profile.status_code == 200
        assert profile.json()["username"] == "auth-test-admin"

    with SessionLocal() as db:
        db.query(AdminUser).filter(AdminUser.username == "auth-test-admin").delete()
        db.commit()
