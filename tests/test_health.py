from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "application": "ST-NLP", "phase": "21-admin-frontend"}


def test_citizen_portal_is_served():
    response = client.get("/citizen/")
    assert response.status_code == 200
    assert "Cổng người dân" in response.text
    assert 'id="report-form"' in response.text
    assert 'id="google-signin"' in response.text


def test_admin_portal_is_served_separately():
    response = client.get("/admin/")
    assert response.status_code == 200
    assert "Đăng nhập quản trị" in response.text
    assert 'id="login-form"' in response.text
    assert 'id="dashboard-view"' in response.text
    assert 'id="reports-view"' in response.text
    assert 'id="map-view"' in response.text
    assert 'id="detail-overlay"' in response.text


def test_admin_frontend_assets_are_available():
    css = client.get("/assets/css/admin.css")
    js = client.get("/assets/js/admin.js")
    assert css.status_code == 200
    assert ".sidebar" in css.text
    assert js.status_code == 200
    assert "auth/staff/login" in js.text
    assert "dashboard/summary" in js.text


def test_database_health_returns_connected_when_query_succeeds(monkeypatch):
    monkeypatch.setattr("app.api.routes.health.check_database_connection", lambda: None)
    response = client.get("/api/v1/health/database")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "connected"}


def test_database_health_returns_503_when_database_is_unavailable(monkeypatch):
    from sqlalchemy.exc import OperationalError

    def fail_connection():
        raise OperationalError("SELECT 1", {}, RuntimeError("connection refused"))

    monkeypatch.setattr("app.api.routes.health.check_database_connection", fail_connection)
    response = client.get("/api/v1/health/database")
    assert response.status_code == 503
    assert "Database is unavailable" in response.json()["detail"]
