from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "application": "ST-NLP",
        "phase": "23-ai-database-integration",
    }


def test_citizen_portal_loads_shared_axios_client_before_app():
    response = client.get("/citizen/")
    assert response.status_code == 200
    html = response.text
    assert "Cổng người dân" in html
    assert 'id="report-form"' in html
    assert 'id="google-signin"' in html
    assert html.index("axios@1.21.1") < html.index("/assets/js/api-client.js") < html.index("/assets/js/app.js")


def test_admin_portal_loads_shared_axios_client_before_app():
    response = client.get("/admin/")
    assert response.status_code == 200
    html = response.text
    assert "Đăng nhập quản trị" in html
    assert 'id="login-form"' in html
    assert html.index("axios@1.21.1") < html.index("/assets/js/api-client.js") < html.index("/assets/js/admin.js")


def test_shared_api_client_is_served_and_has_auth_error_and_role_interceptors():
    response = client.get("/assets/js/api-client.js")
    assert response.status_code == 200
    assert "axios.create" in response.text
    assert "interceptors.request.use" in response.text
    assert "interceptors.response.use" in response.text
    assert "401" in response.text
    assert "assertRole" in response.text


def test_frontend_api_integrations_use_shared_client_without_direct_fetch():
    citizen = client.get("/assets/js/app.js")
    admin = client.get("/assets/js/admin.js")
    assert citizen.status_code == 200
    assert admin.status_code == 200
    assert "createApiClient" in citizen.text
    assert "createApiClient" in admin.text
    assert "fetch(" not in citizen.text
    assert "fetch(" not in admin.text


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
