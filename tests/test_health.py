from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

def test_health_endpoint_returns_ok():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "application": "ST-NLP", "phase": "16-prediction-api"}

def test_citizen_portal_is_served():
    response = client.get("/citizen/")
    assert response.status_code == 200
    assert "Cổng người dân" in response.text

def test_admin_portal_is_served_separately():
    response = client.get("/admin/")
    assert response.status_code == 200
    assert "Cổng quản trị" in response.text

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
