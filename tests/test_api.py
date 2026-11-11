import os

os.environ["DATABASE_URL"] = "sqlite:///./test_stnlp_api.db"

from fastapi.testclient import TestClient

from stnlp.api.main import app


def test_health_endpoint_reports_database_state():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["database_ready"] is True


def test_admin_endpoint_rejects_missing_key():
    with TestClient(app) as client:
        response = client.get("/api/v1/admin/reports")
    assert response.status_code == 401
