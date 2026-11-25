from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "application": "ST-NLP",
        "phase": "1-project-skeleton",
    }


def test_citizen_portal_is_served():
    response = client.get("/citizen/")

    assert response.status_code == 200
    assert "Cổng người dân" in response.text


def test_admin_portal_is_served_separately():
    response = client.get("/admin/")

    assert response.status_code == 200
    assert "Cổng quản trị" in response.text
