from fastapi.testclient import TestClient

from stnlp.api.main import app
from stnlp.database.models import ModerationLog, Prediction as PredictionRow, Report
from stnlp.database.session import SessionLocal
from stnlp.inference.predictor import Prediction
import stnlp.api.routes.reports as report_routes


class FakePredictor:
    def predict(self, text, include_probabilities=True):
        return Prediction(
            predicted_code="R03",
            confidence=0.55,
            probabilities={"R03": 0.55},
            model_name="fake-test-model",
            model_version="test",
        )


def test_report_submission_and_moderation_preserve_ai_prediction(monkeypatch):
    monkeypatch.setattr(report_routes, "get_predictor", lambda: FakePredictor())
    admin_headers = {"X-Admin-Key": "change-this-demo-key-before-use"}

    with TestClient(app) as client:
        # Keep the test repeatable when run more than once on the same workstation.
        with SessionLocal() as db:
            db.query(ModerationLog).delete()
            db.query(PredictionRow).delete()
            db.query(Report).delete()
            db.commit()

        created = client.post(
            "/api/v1/reports",
            json={
                "text": "Hố ga trước nhà bị mất nắp và gây nguy hiểm",
                "latitude": 16.0471,
                "longitude": 108.2068,
            },
        )
        assert created.status_code == 201
        report_data = created.json()
        assert report_data["predicted_category"]["code"] == "R03"
        assert report_data["needs_review"] is True
        report_id = report_data["report_id"]

        queue_before = client.get(
            "/api/v1/admin/reports?needs_review=true", headers=admin_headers
        )
        assert queue_before.status_code == 200
        assert queue_before.json()["total"] == 1

        moderated = client.patch(
            f"/api/v1/admin/reports/{report_id}/moderation",
            headers=admin_headers,
            json={"category_code": "R03", "note": "Đã kiểm tra thủ công."},
        )
        assert moderated.status_code == 200
        assert moderated.json()["report"]["status"] == "reviewed"

        queue_after = client.get(
            "/api/v1/admin/reports?needs_review=true", headers=admin_headers
        )
        assert queue_after.status_code == 200
        assert queue_after.json()["total"] == 0

        detail = client.get(
            f"/api/v1/admin/reports/{report_id}", headers=admin_headers
        )
        assert detail.status_code == 200
        assert detail.json()["predicted_category"]["code"] == "R03"
        assert detail.json()["confirmed_category"]["code"] == "R03"
        assert len(detail.json()["moderation_history"]) == 1
