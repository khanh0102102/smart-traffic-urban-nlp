from app.database.base import Base
from app.database import models
from app.database.seed import CATEGORIES

EXPECTED_TABLES = {
"users","incident_categories","reports","report_attachments","preprocessing_runs","model_versions",
"predictions","prediction_reviews","datasets","dataset_samples","training_runs","training_metrics",
}
def test_exactly_twelve_database_tables_are_registered():
    assert set(Base.metadata.tables) == EXPECTED_TABLES
    assert len(Base.metadata.tables) == 12
def test_eleven_fixed_categories_have_ordered_codes():
    assert len(CATEGORIES) == 11
    assert [item[0] for item in CATEGORIES] == [f"R{i:02d}" for i in range(1,12)]
    assert [item[4] for item in CATEGORIES] == list(range(1,12))
def test_models_have_expected_foreign_key_relationships():
    assert models.Report.__table__.c.user_id.foreign_keys
    assert models.Prediction.__table__.c.model_version_id.foreign_keys
    assert models.PredictionReview.__table__.c.reviewer_user_id.foreign_keys
    assert models.TrainingMetric.__table__.c.training_run_id.foreign_keys


def test_prediction_confidence_can_be_unavailable_for_uncalibrated_models():
    assert models.Prediction.__table__.c.confidence.nullable is True
    assert "confidence_type" in models.Prediction.__table__.c
    assert "decision_margin" in models.Prediction.__table__.c
    assert "probability_calibrated" in models.Prediction.__table__.c
