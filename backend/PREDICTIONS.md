# AI Prediction API – Phase 16

## Purpose and prerequisites

Phase 16 exposes the Phase 5 Vietnamese preprocessing pipeline and the Phase 13 shared `ModelPredictor` through FastAPI. It does not train a model inside the web request. Build the selected inference bundle first:

1. Complete model training and validation evaluation (Phases 8–12).
2. Build the shared, validation-selected bundle:
   ```powershell
   python ml/scripts/build_inference.py
   ```
3. Apply the database migration:
   ```powershell
   alembic upgrade head
   ```
4. Install the API and ML runtime dependencies:
   ```powershell
   python -m pip install -r backend/requirements-dev.txt
   ```

The API reads `PREDICTION_BUNDLE_PATH` (default `ml/artifacts/inference/best_model_bundle.joblib`). The inference artifacts are generated locally and are not committed to Git. Restart each API worker after replacing/rebuilding the bundle because the predictor is cached once per worker.

CNN/LSTM inference additionally requires TensorFlow:
```powershell
python -m pip install -r ml/requirements-embedding.txt
``
Keep TensorFlow optional when the selected bundle is Naive Bayes or SVM.

## API

All paths are under `/api/v1`. Send `Authorization: Bearer <access_token>` to every endpoint.

| Method | Endpoint | Permission | Purpose |
|---|---|---|---|
| POST | `/predictions` | Report owner or staff | Classify an existing report and persist the result |
| GET | `/predictions/{prediction_id}` | Report owner or staff | Retrieve one stored prediction |
| GET | `/predictions/report/{report_id}` | Report owner or staff | Retrieve prediction history for a report |

A citizen can classify only their own report. `ADMIN` and `RECEIVER` can classify and inspect any report. The client sends only `report_id`; the classifier always reads `raw_content` from the database, not caller-supplied text. Each request creates a new immutable inference-history row; it does not overwrite `reports.final_category_id`, which remains the staff-confirmed category. The model-version record also stores the preprocessing pipeline version and stopword configuration used for reproducibility.

### Run a prediction

```http
POST /api/v1/predictions
Authorization: Bearer <access_token>
Content-Type: application/json
```
```json
{
  "report_id": "UUID-OF-AN-EXISTING-REPORT"
}
```

The API:
- Runs the same Phase 5 Vietnamese preprocessing function using the selected bundle's `remove_stopwords` configuration.
- Records `cleaned_text`, `normalized_text`, `segmented_text`, `stopwords_removed_text` and preprocessing time in `preprocessing_runs`.
- Calls the selected Phase 13 classifier and resolves its output code against the active R01–R11 database categories.
- Creates or reuses a `model_versions` row identified by the selected bundle SHA-256, then stores the predicted category, confidence metadata, class scores, inference time and manual-review signal in `predictions`.

### Confidence is model-dependent

The API preserves the meaning defined by the ML predictor; it does not turn an SVM margin into a probability.

| `confidence_type` | Meaning |
|---|---|
| `model_score` | Naive Bayes or CNN/LSTM maximum model score; not a separately calibrated probability |
| `calibrated_probability` | SVM confidence after the training pipeline's probability calibration |
| `unavailable` | Uncalibrated SVM; `confidence` is null and `decision_margin` is stored separately |

A result marked `needs_review=true` must be checked by a staff member. The selected bundle's default review threshold is 60%; unknown-vocabulary/token cases are also flagged by the underlying predictor.

## Persistence and migration

The module reuses the existing `preprocessing_runs`, `predictions`, `model_versions`, and `incident_categories` tables. It does not add a table. Staff may optionally pass both `dataset_id` and `dataset_split` when submitting a review. That atomically materializes the confirmed report as a verified `dataset_samples` row. Later reviews refresh that linked sample's confirmed label while every `prediction_reviews` decision remains in append-only history. Migration `20261125_0002`:
- makes `predictions.confidence` nullable so an uncalibrated SVM is represented honestly;
- adds `confidence_type`, `probability_calibrated`, `decision_margin`, and `review_reasons` for reproducibility and review traceability.

The database remains at 12 tables. Existing rows receive `confidence_type='model_score'` and null defaults for the new optional metadata.

## Tests

```powershell
python -m pytest backend/tests/test_predictions.py -q
python -m pytest -q
```

Tests use SQLite, deterministic fake model outputs and the real Phase 5 preprocessing stages (with word segmentation stubbed). They cover successful persistence, an uncalibrated SVM, owner-only access, staff history access, and ensuring AI predictions do not overwrite the staff-final category.
