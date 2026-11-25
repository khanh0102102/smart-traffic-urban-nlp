# ST-NLP Test Suite – Phase 24

The test suite is divided into four levels and runs from the repository root.

## 1. Unit tests

Small, deterministic tests for password hashing, JWT invariants, Vietnamese text normalization, abbreviation expansion, tokenization and stopword handling.

```powershell
python -m pytest tests/unit ml/tests -q
```

## 2. API tests

FastAPI contract tests cover payload validation, authentication, current database roles, staff-only routes and error responses. The existing `backend/tests` suite also covers authentication, reports, predictions, reviews, dashboard and ML-management endpoints.

```powershell
python -m pytest tests/api backend/tests -q
```

## 3. Integration tests

Integration tests use SQLAlchemy with an isolated SQLite database and deterministic model boundary. They exercise the connected persistence flow:

`reports → preprocessing_runs → model_versions → predictions → prediction_reviews → datasets → dataset_samples → training_runs → training_metrics`.

```powershell
python -m pytest tests/integration -q
```

This tests the application/ORM integration without requiring a production database or real trained model artifact. GitHub Actions additionally runs the browser journey against PostgreSQL 16.

## 4. End-to-end browser tests

Playwright launches a real Chromium browser and a real FastAPI server. The test submits a report from the Citizen portal, checks the saved-report fallback when no inference artifact is installed, logs into the Admin portal through the staff login form, opens the case and confirms its seeded prediction. It then verifies the final category and review history in the database.

Install the test dependencies and browser once:

```powershell
python -m pip install -r backend/requirements-dev.txt
python -m playwright install chromium
```

Run the E2E suite locally (SQLite is used unless `STNLP_E2E_DATABASE_URL` is set):

```powershell
$env:RUN_BROWSER_E2E = "1"
python -m pytest tests/e2e -q
Remove-Item Env:RUN_BROWSER_E2E
```

The E2E test isolates external Google Identity, Leaflet tile and font services. It restores a valid citizen session to test the authenticated citizen journey; actual Google ID-token verification is covered by API/auth tests, not a live Google account. The browser test deliberately uses the application's real "report saved, AI unavailable" fallback instead of pretending a trained model bundle is present. The AI classification-to-database chain is covered by the integration suite.

## CI matrix

GitHub Actions runs:

- JavaScript syntax checks and the existing Node Axios client tests.
- All discovered Python unit, API, integration and existing backend/ML tests, with a coverage XML artifact.
- A separate Playwright/Chromium E2E job backed by PostgreSQL 16.

The default `python -m pytest -q` skips the opt-in browser suite. The dedicated E2E job enables it with `RUN_BROWSER_E2E=1`.
