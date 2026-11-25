# Database
ST-NLP uses PostgreSQL 17, SQLAlchemy 2 and Alembic. The schema contains exactly 12 application tables as specified in `DATABASE_STRUCTURE_ST_NLP.txt`.
Tables: `users`, `incident_categories`, `reports`, `report_attachments`, `preprocessing_runs`, `model_versions`, `predictions`, `prediction_reviews`, `datasets`, `dataset_samples`, `training_runs`, `training_metrics`.
## Run from repository root (PowerShell)
```powershell
docker compose up -d postgres
python -m pip install -r backend/requirements-dev.txt
alembic upgrade head
python -m app.database.seed
```
Set `DATABASE_URL` in `.env` if the PostgreSQL host, port or credentials differ from development defaults. Migrations create schema; the seed command inserts the fixed 11 category rows idempotently.
Uploaded images and model binaries stay outside PostgreSQL. Store only their storage paths in `report_attachments.storage_path` and `model_versions.artifact_path`.
