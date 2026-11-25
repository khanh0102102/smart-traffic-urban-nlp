# Smart Traffic & Urban Incident Classifier (ST-NLP)

ST-NLP is a Vietnamese NLP system for classifying citizen reports about traffic and urban infrastructure.

## Project workspaces
| Folder | Responsibility |
|---|---|
| `backend/` | FastAPI API, SQLAlchemy models, PostgreSQL and application services |
| `frontend/` | Separate citizen and admin portals |
| `ml/` | Dataset preparation, NLP preprocessing, training and evaluation |
| `database/` | PostgreSQL schema, Alembic migrations and seed data |
| `tests/` | Automated backend and integration tests |

## Initialize local database (PowerShell)
```powershell
Copy-Item .env.example .env
docker compose up -d postgres
python -m pip install -r backend/requirements-dev.txt
alembic upgrade head
python scripts/seed_categories.py
```
Wait until `stnlp-postgres` is healthy before running migrations. The default database is `stnlp` on port `5432`.

## Phase 4 – Prepare a dataset
Install the NLP data-processing dependencies and run the included sample:

```powershell
python -m pip install -r ml/requirements.txt
python ml/scripts/prepare_dataset.py --input ml/data/sample_reports.csv --text-column content --label-column category
```

See [`ml/README.md`](ml/README.md) for supported input formats, custom column mapping, output files, weak-label review rules and split details. Generated datasets remain local and are excluded from Git.

## Run backend
```powershell
uvicorn app.main:app --app-dir backend --reload
```
Open http://127.0.0.1:8000/docs. Health checks are at `/api/v1/health` and `/api/v1/health/database`; the citizen and admin portals remain separate at `/citizen/` and `/admin/`.

## Tests
```powershell
python -m pytest -q
```
Never commit secrets, private citizen data, uploaded images, raw datasets or generated model artifacts.
