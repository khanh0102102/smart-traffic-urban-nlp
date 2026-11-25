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

## Configure local environment
Run from the repository root in PowerShell:
```powershell
Copy-Item .env.example .env
```
Keep `.env` local; do not commit credentials.

## Start PostgreSQL and initialize schema
```powershell
docker compose up -d postgres
python -m pip install -r backend/requirements-dev.txt
alembic upgrade head
python -m app.database.seed
```
Wait until `stnlp-postgres` is healthy before running migrations. Default local database: `stnlp`, port `5432`.

## Run backend
```powershell
uvicorn app.main:app --app-dir backend --reload
```
Open http://127.0.0.1:8000/docs.

- API liveness: http://127.0.0.1:8000/api/v1/health
- PostgreSQL connectivity: http://127.0.0.1:8000/api/v1/health/database
- Citizen portal: http://127.0.0.1:8000/citizen/
- Admin portal: http://127.0.0.1:8000/admin/

## Tests
```powershell
python -m pytest -q
```

Never commit secrets, private citizen data, uploaded images, raw datasets or generated model artifacts.
