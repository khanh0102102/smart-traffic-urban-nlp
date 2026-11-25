# Smart Traffic & Urban Incident Classifier (ST-NLP)

ST-NLP is a Vietnamese NLP system for classifying citizen reports about traffic and urban infrastructure.

## Project workspaces

| Folder | Responsibility |
|---|---|
| `backend/` | FastAPI API, settings, database engine and application services |
| `frontend/` | Separate citizen and admin portals |
| `ml/` | Dataset preparation, NLP preprocessing, training and evaluation |
| `database/` | Database documentation, migrations and seed data |
| `tests/` | Automated backend and integration tests |
| `docs/` | Project documentation |

## Requirements

- Python 3.11+
- Git
- Docker Desktop with Docker Compose (recommended for local MySQL 8.4)

## Configure local environment

Run from the repository root in PowerShell:

```powershell
Copy-Item .env.example .env
```

Open `.env` and change the development passwords as needed. Do not commit `.env`.

## Start MySQL

```powershell
docker compose up -d mysql
docker compose ps
```

Wait until `stnlp-mysql` reports `healthy`.

## Install and run backend

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements-dev.txt
uvicorn app.main:app --app-dir backend --reload
```

## Verify the backend

Open the API docs at http://127.0.0.1:8000/docs.

- API liveness: http://127.0.0.1:8000/api/v1/health
- MySQL connectivity (`SELECT 1`): http://127.0.0.1:8000/api/v1/health/database
- Citizen portal: http://127.0.0.1:8000/citizen/
- Admin portal: http://127.0.0.1:8000/admin/

Run automated tests:

```powershell
python -m pytest -q
```

Run the Phase 1 repository skeleton check:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\check-phase1.ps1
```

## Notes

Phase 2 configures the backend runtime and MySQL connection. It does not create application tables or implement Google OAuth, report submission, image upload workflows or NLP models. Those are handled in their corresponding later phases.

Never commit `.env`, credentials, private citizen data, uploaded images, raw datasets or generated model artifacts.
