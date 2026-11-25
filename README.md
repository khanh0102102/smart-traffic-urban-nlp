# Smart Traffic & Urban Incident Classifier (ST-NLP)

Project skeleton for a Vietnamese NLP system that classifies citizen reports about traffic and urban infrastructure.

## Project workspaces

| Folder | Responsibility |
|---|---|
| `backend/` | FastAPI application and API routes |
| `frontend/` | Citizen portal and admin portal |
| `ml/` | Dataset preparation, NLP preprocessing, training and evaluation |
| `database/` | Database documentation, migrations and seed data |
| `tests/` | Automated backend and integration tests |
| `docs/` | Project structure and technical documentation |

## Requirements

- Python 3.11+
- Git
- PowerShell on Windows (for the verification script)

## Run Phase 1 checks (Windows PowerShell)

Run these commands from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements-dev.txt
python -m pytest -q
```

Check the folders, files and Git ignore rules:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\check-phase1.ps1
```

## Run the skeleton application

```powershell
uvicorn app.main:app --app-dir backend --reload
```

Open:
- Citizen portal: http://127.0.0.1:8000/citizen/
- Admin portal: http://127.0.0.1:8000/admin/
- Health API: http://127.0.0.1:8000/api/v1/health
- API documentation: http://127.0.0.1:8000/docs

The portals and API are starter entry points only. Authentication, database operations, report submission and NLP inference are later phases.

## Git workflow

- `main`: stable branch.
- `feature/<name>`: new functionality.
- `fix/<name>`: bug fixes.
- Test work on a feature branch before opening a Pull Request into `main`.

Never commit `.env`, credentials, private citizen data, uploaded images, raw datasets or generated model artifacts.
