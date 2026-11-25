# Backend

Python 3.11+ and FastAPI starter application.

- `app/main.py`: FastAPI application and portal page routes.
- `app/api/routes/health.py`: health-check endpoint.
- `app/core/`: shared configuration and infrastructure.

Start from repository root:

```powershell
pip install -r backend/requirements-dev.txt
uvicorn app.main:app --app-dir backend --reload
```

This phase intentionally contains no database connection, authentication, report workflow or ML inference.
