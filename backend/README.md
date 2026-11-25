# Backend configuration

Backend stack: Python 3.11+, FastAPI, Uvicorn, Pydantic Settings, SQLAlchemy 2 and PyMySQL.

## Structure

- `app/main.py`: FastAPI application and portal routes.
- `app/core/config.py`: typed settings loaded from the project-root `.env` file or environment variables.
- `app/database/base.py`: SQLAlchemy declarative base for the ORM models added later.
- `app/database/session.py`: MySQL engine, connection pool, session factory and `SELECT 1` connection check.
- `app/api/routes/health.py`: API liveness and database connectivity endpoints.

## Configure environment

From the repository root, copy `.env.example` to `.env` and set `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` and `MYSQL_ROOT_PASSWORD`. Keep `.env` local; it is ignored by Git.

`DATABASE_URL` is an optional SQLAlchemy URL override. When it is blank, the application constructs a `mysql+pymysql` URL from the `DB_*` settings.

## Start MySQL with Docker Compose

```powershell
docker compose up -d mysql
docker compose ps
```

Wait until the MySQL container reports `healthy`.

## Install and run the API

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements-dev.txt
uvicorn app.main:app --app-dir backend --reload
```

## Health endpoints

- `GET /api/v1/health`: API liveness; does not require a database connection.
- `GET /api/v1/health/database`: runs `SELECT 1` against the configured MySQL database; returns HTTP 503 if the connection fails.
- `/docs`: interactive API documentation.

No application tables or report business logic are created in this phase. Schema and migrations belong to the database implementation phase.
