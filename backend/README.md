# Backend
Python 3.11+, FastAPI, SQLAlchemy 2, PostgreSQL (psycopg 3) and Alembic.
- `app/core/config.py`: environment settings and SQLAlchemy URL.
- `app/database/base.py`: declarative base and timestamp mixin.
- `app/database/models.py`: the 12 ORM models and relationships.
- `app/database/session.py`: engine and request-scoped sessions.
- `app/database/seed.py`: idempotent seed for 11 incident categories.
Run commands from the repository root. See `database/README.md` for migration and seed instructions.
