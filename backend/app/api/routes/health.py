"""Application and database health-check routes."""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError

from app.database.session import check_database_connection

router = APIRouter()


@router.get("/health", tags=["System"], summary="Check API health")
def health_check() -> dict[str, str]:
    """Liveness check; does not require MySQL to be online."""
    return {
        "status": "ok",
        "application": "ST-NLP",
        "phase": "2-backend-setup",
    }


@router.get("/health/database", tags=["System"], summary="Check MySQL connectivity")
def database_health_check() -> dict[str, str]:
    """Check the configured database using SELECT 1."""
    try:
        check_database_connection()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unavailable. Check the MySQL service and .env settings.",
        ) from exc

    return {"status": "ok", "database": "connected"}
