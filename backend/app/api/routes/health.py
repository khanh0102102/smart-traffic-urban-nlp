"""Application and database health-check routes."""
from fastapi import APIRouter, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from app.database.session import check_database_connection

router = APIRouter()


@router.get("/health", tags=["System"], summary="Check API health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "application": "ST-NLP", "phase": "23-ai-database-integration"}


@router.get("/health/database", tags=["System"], summary="Check PostgreSQL connectivity")
def database_health_check() -> dict[str, str]:
    try:
        check_database_connection()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unavailable. Check the PostgreSQL service and .env settings.",
        ) from exc
    return {"status": "ok", "database": "connected"}
