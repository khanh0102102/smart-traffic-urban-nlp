"""Health-check route."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["System"], summary="Check API health")
def health_check() -> dict[str, str]:
    """Return a minimal readiness response."""
    return {
        "status": "ok",
        "application": "ST-NLP",
        "phase": "1-project-skeleton",
    }
