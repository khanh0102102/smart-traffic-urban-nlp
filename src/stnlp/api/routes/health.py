"""Health and readiness endpoint."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from stnlp.database.session import get_db
from stnlp.inference.predictor import get_predictor

router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: Session = Depends(get_db)):
    database_ready = True
    database_error = None
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        database_ready = False
        database_error = type(exc).__name__

    predictor = get_predictor()
    return {
        "status": "ok" if database_ready else "degraded",
        "database_ready": database_ready,
        "database_error": database_error,
        "model_ready": predictor.ready,
        "model_name": predictor.model_name,
        "model_error": predictor.load_error,
    }
