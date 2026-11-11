"""Stateless prediction endpoint."""

from fastapi import APIRouter, HTTPException, status

from stnlp.api.schemas import CategoryOut, PredictRequest, PredictionOut
from stnlp.config import settings
from stnlp.inference.predictor import ModelNotReadyError, get_predictor
from stnlp.labels import label_name

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionOut)
def predict(payload: PredictRequest):
    try:
        result = get_predictor().predict(
            payload.text, include_probabilities=payload.include_probabilities
        )
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Mô hình chưa sẵn sàng: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    probabilities = result.probabilities or None
    return PredictionOut(
        predicted_category=CategoryOut(
            code=result.predicted_code, name=label_name(result.predicted_code)
        ),
        confidence=result.confidence,
        needs_review=result.confidence <= settings.confidence_threshold,
        model_name=result.model_name,
        model_version=result.model_version,
        probabilities=probabilities,
    )
