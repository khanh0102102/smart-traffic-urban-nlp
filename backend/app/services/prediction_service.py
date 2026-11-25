"""Bridge the Phase 5/13 NLP inference pipeline to persistent API records."""
from __future__ import annotations

import hashlib
import sys
import threading
from pathlib import Path
from time import perf_counter
from typing import Any
from decimal import Decimal

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import PROJECT_ROOT, settings
from app.database.models import (
    IncidentCategory,
    ModelVersion,
    Prediction,
    PreprocessingRun,
    Report,
)

# Phase 13 exposes its reusable package as "src" from the repository's ml/ folder.
ML_ROOT = PROJECT_ROOT / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from src.inference import ModelPredictor  # noqa: E402
from src.nlp_preprocessing import preprocess_vietnamese_text  # noqa: E402

PREPROCESSING_PIPELINE_VERSION = "phase5-v1"
MODEL_ALGORITHMS = {
    "naive_bayes": ("NAIVE_BAYES", "TF_IDF"),
    "svm": ("SVM", "TF_IDF"),
    "cnn": ("CNN", "WORD_EMBEDDING"),
    "lstm": ("LSTM", "WORD_EMBEDDING"),
}

_cached_predictor: ModelPredictor | None = None
_predictor_lock = threading.Lock()


def _resolved_bundle_path() -> Path:
    path = settings.prediction_bundle_path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_model_predictor() -> ModelPredictor:
    """Load the selected inference bundle once per API worker."""
    global _cached_predictor
    if _cached_predictor is not None:
        return _cached_predictor
    with _predictor_lock:
        if _cached_predictor is not None:
            return _cached_predictor
        try:
            _cached_predictor = ModelPredictor(_resolved_bundle_path())
        except (OSError, ValueError, RuntimeError, ImportError, TypeError) as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The trained inference bundle is unavailable or invalid. "
                    "Complete ML training/evaluation and run "
                    "'python ml/scripts/build_inference.py'."
                ),
            ) from exc
    return _cached_predictor


def _bundle_sha256(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError("Loaded inference bundle is no longer available on disk.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_model_version(db: Session, predictor: ModelPredictor) -> ModelVersion:
    meta = predictor.meta
    model_key = str(meta.get("selected_model") or predictor.key or "").lower()
    if model_key not in MODEL_ALGORITHMS:
        raise HTTPException(status_code=503, detail="Inference bundle selects an unsupported model.")
    algorithm, representation = MODEL_ALGORITHMS[model_key]
    model_name = str(meta.get("selected_model_name") or model_key.replace("_", " ").title())
    name = f"ST-NLP {model_name}"[:120]
    bundle_path = Path(predictor.path).expanduser().resolve()
    digest = _bundle_sha256(bundle_path)
    version = f"sha256-{digest[:16]}"
    artifact_path = (
        bundle_path.relative_to(PROJECT_ROOT).as_posix()
        if bundle_path.is_relative_to(PROJECT_ROOT)
        else str(bundle_path)
    )
    calibration = predictor.bundle.get("probability_calibrated")
    parameters: dict[str, Any] = {
        "selected_model": model_key,
        "selection_metric": meta.get("selection_metric"),
        "selection_value": meta.get("selection_value"),
        "confidence_threshold": meta.get("confidence_threshold"),
        "probability_calibrated": calibration,
        "bundle_sha256": digest,
    }

    model_version = db.scalar(
        select(ModelVersion).where(
            ModelVersion.name == name,
            ModelVersion.version == version,
        )
    )
    if model_version is None:
        model_version = ModelVersion(
            name=name,
            version=version,
            algorithm=algorithm,
            representation=representation,
            artifact_path=artifact_path,
            parameters=parameters,
            is_active=True,
        )
        db.add(model_version)
        db.flush()
    else:
        model_version.algorithm = algorithm
        model_version.representation = representation
        model_version.artifact_path = artifact_path
        model_version.parameters = parameters
        model_version.is_active = True

    db.execute(
        update(ModelVersion)
        .where(ModelVersion.id != model_version.id, ModelVersion.is_active.is_(True))
        .values(is_active=False)
    )
    return model_version


def execute_prediction(
    db: Session,
    report: Report,
    predictor: ModelPredictor,
) -> Prediction:
    """Preprocess and classify one report, persisting one complete inference run."""
    preprocessing_config = predictor.bundle.get("preprocessing", {})
    remove_stopwords_enabled = bool(preprocessing_config.get("remove_stopwords", True))

    preprocessing_started = perf_counter()
    processed = preprocess_vietnamese_text(
        report.raw_content,
        remove_stopwords_enabled=remove_stopwords_enabled,
    )
    preprocessing_ms = max(0, round((perf_counter() - preprocessing_started) * 1000))
    if not processed.processed_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Report text became empty after Vietnamese text preprocessing.",
        )

    inference_started = perf_counter()
    try:
        outputs = predictor.predict([processed.processed_text], already_processed=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The trained model could not classify this report.") from exc
    inference_ms = max(0, round((perf_counter() - inference_started) * 1000))
    if len(outputs) != 1:
        raise HTTPException(status_code=503, detail="The inference model returned an invalid result count.")

    output = outputs[0]
    category_code = str(output.get("predicted_label_code", "")).upper()
    category = db.scalar(
        select(IncidentCategory).where(
            IncidentCategory.code == category_code,
            IncidentCategory.is_active.is_(True),
        )
    )
    if category is None:
        raise HTTPException(
            status_code=503,
            detail="The model output does not match an active R01-R11 database category.",
        )

    confidence_value = output.get("confidence")
    confidence = None
    if confidence_value is not None:
        confidence = Decimal(str(confidence_value))
        if not confidence.is_finite() or confidence < 0 or confidence > 1:
            raise HTTPException(status_code=503, detail="The model returned an invalid confidence score.")

    decision_margin_value = output.get("decision_margin")
    decision_margin = (
        Decimal(str(decision_margin_value))
        if decision_margin_value is not None
        else None
    )
    class_probabilities = output.get("class_probabilities")
    if class_probabilities is not None:
        class_probabilities = {
            str(code): float(value)
            for code, value in class_probabilities.items()
        }

    reasons = [str(item) for item in output.get("warning_reasons", [])]
    needs_review = bool(output.get("needs_manual_review", False)) or confidence is None
    if confidence is None and not reasons:
        reasons.append("The selected model does not provide a calibrated confidence probability.")

    try:
        model_version = _ensure_model_version(db, predictor)
        preprocessing_run = PreprocessingRun(
            report_id=report.id,
            pipeline_version=PREPROCESSING_PIPELINE_VERSION,
            cleaned_text=processed.raw_cleaned_text,
            normalized_text=processed.normalized_text,
            segmented_text=processed.segmented_text,
            stopwords_removed_text=processed.processed_text,
            processing_time_ms=preprocessing_ms,
        )
        db.add(preprocessing_run)
        db.flush()

        prediction = Prediction(
            report_id=report.id,
            preprocessing_run_id=preprocessing_run.id,
            model_version_id=model_version.id,
            predicted_category_id=category.id,
            confidence=confidence,
            confidence_type=str(output.get("confidence_type") or (
                "unavailable" if confidence is None else "model_score"
            )),
            probability_calibrated=output.get("probability_calibrated"),
            decision_margin=decision_margin,
            class_probabilities=class_probabilities,
            inference_time_ms=inference_ms,
            needs_review=needs_review,
            review_reasons=reasons,
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        # Keep the relationships available for response serialization.
        prediction.preprocessing_run = preprocessing_run
        prediction.predicted_category = category
        prediction.model_version = model_version
        return prediction
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="Prediction results could not be saved. Check the database migration and connection.",
        ) from exc
