"""Unified dispatch API over the four model-specific prediction functions."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
import joblib
from src.inference.selection import MODEL_FILES, MODEL_NAMES


class ModelPredictor:
    """One prediction interface regardless of the selected model architecture."""

    def __init__(self, bundle_path="ml/artifacts/inference/best_model_bundle.joblib"):
        self.bundle_path = Path(bundle_path)
        if not self.bundle_path.is_file():
            raise FileNotFoundError(
                f"{self.bundle_path} not found. Run ml/scripts/build_inference.py first."
            )
        self.bundle = joblib.load(self.bundle_path)
        if not isinstance(self.bundle, dict) or not isinstance(self.bundle.get("_inference"), dict):
            raise ValueError("Invalid best_model_bundle.joblib format.")
        self.metadata = self.bundle["_inference"]
        self.model_key = self.metadata.get("selected_model")
        if self.model_key not in MODEL_FILES:
            raise ValueError("Inference bundle contains an unsupported selected_model.")
        self.model_name = MODEL_NAMES[self.model_key]
        self.threshold = float(self.metadata.get("confidence_threshold", 0.60))
        if not 0 <= self.threshold <= 1:
            raise ValueError("Invalid confidence threshold in inference bundle.")

    def predict(self, texts: Iterable[object], *, already_processed: bool = False) -> list[dict[str, Any]]:
        values = ["" if value is None else str(value) for value in texts]
        if not values:
            return []
        if any(not value.strip() for value in values):
            raise ValueError("Every report must contain non-empty text.")

        path = str(self.bundle_path)
        if self.model_key == "naive_bayes":
            from src.naive_bayes.model import predict_reports
            predictions = predict_reports(values, model_path=path, already_processed=already_processed)
        elif self.model_key == "svm":
            from src.svm.model import predict_svm_reports
            predictions = predict_svm_reports(values, model_path=path, already_processed=already_processed)
        elif self.model_key == "cnn":
            from src.cnn.model import predict_cnn_reports
            predictions = predict_cnn_reports(values, model_path=path, already_processed=already_processed)
        else:
            from src.lstm.model import predict_lstm_reports
            predictions = predict_lstm_reports(values, model_path=path, already_processed=already_processed)

        results = []
        for row in predictions:
            item = dict(row)
            item["model_key"] = self.model_key
            item["model_name"] = self.model_name
            if "has_known_features" not in item:
                item["has_known_features"] = bool(item.pop("has_known_tokens", True))
            confidence = item.get("confidence")
            if self.model_key == "svm":
                calibrated = bool(item.get("probability_calibrated", False))
                item["confidence_type"] = "calibrated_probability" if calibrated else "unavailable"
            else:
                calibrated = False if self.model_key in {"cnn", "lstm"} else None
                item["confidence_type"] = "model_score"
                item.setdefault("probability_calibrated", calibrated)
            reasons = []
            if not item["has_known_features"]:
                reasons.append("No token from the report was found in the trained vocabulary.")
            if confidence is None:
                reasons.append("SVM probability calibration is unavailable; decision margin is not a probability.")
            elif float(confidence) <= self.threshold:
                reasons.append(f"Confidence is at or below the {self.threshold:.0%} manual-review threshold.")
            if item.get("warning") and not reasons:
                reasons.append(str(item["warning"]))
            item["needs_manual_review"] = bool(reasons)
            item["warning_reasons"] = reasons
            item["confidence_threshold"] = self.threshold
            results.append(item)
        return results
