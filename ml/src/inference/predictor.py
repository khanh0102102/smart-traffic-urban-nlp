"""Shared single/batch prediction API for the selected ST-NLP model."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.data_processing.categories import CATEGORIES
from src.embedding import SequenceTokenizer, pad_sequences
from src.inference.selection import MODEL_FILES, MODEL_NAMES
from src.nlp_preprocessing import preprocess_vietnamese_text


class ModelPredictor:
    """Load the selected model once, then reuse it for repeated predictions."""

    def __init__(self, bundle_path: str | Path = "ml/artifacts/inference/best_model_bundle.joblib"):
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
        self.threshold = float(self.metadata.get("confidence_threshold", 0.60))
        if not 0 <= self.threshold <= 1:
            raise ValueError("Invalid confidence threshold in inference bundle.")
        self.encoder = self.bundle.get("label_encoder")
        expected = np.asarray([category.code for category in CATEGORIES])
        if not isinstance(self.encoder, LabelEncoder) or not np.array_equal(self.encoder.classes_, expected):
            raise ValueError("Inference bundle must use the canonical R01-R11 label encoder.")
        self.category_names = {category.code: category.name for category in CATEGORIES}
        self.model = self.bundle.get("model")
        self.keras_model = None

        if self.model_key in {"naive_bayes", "svm"}:
            if not isinstance(self.bundle.get("vectorizer"), TfidfVectorizer):
                raise ValueError("Inference bundle is missing its fitted TF-IDF vectorizer.")
            if self.model is None or not hasattr(self.model, "predict"):
                raise ValueError("Inference bundle is missing its fitted classifier.")
        else:
            self._load_neural_model()

    def _load_neural_model(self) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is required by the selected CNN/LSTM model. "
                "Install it using: python -m pip install -r ml/requirements-embedding.txt"
            ) from exc
        model_file = self.bundle.get("model_file")
        path = self.bundle_path.parent / str(model_file or "")
        if not path.is_file():
            raise FileNotFoundError(f"Packaged Keras model file not found: {path}")
        if self.model_key == "cnn":
            from src.cnn.model import _register_custom_layers
            pooling_layer, zero_pad_layer = _register_custom_layers(tf)
            self.keras_model = tf.keras.models.load_model(
                path,
                custom_objects={
                    "MaskedGlobalMaxPooling1D": pooling_layer,
                    "ZeroPadEmbeddingOutput": zero_pad_layer,
                },
                compile=False,
            )
        else:
            self.keras_model = tf.keras.models.load_model(path, compile=False)

    def _preprocess(self, text: str, already_processed: bool) -> str:
        if already_processed:
            return text
        remove_stopwords = bool(self.bundle.get("preprocessing", {}).get("remove_stopwords", True))
        return preprocess_vietnamese_text(
            text, remove_stopwords_enabled=remove_stopwords
        ).processed_text

    def predict(
        self,
        texts: Iterable[object],
        *,
        already_processed: bool = False,
    ) -> list[dict[str, Any]]:
        raw = ["" if value is None else str(value) for value in texts]
        if not raw:
            return []
        if any(not value.strip() for value in raw):
            raise ValueError("Every report must contain non-empty text.")
        processed = [self._preprocess(value, already_processed) for value in raw]
        if self.model_key in {"naive_bayes", "svm"}:
            return self._predict_tfidf(raw, processed)
        return self._predict_sequence(raw, processed)

    def _format(
        self,
        raw: str,
        processed: str,
        class_id: int,
        confidence: float | None,
        confidence_type: str,
        probabilities: dict[str, float] | None,
        known_features: bool,
        calibrated: bool | None,
        decision_margin: float | None = None,
    ) -> dict[str, Any]:
        code = str(self.encoder.inverse_transform([int(class_id)])[0])
        reasons = []
        if not known_features:
            reasons.append("No trained vocabulary token was found in the report.")
        if confidence is None:
            reasons.append("SVM probability calibration is unavailable; decision margin is not a probability.")
        elif confidence <= self.threshold:
            reasons.append(f"Confidence is at or below the {self.threshold:.0%} manual-review threshold.")
        return {
            "raw_content": raw,
            "processed_text": processed,
            "model_key": self.model_key,
            "model_name": MODEL_NAMES[self.model_key],
            "predicted_label_code": code,
            "predicted_label_name": self.category_names[code],
            "confidence": confidence,
            "confidence_type": confidence_type,
            "probability_calibrated": calibrated,
            "decision_margin": decision_margin,
            "class_probabilities": probabilities,
            "has_known_features": known_features,
            "needs_manual_review": bool(reasons),
            "warning_reasons": reasons,
        }

    def _predict_tfidf(self, raw: list[str], processed: list[str]) -> list[dict[str, Any]]:
        matrix = self.bundle["vectorizer"].transform(processed).tocsr()
        predicted = self.model.predict(matrix).astype(np.int64)
        probability_values = None
        margins: list[float | None] = [None] * len(raw)

        if self.model_key == "naive_bayes":
            probability_values = np.asarray(self.model.predict_proba(matrix), dtype=float)
            classes = np.asarray(self.model.classes_, dtype=np.int64)
            confidence_type, calibrated = "model_score", None
        elif bool(self.bundle.get("probability_calibrated", False)):
            probability_values = np.asarray(self.model.predict_proba(matrix), dtype=float)
            classes = np.asarray(self.model.classes_, dtype=np.int64)
            confidence_type, calibrated = "calibrated_probability", True
        else:
            from src.svm.model import _decision_scores, _model_classes
            classes = _model_classes(self.model)
            scores = _decision_scores(self.model, matrix)
            class_columns = {int(value): index for index, value in enumerate(classes)}
            margins = [
                float(scores[row, class_columns[int(class_id)]])
                for row, class_id in enumerate(predicted)
            ]
            confidence_type, calibrated = "unavailable", False

        output = []
        for row, class_id in enumerate(predicted):
            probs = None
            confidence = None
            if probability_values is not None:
                probs = {
                    str(self.encoder.inverse_transform([int(model_class)])[0]): float(probability_values[row, col])
                    for col, model_class in enumerate(classes)
                }
                confidence = float(probability_values[row].max())
            output.append(self._format(
                raw[row], processed[row], int(class_id), confidence, confidence_type,
                probs, bool(matrix[row].nnz), calibrated, margins[row],
            ))
        return output

    def _predict_sequence(self, raw: list[str], processed: list[str]) -> list[dict[str, Any]]:
        tokenizer = self.bundle.get("tokenizer")
        sequence = self.bundle.get("sequence")
        if not isinstance(tokenizer, SequenceTokenizer) or not isinstance(sequence, dict):
            raise ValueError("Neural inference bundle is missing tokenizer/sequence metadata.")
        encoded = tokenizer.texts_to_sequences(processed)
        matrix = pad_sequences(
            encoded,
            int(sequence["max_length"]),
            padding=sequence.get("padding", "post"),
            truncating=sequence.get("truncating", "post"),
        )
        values = np.asarray(self.keras_model.predict(matrix, batch_size=256, verbose=0), dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.encoder.classes_):
            raise ValueError("Neural model output does not match canonical R01-R11 labels.")
        predicted = values.argmax(axis=1).astype(np.int64)
        output = []
        for row, class_id in enumerate(predicted):
            probs = {
                str(self.encoder.inverse_transform([class_id])[0]) if index == class_id
                else str(self.encoder.inverse_transform([index])[0]): float(value)
                for index, value in enumerate(values[row])
            }
            output.append(self._format(
                raw[row], processed[row], int(class_id), float(values[row, class_id]),
                "model_score", probs, bool(np.any(matrix[row] > 1)), False,
            ))
        return output
