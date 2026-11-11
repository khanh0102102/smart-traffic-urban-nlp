"""Load a trained artifact and expose a consistent prediction interface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from stnlp.config import Settings, settings
from stnlp.labels import LABEL_CODES, LABEL_TO_NAME
from stnlp.preprocessing.text import preprocess_text


class ModelNotReadyError(RuntimeError):
    """Raised when a trained artifact is not available or cannot be loaded."""


@dataclass
class Prediction:
    predicted_code: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str
    model_version: str


class ModelPredictor:
    def __init__(self, config: Settings = settings):
        self.config = config
        self.model_name = config.model_name
        self.model = None
        self.tokenizer = None
        self.max_len: int | None = None
        self.labels = list(LABEL_CODES)
        self.model_version = "unversioned"
        self.load_error: str | None = None
        self._load()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _load(self) -> None:
        model_dir = Path(self.config.model_artifact_dir) / self.model_name
        try:
            classical_path = model_dir / "model.joblib"
            deep_path = model_dir / "model.keras"
            if classical_path.exists():
                self.model = joblib.load(classical_path)
                classifier = self.model.named_steps.get("classifier")
                classes = getattr(classifier, "classes_", None)
                if classes is not None:
                    self.labels = [str(value) for value in classes]
            elif deep_path.exists():
                from tensorflow.keras.models import load_model

                self.model = load_model(deep_path)
                tokenizer_path = model_dir / "tokenizer.json"
                metadata_path = model_dir / "metadata.json"
                labels_path = model_dir / "labels.json"
                if not tokenizer_path.exists() or not metadata_path.exists():
                    raise FileNotFoundError("Thiếu tokenizer.json hoặc metadata.json.")
                self.tokenizer = tokenizer_from_json(
                    tokenizer_path.read_text(encoding="utf-8")
                )
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.max_len = int(metadata["max_len"])
                if labels_path.exists():
                    self.labels = json.loads(labels_path.read_text(encoding="utf-8"))
            else:
                raise FileNotFoundError(f"Không tìm thấy model.joblib hoặc model.keras trong {model_dir}")
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.model_version = str(metadata.get("model_version", metadata.get("trained_at", "v1")))
        except Exception as exc:
            self.model = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    def predict(self, text: str, include_probabilities: bool = True) -> Prediction:
        if not self.ready:
            raise ModelNotReadyError(
                self.load_error or "Chưa có model artifact. Hãy huấn luyện mô hình trước."
            )
        if not text or not text.strip():
            raise ValueError("Nội dung phản ánh không được để trống.")

        if self.model_name in {"naive_bayes", "svm"}:
            predicted = str(self.model.predict([text])[0])
            raw_probabilities = self.model.predict_proba([text])[0]
            class_codes = [str(value) for value in self.model.classes_]
        else:
            cleaned = preprocess_text(text)
            sequence = self.tokenizer.texts_to_sequences([cleaned])
            x_value = pad_sequences(
                sequence, maxlen=self.max_len, padding="post", truncating="post"
            )
            raw_probabilities = self.model.predict(x_value, verbose=0)[0]
            predicted_index = int(np.argmax(raw_probabilities))
            predicted = str(self.labels[predicted_index])
            class_codes = list(self.labels)

        probability_map = {
            code: float(probability)
            for code, probability in zip(class_codes, raw_probabilities)
        }
        confidence = float(probability_map[predicted])
        # Normalize ordering to canonical labels, while preserving all class scores.
        ordered = {code: probability_map.get(code, 0.0) for code in LABEL_CODES}
        return Prediction(
            predicted_code=predicted,
            confidence=confidence,
            probabilities=ordered if include_probabilities else {},
            model_name=self.model_name,
            model_version=self.model_version,
        )


@lru_cache(maxsize=1)
def get_predictor() -> ModelPredictor:
    return ModelPredictor()
