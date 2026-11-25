"""Shared cached inference API for the selected ST-NLP classifier."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_processing.categories import CATEGORIES
from src.embedding import SequenceTokenizer, pad_sequences
from src.inference.selection import MODEL_FILES, MODEL_NAMES
from src.nlp_preprocessing import preprocess_vietnamese_text


class ModelPredictor:
    """Load the packaged estimator once; reuse it for all calls to predict()."""
    def __init__(self, bundle_path="ml/artifacts/inference/best_model_bundle.joblib"):
        self.path = Path(bundle_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"{self.path} not found. Run ml/scripts/build_inference.py first.")
        self.bundle = joblib.load(self.path)
        self.meta = self.bundle.get("_inference", {})
        self.key = self.meta.get("selected_model")
        if self.key not in MODEL_FILES:
            raise ValueError("Unsupported or invalid inference bundle.")
        self.threshold = float(self.meta.get("confidence_threshold", .60))
        self.encoder = self.bundle.get("label_encoder")
        codes = np.asarray([c.code for c in CATEGORIES])
        if not isinstance(self.encoder, LabelEncoder) or not np.array_equal(self.encoder.classes_, codes):
            raise ValueError("Inference bundle must use canonical R01-R11 labels.")
        self.names = {c.code: c.name for c in CATEGORIES}
        self.model = self.bundle.get("model")
        self.keras_model = None
        if self.key in {"naive_bayes", "svm"}:
            if not isinstance(self.bundle.get("vectorizer"), TfidfVectorizer) or self.model is None:
                raise ValueError("Missing sklearn estimator or TF-IDF vectorizer.")
        else:
            self._load_keras()

    def _load_keras(self):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("Install TensorFlow using ml/requirements-embedding.txt for CNN/LSTM inference.") from exc
        path = self.path.parent / str(self.bundle.get("model_file") or "")
        if not path.is_file():
            raise FileNotFoundError(f"Packaged Keras model missing: {path}")
        if self.key == "cnn":
            from src.cnn.model import _register_custom_layers
            pool, zero = _register_custom_layers(tf)
            self.keras_model = tf.keras.models.load_model(
                path, custom_objects={"MaskedGlobalMaxPooling1D": pool, "ZeroPadEmbeddingOutput": zero},
                compile=False)
        else:
            self.keras_model = tf.keras.models.load_model(path, compile=False)

    def predict(self, texts: Iterable[object], *, already_processed=False) -> list[dict[str, Any]]:
        raw = ["" if x is None else str(x) for x in texts]
        if not raw:
            return []
        if any(not x.strip() for x in raw):
            raise ValueError("Every report must contain non-empty text.")
        if already_processed:
            processed = raw
        else:
            remove = bool(self.bundle.get("preprocessing", {}).get("remove_stopwords", True))
            processed = [preprocess_vietnamese_text(x, remove_stopwords_enabled=remove).processed_text for x in raw]
        return self._tfidf(raw, processed) if self.key in {"naive_bayes", "svm"} else self._sequence(raw, processed)

    def _result(self, raw, text, class_id, confidence, confidence_type, class_probs, known,
                calibrated, margin=None):
        code = str(self.encoder.inverse_transform([int(class_id)])[0])
        reasons = []
        if not known:
            reasons.append("No token from this report was found in the trained vocabulary.")
        if confidence is None:
            reasons.append("SVM calibration is unavailable; decision margin is not a probability.")
        elif confidence <= self.threshold:
            reasons.append(f"Confidence is at or below the {self.threshold:.0%} manual-review threshold.")
        return {
            "raw_content": raw, "processed_text": text, "model_key": self.key,
            "model_name": MODEL_NAMES[self.key], "predicted_label_code": code,
            "predicted_label_name": self.names[code], "confidence": confidence,
            "confidence_type": confidence_type, "probability_calibrated": calibrated,
            "decision_margin": margin, "class_probabilities": class_probs,
            "has_known_features": known, "needs_manual_review": bool(reasons),
            "warning_reasons": reasons,
        }

    def _tfidf(self, raw, processed):
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        matrix = self.bundle["vectorizer"].transform(processed).tocsr()
        ids = self.model.predict(matrix).astype(int)
        prob = None
        margins = [None] * len(raw)
        if self.key == "naive_bayes" or bool(self.bundle.get("probability_calibrated", False)):
            prob = np.asarray(self.model.predict_proba(matrix), dtype=float)
            classes = np.asarray(self.model.classes_, dtype=int)
            kind = "model_score" if self.key == "naive_bayes" else "calibrated_probability"
            calibrated = None if self.key == "naive_bayes" else True
        else:
            from src.svm.model import _decision_scores, _model_classes
            classes = _model_classes(self.model)
            score = _decision_scores(self.model, matrix)
            positions = {int(v): i for i, v in enumerate(classes)}
            margins = [float(score[i, positions[int(cid)]]) for i, cid in enumerate(ids)]
            kind, calibrated = "unavailable", False
        rows = []
        for i, cid in enumerate(ids):
            probs = None if prob is None else {
                str(self.encoder.inverse_transform([int(k)])[0]): float(prob[i, j])
                for j, k in enumerate(classes)}
            conf = None if prob is None else float(prob[i].max())
            rows.append(self._result(raw[i], processed[i], cid, conf, kind, probs,
                                     bool(matrix[i].nnz), calibrated, margins[i]))
        return rows

    def _sequence(self, raw, processed):
        tokenizer, cfg = self.bundle.get("tokenizer"), self.bundle.get("sequence")
        if not isinstance(tokenizer, SequenceTokenizer) or not isinstance(cfg, dict):
            raise ValueError("Neural bundle is missing tokenizer or sequence configuration.")
        seq = tokenizer.texts_to_sequences(processed)
        matrix = pad_sequences(seq, int(cfg["max_length"]),
                               padding=cfg.get("padding", "post"),
                               truncating=cfg.get("truncating", "post"))
        probs = np.asarray(self.keras_model.predict(matrix, batch_size=256, verbose=0), dtype=float)
        if probs.ndim != 2 or probs.shape[1] != len(self.encoder.classes_):
            raise ValueError("Neural output does not match the canonical 11 categories.")
        ids = probs.argmax(axis=1)
        return [
            self._result(raw[i], processed[i], int(cid), float(probs[i, cid]), "model_score",
                         {str(self.encoder.inverse_transform([j])[0]): float(v)
                          for j, v in enumerate(probs[i])},
                         bool(np.any(matrix[i] > 1)), False)
            for i, cid in enumerate(ids)
        ]
