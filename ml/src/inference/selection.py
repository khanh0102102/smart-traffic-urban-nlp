"""Select and package the best model using validation Macro F1 only."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.data_processing.categories import CATEGORIES
from src.embedding import SequenceTokenizer

MODEL_FILES = {
    "naive_bayes": "naive_bayes_bundle.joblib",
    "svm": "svm_bundle.joblib",
    "cnn": "cnn_bundle.joblib",
    "lstm": "lstm_bundle.joblib",
}
MODEL_NAMES = {
    "naive_bayes": "Naive Bayes", "svm": "SVM",
    "cnn": "1D-CNN", "lstm": "LSTM",
}
DEFAULT_MODEL_DIRS = {key: f"ml/artifacts/{key}" for key in MODEL_FILES}


@dataclass
class InferenceBuildResult:
    selected_model: str
    selected_model_name: str
    selection_value: float
    bundle_path: str
    metadata_path: str
    output_dir: str


def _load_ranking(evaluation_dir: Path) -> pd.DataFrame:
    path = evaluation_dir / "model_comparison.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run Phase 12 first: python ml/scripts/evaluate_models.py"
        )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    needed = {"model", "split", "macro_f1", "weighted_f1", "accuracy"}
    if not needed.issubset(frame.columns):
        raise ValueError(f"{path} is missing columns: {sorted(needed - set(frame.columns))}")
    frame["model"] = frame["model"].astype(str).str.strip().str.lower()
    rows = frame[frame["split"].astype(str).str.lower() == "validation"].copy()
    if set(rows["model"]) != set(MODEL_FILES) or rows["model"].duplicated().any():
        raise ValueError("Need exactly one validation score for each of Naive Bayes, SVM, CNN and LSTM.")
    for column in ("macro_f1", "weighted_f1", "accuracy"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
        values = rows[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Validation metric {column} must be finite and between 0 and 1.")
    return rows.sort_values(
        ["macro_f1", "weighted_f1", "accuracy", "model"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _validate_bundle(key: str, bundle: dict, path: Path) -> None:
    if not isinstance(bundle, dict):
        raise ValueError(f"Invalid model bundle: {path}")
    encoder = bundle.get("label_encoder")
    expected = np.asarray([category.code for category in CATEGORIES])
    if not isinstance(encoder, LabelEncoder) or not np.array_equal(encoder.classes_, expected):
        raise ValueError(f"{path} must use the canonical R01-R11 label encoder.")
    if key in {"naive_bayes", "svm"}:
        if not isinstance(bundle.get("vectorizer"), TfidfVectorizer):
            raise TypeError(f"{path} is missing its fitted TF-IDF vectorizer.")
        if key == "naive_bayes" and not isinstance(bundle.get("model"), MultinomialNB):
            raise TypeError(f"{path} is not a MultinomialNB bundle.")
        if key == "svm" and not isinstance(bundle.get("model"), (LinearSVC, CalibratedClassifierCV)):
            raise TypeError(f"{path} is not a supported SVM bundle.")
    else:
        expected_type = "TensorFlow/Keras 1D TextCNN" if key == "cnn" else "TensorFlow/Keras LSTM text classifier"
        if bundle.get("model_type") != expected_type:
            raise ValueError(f"{path} is not a supported {MODEL_NAMES[key]} bundle.")
        if not isinstance(bundle.get("tokenizer"), SequenceTokenizer) or "sequence" not in bundle:
            raise ValueError(f"{path} is missing tokenizer/sequence metadata.")


def build_inference_bundle(
    evaluation_dir: str | Path = "ml/artifacts/evaluation",
    model_dirs: dict[str, str | Path] | None = None,
    output_dir: str | Path = "ml/artifacts/inference",
    confidence_threshold: float = 0.60,
) -> InferenceBuildResult:
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    dirs = {key: Path(value) for key, value in (model_dirs or DEFAULT_MODEL_DIRS).items()}
    if set(dirs) != set(MODEL_FILES):
        raise ValueError("model_dirs must contain exactly naive_bayes, svm, cnn and lstm.")
    ranking = _load_ranking(Path(evaluation_dir))
    best = ranking.iloc[0]
    key = str(best["model"])
    source = dirs[key] / MODEL_FILES[key]
    if not source.is_file():
        raise FileNotFoundError(f"Selected {MODEL_NAMES[key]} model bundle not found: {source}")
    bundle = joblib.load(source)
    _validate_bundle(key, bundle, source)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    packaged = dict(bundle)
    keras_file = None
    if key in {"cnn", "lstm"}:
        keras_source = dirs[key] / str(bundle.get("model_file", ""))
        if not keras_source.is_file():
            raise FileNotFoundError(f"Selected Keras model file not found: {keras_source}")
        keras_target = output / "best_model.keras"
        shutil.copy2(keras_source, keras_target)
        packaged["model_file"] = keras_target.name
        keras_file = keras_target.name
    elif (output / "best_model.keras").exists():
        (output / "best_model.keras").unlink()

    packaged["_inference"] = {
        "schema_version": 1,
        "selected_model": key,
        "selected_model_name": MODEL_NAMES[key],
        "selection_metric": "validation_macro_f1",
        "selection_value": float(best["macro_f1"]),
        "validation_weighted_f1": float(best["weighted_f1"]),
        "validation_accuracy": float(best["accuracy"]),
        "confidence_threshold": float(confidence_threshold),
        "source_bundle": str(source),
    }
    bundle_path = output / "best_model_bundle.joblib"
    joblib.dump(packaged, bundle_path, compress=3)
    metadata = {
        **packaged["_inference"],
        "selection_rule": "Highest validation Macro F1; tie-break Weighted F1, Accuracy, then model key.",
        "bundle_file": bundle_path.name,
        "keras_model_file": keras_file,
        "note": "Model choice uses validation only; test scores are never used for selection.",
        "taxonomy": [{"code": c.code, "name": c.name} for c in CATEGORIES],
    }
    metadata_path = output / "model_selection.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return InferenceBuildResult(
        selected_model=key,
        selected_model_name=MODEL_NAMES[key],
        selection_value=float(best["macro_f1"]),
        bundle_path=str(bundle_path),
        metadata_path=str(metadata_path),
        output_dir=str(output),
    )
