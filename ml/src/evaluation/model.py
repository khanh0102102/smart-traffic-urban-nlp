"""Unified evaluation for the four ST-NLP classifiers.

Consumes the held-out prediction CSVs written by Phases 8-11. It never fits a
model and never changes predictions. Every metric uses the fixed R01-R11
taxonomy so scores remain comparable when a split omits a category.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_processing.categories import CATEGORIES


MODEL_DIRS = {
    "naive_bayes": "ml/artifacts/naive_bayes",
    "svm": "ml/artifacts/svm",
    "cnn": "ml/artifacts/cnn",
    "lstm": "ml/artifacts/lstm",
}
SPLITS = ("validation", "test")
LABEL_CODES = tuple(category.code for category in CATEGORIES)
LABEL_NAMES = {category.code: category.name for category in CATEGORIES}


@dataclass
class EvaluationConfig:
    model_dirs: dict[str, str] = field(default_factory=lambda: dict(MODEL_DIRS))
    output_dir: str = "ml/artifacts/evaluation"
    splits: tuple[str, ...] = SPLITS
    # A four-model comparison must be complete by default. Set this to False
    # only for an intentional partial diagnostic run.
    require_all_models: bool = True
    confidence_threshold: float = 0.60


@dataclass
class EvaluationResult:
    evaluated_models: list[str]
    skipped_models: dict[str, str]
    evaluated_splits: list[str]
    summary_path: str
    output_dir: str


def _read_predictions(path: Path, model_name: str, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path} is empty and has no CSV header.") from exc
    required = {"actual_label_code", "predicted_label_code"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no prediction rows.")

    frame = frame.copy()
    for column in required:
        frame[column] = frame[column].astype(str).str.strip().str.upper()
        unknown = sorted(set(frame[column]) - set(LABEL_CODES))
        if unknown:
            raise ValueError(f"{path} contains non-canonical {column}: {unknown}")
        if frame[column].eq("").any():
            raise ValueError(f"{path} contains blank values in {column}.")

    if "row_index" in frame.columns:
        indices = pd.to_numeric(frame["row_index"], errors="coerce")
        if indices.isna().any() or indices.duplicated().any():
            raise ValueError(f"{path} has invalid or duplicate row_index values.")
        frame["row_index"] = indices.astype(int)
    else:
        # Legacy-compatible fallback. Current phase outputs include row_index.
        frame["row_index"] = np.arange(len(frame), dtype=int)

    if "confidence" in frame.columns:
        confidence = pd.to_numeric(frame["confidence"], errors="coerce")
        finite = confidence.dropna()
        if len(finite) and ((finite < 0.0).any() or (finite > 1.0).any()):
            raise ValueError(
                f"{path} contains confidence outside [0, 1]. "
                "Confidence must be a probability-like score; SVM NaN values are allowed."
            )
        frame["confidence"] = confidence

    frame = frame.sort_values("row_index").reset_index(drop=True)
    frame["model"] = model_name
    frame["split"] = split
    return frame


def _evaluate_frame(
    frame: pd.DataFrame,
    output: Path,
    model: str,
    split: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    y_true = frame["actual_label_code"].tolist()
    y_pred = frame["predicted_label_code"].tolist()
    labels = list(LABEL_CODES)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=labels,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prefix = f"{model}_{split}"

    confusion_frame = pd.DataFrame(cm, index=labels, columns=labels)
    confusion_frame.index.name = "actual_label_code"
    confusion_frame.columns.name = "predicted_label_code"
    confusion_file = f"confusion_matrix_{prefix}.csv"
    confusion_frame.to_csv(output / confusion_file, encoding="utf-8-sig")

    row_totals = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm.astype(float), row_totals,
        out=np.zeros_like(cm, dtype=float), where=row_totals != 0,
    )
    normalized_frame = pd.DataFrame(normalized, index=labels, columns=labels)
    normalized_frame.index.name = "actual_label_code"
    normalized_frame.columns.name = "predicted_label_code"
    normalized_file = f"confusion_matrix_normalized_{prefix}.csv"
    normalized_frame.to_csv(output / normalized_file, encoding="utf-8-sig")

    class_rows = []
    for code in labels:
        item = report[code]
        class_rows.append({
            "model": model,
            "split": split,
            "label_code": code,
            "label_name": LABEL_NAMES[code],
            "precision": float(item["precision"]),
            "recall": float(item["recall"]),
            "f1": float(item["f1-score"]),
            "support": int(item["support"]),
        })
    report_file = f"classification_report_{prefix}.csv"
    pd.DataFrame(class_rows).to_csv(
        output / report_file, index=False, encoding="utf-8-sig"
    )

    errors = frame.loc[
        frame["actual_label_code"] != frame["predicted_label_code"],
        ["row_index", "actual_label_code", "predicted_label_code"],
    ].copy()
    errors.insert(0, "split", split)
    errors.insert(0, "model", model)
    errors["actual_label_name"] = errors["actual_label_code"].map(LABEL_NAMES)
    errors["predicted_label_name"] = errors["predicted_label_code"].map(LABEL_NAMES)
    errors_file = f"errors_{prefix}.csv"
    errors.to_csv(output / errors_file, index=False, encoding="utf-8-sig")

    confusion_pairs = []
    for actual_index, actual_code in enumerate(labels):
        for predicted_index, predicted_code in enumerate(labels):
            amount = int(cm[actual_index, predicted_index])
            if amount and actual_code != predicted_code:
                confusion_pairs.append({
                    "actual_label_code": actual_code,
                    "actual_label_name": LABEL_NAMES[actual_code],
                    "predicted_label_code": predicted_code,
                    "predicted_label_name": LABEL_NAMES[predicted_code],
                    "count": amount,
                })
    confusion_pairs.sort(
        key=lambda row: (-row["count"], row["actual_label_code"], row["predicted_label_code"])
    )
    pairs_file = f"confusion_pairs_{prefix}.csv"
    pd.DataFrame(confusion_pairs, columns=[
        "actual_label_code", "actual_label_name",
        "predicted_label_code", "predicted_label_name", "count",
    ]).to_csv(output / pairs_file, index=False, encoding="utf-8-sig")

    result: dict[str, Any] = {
        "model": model,
        "split": split,
        "samples": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "weighted_precision": float(precision_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )),
        "weighted_recall": float(recall_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )),
        "misclassified_samples": int(len(errors)),
        "confusion_matrix_file": confusion_file,
        "normalized_confusion_matrix_file": normalized_file,
        "classification_report_file": report_file,
        "errors_file": errors_file,
        "confusion_pairs_file": pairs_file,
    }

    # Confidence is a review signal, not ground truth and not a factor in
    # Accuracy/Precision/Recall/F1. NaN SVM values are excluded from coverage.
    if "confidence" in frame.columns:
        confidence = frame["confidence"].dropna()
        result.update({
            "confidence_scored_samples": int(len(confidence)),
            "mean_confidence": float(confidence.mean()) if len(confidence) else None,
            "low_confidence_samples": int((confidence <= confidence_threshold).sum()),
            "low_confidence_rate": (
                float((confidence <= confidence_threshold).mean())
                if len(confidence) else None
            ),
        })
    else:
        result.update({
            "confidence_scored_samples": 0,
            "mean_confidence": None,
            "low_confidence_samples": None,
            "low_confidence_rate": None,
        })
    result["confidence_threshold"] = confidence_threshold
    return result


def evaluate_models(config: EvaluationConfig | None = None) -> EvaluationResult:
    """Evaluate saved predictions from all four models on aligned holdout rows."""
    config = config or EvaluationConfig()
    expected_models = set(MODEL_DIRS)
    configured_models = set(config.model_dirs)
    if configured_models != expected_models:
        raise ValueError(
            "model_dirs must define exactly naive_bayes, svm, cnn and lstm. "
            f"Missing: {sorted(expected_models - configured_models)}; "
            f"extra: {sorted(configured_models - expected_models)}."
        )
    if not config.splits or len(set(config.splits)) != len(config.splits) or any(
        split not in SPLITS for split in config.splits
    ):
        raise ValueError("splits must contain validation and/or test without duplicates.")
    if not 0.0 <= config.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaded: dict[tuple[str, str], pd.DataFrame] = {}
    skipped: dict[str, str] = {}
    missing_files: list[str] = []

    for model_name in MODEL_DIRS:
        directory = config.model_dirs[model_name]
        found_any = False
        missing_for_model = []
        for split in config.splits:
            path = Path(directory) / f"predictions_{split}.csv"
            if not path.exists():
                missing_files.append(f"{model_name}/{path.name}")
                missing_for_model.append(split)
                continue
            loaded[(model_name, split)] = _read_predictions(path, model_name, split)
            found_any = True
        if not found_any:
            skipped[model_name] = f"No prediction files found in {directory}."
        elif missing_for_model:
            skipped[model_name] = (
                "Missing requested split file(s): " + ", ".join(missing_for_model)
            )

    if config.require_all_models and missing_files:
        raise FileNotFoundError(
            "A complete four-model evaluation requires every requested split "
            "from all four models. Missing: " + "; ".join(missing_files)
        )
    if not loaded:
        raise FileNotFoundError(
            "No model prediction files found. Train the models in Phases 8-11 first."
        )

    # Each model must be evaluated against the same rows and true labels for
    # a given split. Sorting by row_index makes ordering differences explicit.
    for split in config.splits:
        reference: pd.DataFrame | None = None
        reference_name: str | None = None
        split_models = [
            (model_name, frame)
            for (model_name, current_split), frame in loaded.items()
            if current_split == split
        ]
        if not split_models:
            if config.require_all_models:
                raise FileNotFoundError(f"No prediction files found for split {split}.")
            continue
        for model_name, frame in split_models:
            aligned = frame[["row_index", "actual_label_code"]].sort_values(
                "row_index"
            ).reset_index(drop=True)
            if reference is None:
                reference, reference_name = aligned, model_name
            elif not aligned.equals(reference):
                raise ValueError(
                    f"Evaluation data mismatch for split '{split}': {model_name} "
                    f"does not contain the same row indices and actual labels as {reference_name}."
                )

    metrics = [
        _evaluate_frame(
            frame, output, model_name, split, config.confidence_threshold
        )
        for (model_name, split), frame in loaded.items()
    ]
    metrics_frame = pd.DataFrame(metrics).sort_values(
        ["split", "macro_f1"], ascending=[True, False]
    )
    metrics_frame.to_csv(
        output / "model_comparison.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "taxonomy": [
            {"code": code, "name": LABEL_NAMES[code]} for code in LABEL_CODES
        ],
        "models_evaluated": sorted({item["model"] for item in metrics}),
        "skipped_models_or_splits": skipped,
        "splits": list(config.splits),
        "metrics": metrics,
        "ranking": {
            split: metrics_frame.loc[
                metrics_frame["split"] == split,
                ["model", "macro_f1", "weighted_f1", "accuracy"],
            ].to_dict("records")
            for split in config.splits
        },
        "methodology": [
            "Metrics are computed from saved held-out prediction CSVs; no model is retrained.",
            "All metrics use the fixed R01-R11 label set and zero_division=0.",
            "Macro metrics weight each category equally; weighted metrics account for category support.",
            "Validation is used for model selection; test remains the final holdout.",
            "A complete comparison requires all four models and identical row_index/actual_label_code values within each split.",
            "Confidence <= configured threshold is counted as a manual-review signal only; it does not change predictions or classification metrics.",
            "Uncalibrated SVM confidence values represented by NaN are excluded from confidence summaries.",
        ],
    }
    summary_path = output / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return EvaluationResult(
        evaluated_models=sorted({item["model"] for item in metrics}),
        skipped_models=skipped,
        evaluated_splits=sorted({item["split"] for item in metrics}),
        summary_path=str(summary_path),
        output_dir=str(output),
    )
