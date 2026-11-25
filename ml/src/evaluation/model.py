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
    require_all_models: bool = False


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
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    required = {"actual_label_code", "predicted_label_code"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no prediction rows.")
    frame = frame.copy()
    for column in required:
        frame[column] = frame[column].astype(str).str.strip().str.upper()
    valid = set(LABEL_CODES)
    for column in required:
        unknown = sorted(set(frame[column]) - valid)
        if unknown:
            raise ValueError(f"{path} contains non-canonical {column}: {unknown}")
    if "row_index" in frame.columns:
        indices = pd.to_numeric(frame["row_index"], errors="coerce")
        if indices.isna().any() or indices.duplicated().any():
            raise ValueError(f"{path} has invalid or duplicate row_index values.")
        frame["row_index"] = indices.astype(int)
    else:
        frame["row_index"] = np.arange(len(frame), dtype=int)
    frame["model"] = model_name
    frame["split"] = split
    return frame


def _evaluate_frame(frame: pd.DataFrame, output: Path, model: str, split: str) -> dict[str, Any]:
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
    confusion_frame.to_csv(output / f"confusion_matrix_{prefix}.csv", encoding="utf-8-sig")

    class_rows = []
    for code in labels:
        item = report[code]
        class_rows.append({
            "model": model, "split": split, "label_code": code,
            "label_name": LABEL_NAMES[code],
            "precision": float(item["precision"]),
            "recall": float(item["recall"]),
            "f1": float(item["f1"]),
            "support": int(item["support"]),
        })
    pd.DataFrame(class_rows).to_csv(
        output / f"classification_report_{prefix}.csv", index=False, encoding="utf-8-sig"
    )

    errors = frame.loc[
        frame["actual_label_code"] != frame["predicted_label_code"],
        ["row_index", "actual_label_code", "predicted_label_code"],
    ].copy()
    errors.insert(0, "split", split)
    errors.insert(0, "model", model)
    errors.to_csv(output / f"errors_{prefix}.csv", index=False, encoding="utf-8-sig")

    return {
        "model": model,
        "split": split,
        "samples": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix_file": f"confusion_matrix_{prefix}.csv",
        "classification_report_file": f"classification_report_{prefix}.csv",
        "errors_file": f"errors_{prefix}.csv",
    }


def evaluate_models(config: EvaluationConfig | None = None) -> EvaluationResult:
    config = config or EvaluationConfig()
    if not config.splits or any(split not in SPLITS for split in config.splits):
        raise ValueError("splits must contain validation and/or test.")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any]] = []
    class_reports: list[pd.DataFrame] = []
    loaded: dict[tuple[str, str], pd.DataFrame] = {}
    skipped: dict[str, str] = {}

    for model_name, directory in config.model_dirs.items():
        found_any = False
        for split in config.splits:
            path = Path(directory) / f"predictions_{split}.csv"
            if not path.exists():
                continue
            frame = _read_predictions(path, model_name, split)
            loaded[(model_name, split)] = frame
            found_any = True
        if not found_any:
            skipped[model_name] = f"No predictions found in {directory} for requested splits."

    if config.require_all_models and skipped:
        raise FileNotFoundError("Missing model predictions: " + "; ".join(
            f"{name}: {reason}" for name, reason in skipped.items()
        ))
    if not loaded:
        raise FileNotFoundError(
            "No model prediction files found. Train the models in Phases 8-11 first."
        )

    # For each split, compare only when the prediction files refer to the same rows.
    for split in config.splits:
        reference: pd.DataFrame | None = None
        reference_name: str | None = None
        for (model_name, current_split), frame in loaded.items():
            if current_split != split:
                continue
            aligned = frame[["row_index", "actual_label_code"]].sort_values("row_index").reset_index(drop=True)
            if reference is None:
                reference, reference_name = aligned, model_name
            elif not aligned.equals(reference):
                raise ValueError(
                    f"Evaluation data mismatch for split '{split}': {model_name} "
                    f"does not contain the same row indices and actual labels as {reference_name}."
                )

    for (model_name, split), frame in loaded.items():
        metrics.append(_evaluate_frame(frame, output, model_name, split))

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame = metrics_frame.sort_values(["split", "macro_f1"], ascending=[True, False])
    metrics_frame.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")

    summary = {
        "taxonomy": [{"code": code, "name": LABEL_NAMES[code]} for code in LABEL_CODES],
        "models_evaluated": sorted({item["model"] for item in metrics}),
        "skipped_models": skipped,
        "splits": list(config.splits),
        "metrics": metrics,
        "ranking": {
            split: metrics_frame.loc[metrics_frame["split"] == split, ["model", "macro_f1", "weighted_f1", "accuracy"]].to_dict("records")
            for split in config.splits
        },
        "methodology": [
            "Metrics are computed from saved held-out prediction CSVs; no model is retrained.",
            "All metrics use the fixed R01-R11 label set and zero_division=0.",
            "Macro metrics weight each category equally; weighted metrics account for category support.",
            "Validation and test results are reported separately. Use validation for model selection and test as the final holdout.",
            "Model comparisons require identical row_index and actual_label_code sequences within each split.",
        ],
    }
    summary_path = output / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return EvaluationResult(
        evaluated_models=sorted({item["model"] for item in metrics}),
        skipped_models=skipped,
        evaluated_splits=list(config.splits),
        summary_path=str(summary_path),
        output_dir=str(output),
    )
