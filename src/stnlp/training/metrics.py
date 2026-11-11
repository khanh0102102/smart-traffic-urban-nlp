"""Common multi-class metrics and evaluation artifact writers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from stnlp.labels import LABEL_CODES, LABEL_TO_NAME


def calculate_metrics(y_true, y_pred) -> dict:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABEL_CODES), average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABEL_CODES), average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "macro_f1": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
    }


def save_evaluation(output_dir: Path, model_name: str, y_true, y_pred) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = calculate_metrics(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(LABEL_CODES),
        target_names=[LABEL_TO_NAME[c] for c in LABEL_CODES],
        output_dict=True,
        zero_division=0,
    )
    (output_dir / f"{model_name}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"{model_name}_classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(y_true, y_pred, labels=list(LABEL_CODES))
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(LABEL_CODES)),
        yticks=range(len(LABEL_CODES)),
        xticklabels=LABEL_CODES,
        yticklabels=LABEL_CODES,
        xlabel="Nhãn dự đoán",
        ylabel="Nhãn thực tế",
        title=f"Confusion Matrix — {model_name}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col, row, str(matrix[row, col]),
                ha="center", va="center",
                color="white" if matrix[row, col] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return metrics
