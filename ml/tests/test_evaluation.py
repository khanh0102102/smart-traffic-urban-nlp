from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation import EvaluationConfig, evaluate_models
from src.evaluation.model import MODEL_DIRS
from src.data_processing.categories import CATEGORIES


ACTUAL = ["R01", "R02", "R02", "R03", "R11"]
PREDICTED = ["R01", "R03", "R02", "R03", "R10"]


def make_model_directories(root: Path):
    directories = {}
    for model_name in MODEL_DIRS:
        directory = root / model_name
        directory.mkdir(parents=True)
        directories[model_name] = str(directory)
    return directories


def write_predictions(directory: str | Path, split="test", actual=None, predicted=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    actual = list(ACTUAL if actual is None else actual)
    predicted = list(PREDICTED if predicted is None else predicted)
    pd.DataFrame({
        "row_index": range(len(actual)),
        "actual_label_code": actual,
        "predicted_label_code": predicted,
        "confidence": [0.95, 0.55, 0.80, 0.61, ""][:len(actual)],
    }).to_csv(directory / f"predictions_{split}.csv", index=False, encoding="utf-8-sig")


def populate_all_models(directories, split="test"):
    for directory in directories.values():
        write_predictions(directory, split=split)


def test_evaluates_four_models_and_exports_metrics_and_confusion_matrices(tmp_path):
    directories = make_model_directories(tmp_path / "models")
    populate_all_models(directories)
    output = tmp_path / "evaluation"

    result = evaluate_models(EvaluationConfig(
        model_dirs=directories,
        output_dir=str(output),
        splits=("test",),
    ))

    assert result.evaluated_models == sorted(MODEL_DIRS)
    assert result.evaluated_splits == ["test"]
    comparison = pd.read_csv(output / "model_comparison.csv")
    assert set(comparison["model"]) == set(MODEL_DIRS)
    assert set(comparison["split"]) == {"test"}
    assert comparison["accuracy"].iloc[0] == pytest.approx(0.6)
    assert {"macro_precision", "macro_recall", "macro_f1",
            "weighted_precision", "weighted_recall", "weighted_f1"} <= set(comparison.columns)

    matrix = pd.read_csv(output / "confusion_matrix_naive_bayes_test.csv", index_col=0)
    assert matrix.shape == (11, 11)
    assert int(matrix.to_numpy().sum()) == 5
    assert matrix.loc["R02", "R03"] == 1

    normalized = pd.read_csv(
        output / "confusion_matrix_normalized_naive_bayes_test.csv", index_col=0
    )
    assert normalized.shape == (11, 11)
    assert (output / "classification_report_naive_bayes_test.csv").is_file()
    assert (output / "errors_naive_bayes_test.csv").is_file()
    assert (output / "confusion_pairs_naive_bayes_test.csv").is_file()

    summary = json.loads((output / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert len(summary["taxonomy"]) == len(CATEGORIES) == 11
    assert summary["methodology"]


def test_rejects_models_evaluated_on_different_actual_rows(tmp_path):
    directories = make_model_directories(tmp_path / "models")
    populate_all_models(directories)
    write_predictions(
        directories["cnn"],
        actual=["R04", "R02", "R02", "R03", "R11"],
    )
    with pytest.raises(ValueError, match="same row indices and actual labels"):
        evaluate_models(EvaluationConfig(
            model_dirs=directories,
            output_dir=str(tmp_path / "evaluation"),
            splits=("test",),
        ))


def test_missing_model_split_is_rejected_by_default(tmp_path):
    directories = make_model_directories(tmp_path / "models")
    # Three have the selected test split; LSTM is missing it.
    for model_name, directory in directories.items():
        if model_name != "lstm":
            write_predictions(directory)
    with pytest.raises(FileNotFoundError, match="complete four-model evaluation"):
        evaluate_models(EvaluationConfig(
            model_dirs=directories,
            output_dir=str(tmp_path / "strict"),
            splits=("test",),
        ))


def test_partial_diagnostic_mode_can_skip_missing_models(tmp_path):
    directories = make_model_directories(tmp_path / "models")
    write_predictions(directories["naive_bayes"])
    result = evaluate_models(EvaluationConfig(
        model_dirs=directories,
        output_dir=str(tmp_path / "partial"),
        splits=("test",),
        require_all_models=False,
    ))
    assert result.evaluated_models == ["naive_bayes"]
    assert "lstm" in result.skipped_models


def test_unknown_labels_are_rejected(tmp_path):
    directories = make_model_directories(tmp_path / "models")
    populate_all_models(directories)
    write_predictions(
        directories["svm"],
        actual=["R01", "R02", "R02", "R03", "R11"],
        predicted=["UNKNOWN", "R03", "R02", "R03", "R10"],
    )
    with pytest.raises(ValueError, match="non-canonical"):
        evaluate_models(EvaluationConfig(
            model_dirs=directories,
            output_dir=str(tmp_path / "evaluation"),
            splits=("test",),
        ))


def test_invalid_split_and_model_directory_keys_are_rejected(tmp_path):
    directories = make_model_directories(tmp_path / "models")
    with pytest.raises(ValueError, match="splits"):
        evaluate_models(EvaluationConfig(model_dirs=directories, splits=("train",)))
    with pytest.raises(ValueError, match="exactly naive_bayes, svm, cnn and lstm"):
        evaluate_models(EvaluationConfig(model_dirs={"naive_bayes": directories["naive_bayes"]}))


def test_confidence_threshold_is_reported_separately_from_classification_metrics(tmp_path):
    directories = make_model_directories(tmp_path / "models")
    populate_all_models(directories)
    output = tmp_path / "evaluation"
    evaluate_models(EvaluationConfig(
        model_dirs=directories,
        output_dir=str(output),
        splits=("test",),
        confidence_threshold=0.60,
    ))
    metrics = pd.read_csv(output / "model_comparison.csv")
    naive_bayes = metrics.loc[metrics["model"] == "naive_bayes"].iloc[0]
    assert naive_bayes["confidence_scored_samples"] == 4
    assert naive_bayes["low_confidence_samples"] == 1
    assert naive_bayes["accuracy"] == pytest.approx(0.6)


def test_rejects_confidence_outside_zero_to_one(tmp_path):
    directories = make_model_directories(tmp_path / "models")
    populate_all_models(directories)
    path = Path(directories["naive_bayes"]) / "predictions_test.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "confidence"] = 1.5
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="confidence outside"):
        evaluate_models(EvaluationConfig(
            model_dirs=directories,
            output_dir=str(tmp_path / "evaluation"),
            splits=("test",),
        ))
