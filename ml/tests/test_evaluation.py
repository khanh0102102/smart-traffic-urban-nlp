from __future__ import annotations
import pandas as pd
import pytest
from src.evaluation import EvaluationConfig, evaluate_models


def write_predictions(directory, rows=None):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows or [
        {"row_index": 0, "actual_label_code": "R01", "predicted_label_code": "R01"},
        {"row_index": 1, "actual_label_code": "R02", "predicted_label_code": "R03"},
        {"row_index": 2, "actual_label_code": "R03", "predicted_label_code": "R03"},
    ]).to_csv(directory / "predictions_test.csv", index=False)


def test_metrics_confusion_and_error_exports(tmp_path):
    source = tmp_path / "nb"
    write_predictions(source)
    result = evaluate_models(EvaluationConfig(
        model_dirs={"naive_bayes": str(source)},
        output_dir=str(tmp_path / "evaluation"), splits=("test",)))
    output = tmp_path / "evaluation"
    metrics = pd.read_csv(output / "model_comparison.csv")
    assert result.evaluated_models == ["naive_bayes"]
    assert metrics.loc[0, "accuracy"] == pytest.approx(2 / 3)
    matrix = pd.read_csv(output / "confusion_matrix_naive_bayes_test.csv", index_col=0)
    assert matrix.shape == (11, 11)
    assert matrix.loc["R02", "R03"] == 1
    assert len(pd.read_csv(output / "errors_naive_bayes_test.csv")) == 1


def test_models_must_use_same_actual_rows(tmp_path):
    nb, svm = tmp_path / "nb", tmp_path / "svm"
    write_predictions(nb)
    write_predictions(svm, [
        {"row_index": 0, "actual_label_code": "R01", "predicted_label_code": "R01"},
        {"row_index": 1, "actual_label_code": "R04", "predicted_label_code": "R04"},
        {"row_index": 2, "actual_label_code": "R03", "predicted_label_code": "R03"}])
    with pytest.raises(ValueError, match="same row indices and actual labels"):
        evaluate_models(EvaluationConfig(
            model_dirs={"naive_bayes": str(nb), "svm": str(svm)},
            output_dir=str(tmp_path / "evaluation"), splits=("test",)))


def test_missing_models_can_be_skipped_or_required(tmp_path):
    nb = tmp_path / "nb"
    write_predictions(nb)
    result = evaluate_models(EvaluationConfig(
        model_dirs={"naive_bayes": str(nb), "cnn": str(tmp_path / "cnn")},
        output_dir=str(tmp_path / "evaluation"), splits=("test",)))
    assert "cnn" in result.skipped_models
    with pytest.raises(FileNotFoundError, match="Missing model predictions"):
        evaluate_models(EvaluationConfig(
            model_dirs={"naive_bayes": str(nb), "lstm": str(tmp_path / "lstm")},
            output_dir=str(tmp_path / "required"), splits=("test",),
            require_all_models=True))


def test_unknown_labels_are_rejected(tmp_path):
    source = tmp_path / "nb"
    write_predictions(source, [
        {"row_index": 0, "actual_label_code": "UNKNOWN", "predicted_label_code": "R01"}])
    with pytest.raises(ValueError, match="non-canonical"):
        evaluate_models(EvaluationConfig(
            model_dirs={"naive_bayes": str(source)},
            output_dir=str(tmp_path / "evaluation"), splits=("test",)))


def test_invalid_split_is_rejected():
    with pytest.raises(ValueError, match="splits"):
        evaluate_models(EvaluationConfig(splits=("train",)))
