"""Tests for linear SVM training, probability calibration and inference."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.svm import SVMConfig, predict_svm_reports, train_svm


def _prepare_tfidf_artifacts(
    directory: Path,
    train_labels: list[str] | None = None,
) -> None:
    directory.mkdir(parents=True)
    train_texts = [
        "duong o_ga hu_hong",
        "duong mat_duong nut",
        "duong sat lun",
        "den_duong chieu_sang",
        "den_tin_hieu giao_thong",
        "den_bi_hong",
    ]
    train_labels = train_labels or ["R01", "R01", "R01", "R02", "R02", "R02"]
    validation_texts = ["duong o_ga", "den_duong"]
    test_texts = ["duong nut", "den_tin_hieu"]

    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
    )
    x_train = vectorizer.fit_transform(train_texts).tocsr()
    x_validation = vectorizer.transform(validation_texts).tocsr()
    x_test = vectorizer.transform(test_texts).tocsr()
    joblib.dump(vectorizer, directory / "tfidf_vectorizer.joblib")
    save_npz(directory / "X_train.npz", x_train)
    save_npz(directory / "X_validation.npz", x_validation)
    save_npz(directory / "X_test.npz", x_test)

    for filename, labels in (
        ("y_train.csv", train_labels),
        ("y_validation.csv", ["R01", "R02"]),
        ("y_test.csv", ["R01", "R02"]),
    ):
        pd.DataFrame({
            "row_index": range(len(labels)),
            "label_code": labels,
        }).to_csv(directory / filename, index=False, encoding="utf-8-sig")


def test_svm_trains_calibrates_saves_and_predicts(tmp_path):
    tfidf_dir = tmp_path / "tfidf"
    _prepare_tfidf_artifacts(tfidf_dir)
    output = tmp_path / "svm"

    result = train_svm(SVMConfig(
        tfidf_dir=str(tfidf_dir),
        output_dir=str(output),
        C=1.0,
        calibration_cv=3,
        top_features=5,
    ))

    assert result.train_rows == 6
    assert result.validation_rows == 2
    assert result.test_rows == 2
    assert result.feature_count > 0
    assert result.trained_class_count == 2
    assert result.probability_calibrated is True
    assert result.calibration_method == "sigmoid"
    assert result.calibration_cv == 3
    assert result.calibration_reason is None

    for filename in (
        "svm_bundle.joblib",
        "svm_summary.json",
        "predictions_validation.csv",
        "predictions_test.csv",
        "confusion_matrix_validation.csv",
        "confusion_matrix_test.csv",
        "top_features_by_class.csv",
    ):
        assert (output / filename).exists()

    bundle = joblib.load(output / "svm_bundle.joblib")
    assert bundle["probability_calibrated"] is True
    predictions = predict_svm_reports(
        ["duong o_ga", "tu_khoa_khong_co_trong_train"],
        model_path=output / "svm_bundle.joblib",
        already_processed=True,
    )
    assert len(predictions) == 2
    assert predictions[0]["predicted_label_code"] in {"R01", "R02"}
    assert predictions[0]["probability_calibrated"] is True
    assert 0 <= predictions[0]["confidence"] <= 1
    assert sum(predictions[0]["class_probabilities"].values()) == pytest.approx(1.0)
    assert predictions[0]["has_known_features"] is True
    assert predictions[1]["has_known_features"] is False
    assert predictions[1]["warning"] is not None


def test_svm_falls_back_to_un_calibrated_margins_for_singleton_class(tmp_path):
    tfidf_dir = tmp_path / "tfidf"
    _prepare_tfidf_artifacts(
        tfidf_dir,
        train_labels=["R01", "R01", "R01", "R02", "R02", "R03"],
    )
    output = tmp_path / "svm"

    result = train_svm(SVMConfig(
        tfidf_dir=str(tfidf_dir),
        output_dir=str(output),
    ))

    assert result.probability_calibrated is False
    assert result.calibration_cv is None
    assert result.calibration_reason is not None
    predictions = predict_svm_reports(
        ["duong o_ga"],
        model_path=output / "svm_bundle.joblib",
        already_processed=True,
    )
    assert predictions[0]["confidence"] is None
    assert predictions[0]["probability_calibrated"] is False
    assert predictions[0]["decision_margin"] != 0


def test_svm_rejects_matrix_label_row_mismatch(tmp_path):
    tfidf_dir = tmp_path / "tfidf"
    _prepare_tfidf_artifacts(tfidf_dir)
    pd.DataFrame({"row_index": [0], "label_code": ["R01"]}).to_csv(
        tfidf_dir / "y_train.csv", index=False, encoding="utf-8-sig"
    )
    with pytest.raises(ValueError, match="row mismatch"):
        train_svm(SVMConfig(
            tfidf_dir=str(tfidf_dir),
            output_dir=str(tmp_path / "out"),
        ))


def test_svm_requires_at_least_two_training_classes(tmp_path):
    tfidf_dir = tmp_path / "tfidf"
    _prepare_tfidf_artifacts(
        tfidf_dir,
        train_labels=["R01"] * 6,
    )
    with pytest.raises(ValueError, match="at least two different"):
        train_svm(SVMConfig(
            tfidf_dir=str(tfidf_dir),
            output_dir=str(tmp_path / "out"),
        ))


def test_svm_validates_hyperparameters():
    with pytest.raises(ValueError, match="C"):
        train_svm(SVMConfig(C=0))
    with pytest.raises(ValueError, match="calibration_method"):
        train_svm(SVMConfig(calibration_method="invalid"))

def test_svm_rejects_isotonic_calibration_for_small_dataset(tmp_path):
    tfidf_dir = tmp_path / "tfidf"
    _prepare_tfidf_artifacts(tfidf_dir)
    with pytest.raises(ValueError, match="3000 total"):
        train_svm(SVMConfig(
            tfidf_dir=str(tfidf_dir),
            output_dir=str(tmp_path / "out"),
            calibration_method="isotonic",
        ))


def test_svm_can_train_with_calibration_disabled(tmp_path):
    tfidf_dir = tmp_path / "tfidf"
    _prepare_tfidf_artifacts(tfidf_dir)
    result = train_svm(SVMConfig(
        tfidf_dir=str(tfidf_dir),
        output_dir=str(tmp_path / "svm"),
        calibrate_probabilities=False,
    ))
    assert result.probability_calibrated is False
    assert result.calibration_reason == "Probability calibration disabled by configuration."

