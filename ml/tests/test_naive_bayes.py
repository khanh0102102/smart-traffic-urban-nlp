"""Tests for the TF-IDF Multinomial Naive Bayes baseline."""
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.naive_bayes.model as naive_bayes_module

from src.naive_bayes import (
    NaiveBayesConfig,
    predict_reports,
    train_naive_bayes,
)


def _prepare_tfidf_artifacts(directory: Path) -> None:
    directory.mkdir(parents=True)
    train_texts = [
        "duong o_ga hu_hong",
        "duong mat_duong nut",
        "den_duong chieu_sang",
        "den_tin_hieu den_giao_thong",
    ]
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
        ("y_train.csv", ["R01", "R01", "R02", "R02"]),
        ("y_validation.csv", ["R01", "R02"]),
        ("y_test.csv", ["R01", "R02"]),
    ):
        pd.DataFrame({"row_index": range(len(labels)), "label_code": labels}).to_csv(
            directory / filename, index=False, encoding="utf-8-sig"
        )


def test_naive_bayes_trains_saves_metrics_and_predicts(tmp_path):
    tfidf_dir = tmp_path / "tfidf"
    _prepare_tfidf_artifacts(tfidf_dir)
    output = tmp_path / "naive_bayes"

    result = train_naive_bayes(NaiveBayesConfig(
        tfidf_dir=str(tfidf_dir),
        output_dir=str(output),
        alpha=1.0,
        top_terms=5,
    ))

    assert result.train_rows == 4
    assert result.validation_rows == 2
    assert result.test_rows == 2
    assert result.feature_count > 0
    assert result.trained_class_count == 2
    assert result.validation_accuracy is not None
    assert result.test_accuracy is not None

    for filename in (
        "naive_bayes_bundle.joblib",
        "naive_bayes_summary.json",
        "predictions_validation.csv",
        "predictions_test.csv",
        "confusion_matrix_validation.csv",
        "confusion_matrix_test.csv",
        "top_terms_by_class.csv",
    ):
        assert (output / filename).exists()

    predictions = predict_reports(
        ["duong o_ga", "tu_khoa_hoan_toan_moi"],
        model_path=output / "naive_bayes_bundle.joblib",
        already_processed=True,
    )
    assert len(predictions) == 2
    assert predictions[0]["predicted_label_code"] in {"R01", "R02"}
    assert predictions[0]["has_known_features"] is True
    assert predictions[1]["has_known_features"] is False
    assert predictions[1]["warning"] is not None
    assert 0 <= predictions[0]["confidence"] <= 1


def test_naive_bayes_rejects_matrix_label_row_mismatch(tmp_path):
    tfidf_dir = tmp_path / "tfidf"
    _prepare_tfidf_artifacts(tfidf_dir)
    pd.DataFrame({"row_index": [0], "label_code": ["R01"]}).to_csv(
        tfidf_dir / "y_train.csv", index=False, encoding="utf-8-sig"
    )
    with pytest.raises(ValueError, match="row mismatch"):
        train_naive_bayes(NaiveBayesConfig(
            tfidf_dir=str(tfidf_dir),
            output_dir=str(tmp_path / "out"),
        ))


def test_naive_bayes_rejects_unknown_category_codes(tmp_path):
    tfidf_dir = tmp_path / "tfidf"
    _prepare_tfidf_artifacts(tfidf_dir)
    pd.DataFrame({
        "row_index": range(4),
        "label_code": ["R01", "R01", "R99", "R02"],
    }).to_csv(tfidf_dir / "y_train.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="canonical category codes"):
        train_naive_bayes(NaiveBayesConfig(
            tfidf_dir=str(tfidf_dir),
            output_dir=str(tmp_path / "out"),
        ))


def test_naive_bayes_rejects_negative_alpha(tmp_path):
    with pytest.raises(ValueError, match="alpha"):
        train_naive_bayes(NaiveBayesConfig(alpha=-0.1))

def test_raw_report_prediction_reuses_phase5_preprocessing(tmp_path, monkeypatch):
    tfidf_dir = tmp_path / "tfidf"
    _prepare_tfidf_artifacts(tfidf_dir)
    output = tmp_path / "naive_bayes"
    train_naive_bayes(NaiveBayesConfig(
        tfidf_dir=str(tfidf_dir),
        output_dir=str(output),
    ))

    def fake_preprocessor(text, remove_stopwords_enabled=True):
        assert text == "Raw citizen report"
        assert remove_stopwords_enabled is True
        return SimpleNamespace(processed_text="duong o_ga")

    monkeypatch.setattr(
        naive_bayes_module, "preprocess_vietnamese_text", fake_preprocessor
    )
    predictions = predict_reports(
        ["Raw citizen report"],
        model_path=output / "naive_bayes_bundle.joblib",
    )
    assert predictions[0]["processed_text"] == "duong o_ga"
    assert predictions[0]["has_known_features"] is True

