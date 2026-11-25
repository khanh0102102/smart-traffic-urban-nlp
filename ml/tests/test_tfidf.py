"""Tests for leakage-safe TF-IDF feature generation."""
from pathlib import Path
import sys

import pandas as pd
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tfidf import TfidfConfig, build_tfidf, load_tfidf_vectorizer


def _write_split(path: Path, rows: list[tuple[str, str]]) -> None:
    pd.DataFrame(
        [{"processed_text": text, "label_code": label, "label_name": label}
         for text, label in rows]
    ).to_csv(path, index=False, encoding="utf-8-sig")


def test_tfidf_fits_train_only_and_transforms_holdout_splits(tmp_path):
    dataset = tmp_path / "processed"
    dataset.mkdir()
    _write_split(dataset / "train.csv", [
        ("duong hu hong", "R01"),
        ("den tin hieu hong", "R08"),
        ("duong ngap nuoc", "R03"),
    ])
    _write_split(dataset / "validation.csv", [
        ("tu khoa validation rieng", "R09"),
    ])
    _write_split(dataset / "test.csv", [
        ("tu khoa test rieng", "R10"),
    ])
    output = tmp_path / "tfidf"

    result = build_tfidf(TfidfConfig(
        dataset_dir=str(dataset),
        output_dir=str(output),
        ngram_min=1,
        ngram_max=2,
    ))

    assert result.train_rows == 3
    assert result.validation_rows == 1
    assert result.test_rows == 1
    assert result.feature_count > 0

    vectorizer = load_tfidf_vectorizer(output / "tfidf_vectorizer.joblib")
    vocabulary = set(vectorizer.get_feature_names_out())
    assert "validation" not in vocabulary
    assert "test" not in vocabulary
    assert "duong" in vocabulary

    x_train = load_npz(output / "X_train.npz")
    x_validation = load_npz(output / "X_validation.npz")
    x_test = load_npz(output / "X_test.npz")
    assert x_train.shape == (3, result.feature_count)
    assert x_validation.shape == (1, result.feature_count)
    assert x_test.shape == (1, result.feature_count)
    assert x_validation.nnz == 0
    assert x_test.nnz == 0

    labels = pd.read_csv(output / "y_train.csv", encoding="utf-8-sig")
    assert labels["row_index"].tolist() == [0, 1, 2]
    assert labels["label_code"].tolist() == ["R01", "R08", "R03"]
    assert (output / "vocabulary.csv").exists()
    assert (output / "tfidf_summary.json").exists()


def test_tfidf_rejects_empty_training_text(tmp_path):
    dataset = tmp_path / "processed"
    dataset.mkdir()
    _write_split(dataset / "train.csv", [("", "R01")])
    _write_split(dataset / "validation.csv", [])
    _write_split(dataset / "test.csv", [])

    try:
        build_tfidf(TfidfConfig(dataset_dir=str(dataset), output_dir=str(tmp_path / "out")))
    except ValueError as exc:
        assert "All values" in str(exc)
    else:
        raise AssertionError("Expected an actionable error for an empty training corpus.")


def test_tfidf_requires_phase5_processed_text_column(tmp_path):
    dataset = tmp_path / "processed"
    dataset.mkdir()
    pd.DataFrame([{"cleaned_text": "duong hong", "label_code": "R01"}]).to_csv(
        dataset / "train.csv", index=False
    )
    _write_split(dataset / "validation.csv", [])
    _write_split(dataset / "test.csv", [])

    try:
        build_tfidf(TfidfConfig(dataset_dir=str(dataset), output_dir=str(tmp_path / "out")))
    except ValueError as exc:
        assert "processed_text" in str(exc)
    else:
        raise AssertionError("Expected an error when Phase 5 processed_text is missing.")
