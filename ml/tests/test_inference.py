from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from src.data_processing.categories import CATEGORIES
from src.inference import ModelPredictor, build_inference_bundle


def make_nb_bundle():
    encoder = LabelEncoder().fit([category.code for category in CATEGORIES])
    texts = [
        "duong ho ga hong", "o ga mat duong", "duong nut lun",
        "den duong hong", "mat dien chieu sang",
    ]
    labels = ["R01", "R01", "R01", "R02", "R02"]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    model = MultinomialNB().fit(matrix, encoder.transform(labels))
    return {
        "model": model,
        "vectorizer": vectorizer,
        "label_encoder": encoder,
        "category_names": {c.code: c.name for c in CATEGORIES},
        "preprocessing": {"remove_stopwords": True},
        "model_type": "sklearn.naive_bayes.MultinomialNB",
    }


def make_evaluation(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([
        {"model": "naive_bayes", "split": "validation", "macro_f1": .78, "weighted_f1": .80, "accuracy": .82},
        {"model": "svm", "split": "validation", "macro_f1": .81, "weighted_f1": .82, "accuracy": .83},
        {"model": "cnn", "split": "validation", "macro_f1": .75, "weighted_f1": .79, "accuracy": .80},
        {"model": "lstm", "split": "validation", "macro_f1": .77, "weighted_f1": .78, "accuracy": .79},
        # A very high test score must not influence model selection.
        {"model": "naive_bayes", "split": "test", "macro_f1": .99, "weighted_f1": .99, "accuracy": .99},
        {"model": "svm", "split": "test", "macro_f1": .50, "weighted_f1": .51, "accuracy": .52},
        {"model": "cnn", "split": "test", "macro_f1": .50, "weighted_f1": .51, "accuracy": .52},
        {"model": "lstm", "split": "test", "macro_f1": .50, "weighted_f1": .51, "accuracy": .52},
    ])
    frame.to_csv(path / "model_comparison.csv", index=False)


def test_build_selects_best_validation_macro_f1_and_packages_bundle(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    make_evaluation(evaluation_dir)
    model_dirs = {key: tmp_path / key for key in ("naive_bayes", "svm", "cnn", "lstm")}
    for directory in model_dirs.values():
        directory.mkdir()
    joblib.dump(make_nb_bundle(), model_dirs["svm"] / "svm_bundle.joblib")
    joblib.dump(make_nb_bundle(), model_dirs["naive_bayes"] / "naive_bayes_bundle.joblib")

    result = build_inference_bundle(
        evaluation_dir=evaluation_dir,
        model_dirs=model_dirs,
        output_dir=tmp_path / "inference",
    )

    assert result.selected_model == "svm"
    packed = joblib.load(result.bundle_path)
    assert packed["_inference"]["selection_metric"] == "validation_macro_f1"
    assert packed["_inference"]["selection_value"] == pytest.approx(.81)
    metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    assert metadata["selected_model"] == "svm"


def test_predictor_returns_shared_prediction_contract(tmp_path):
    bundle = make_nb_bundle()
    bundle["_inference"] = {
        "selected_model": "naive_bayes",
        "confidence_threshold": .60,
    }
    path = tmp_path / "best_model_bundle.joblib"
    joblib.dump(bundle, path)
    predictor = ModelPredictor(path)

    result = predictor.predict(["duong ho ga hong"], already_processed=True)[0]

    assert result["model_key"] == "naive_bayes"
    assert result["predicted_label_code"] in {"R01", "R02"}
    assert result["predicted_label_name"]
    assert 0 <= result["confidence"] <= 1
    assert result["confidence_type"] == "model_score"
    assert result["class_probabilities"] is not None
    assert "needs_manual_review" in result


def test_unknown_tfidf_terms_are_flagged_for_review(tmp_path):
    bundle = make_nb_bundle()
    bundle["_inference"] = {
        "selected_model": "naive_bayes",
        "confidence_threshold": .60,
    }
    path = tmp_path / "best_model_bundle.joblib"
    joblib.dump(bundle, path)
    result = ModelPredictor(path).predict(["unseenword xyzabc"], already_processed=True)[0]

    assert result["has_known_features"] is False
    assert result["needs_manual_review"] is True
    assert result["warning_reasons"]


def test_raw_text_uses_shared_preprocessing(monkeypatch, tmp_path):
    bundle = make_nb_bundle()
    bundle["_inference"] = {"selected_model": "naive_bayes", "confidence_threshold": .60}
    path = tmp_path / "best_model_bundle.joblib"
    joblib.dump(bundle, path)

    import src.inference.predictor as predictor_module
    monkeypatch.setattr(
        predictor_module,
        "preprocess_vietnamese_text",
        lambda value, remove_stopwords_enabled=True: SimpleNamespace(processed_text="duong ho ga hong"),
    )
    result = ModelPredictor(path).predict(["Đường có ổ gà"])[0]
    assert result["processed_text"] == "duong ho ga hong"


def test_empty_report_is_rejected(tmp_path):
    bundle = make_nb_bundle()
    bundle["_inference"] = {"selected_model": "naive_bayes", "confidence_threshold": .60}
    path = tmp_path / "best_model_bundle.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="non-empty"):
        ModelPredictor(path).predict(["   "], already_processed=True)


def test_selection_requires_validation_metrics_for_all_four_models(tmp_path):
    path = tmp_path / "evaluation"
    path.mkdir()
    pd.DataFrame([
        {"model": "naive_bayes", "split": "validation", "macro_f1": .8, "weighted_f1": .8, "accuracy": .8}
    ]).to_csv(path / "model_comparison.csv", index=False)
    with pytest.raises(ValueError, match="exactly one validation score"):
        build_inference_bundle(evaluation_dir=path, model_dirs={
            key: tmp_path / key for key in ("naive_bayes", "svm", "cnn", "lstm")
        }, output_dir=tmp_path / "inference")
