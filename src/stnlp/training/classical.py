"""TF-IDF + Multinomial Naive Bayes / SVM training."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from stnlp.preprocessing.transformer import VietnameseTextTransformer


def build_classical_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "naive_bayes": Pipeline([
            ("preprocess", VietnameseTextTransformer(remove_stops=True)),
            ("tfidf", TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2),
                min_df=1,
                max_features=100_000,
                sublinear_tf=True,
            )),
            ("classifier", MultinomialNB(alpha=0.5)),
        ]),
        "svm": Pipeline([
            ("preprocess", VietnameseTextTransformer(remove_stops=True)),
            ("tfidf", TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2),
                min_df=1,
                max_features=100_000,
                sublinear_tf=True,
            )),
            ("classifier", SVC(
                kernel="linear",
                C=1.0,
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
    }


def fit_classical(name: str, pipeline: Pipeline, texts, labels) -> Pipeline:
    if name == "naive_bayes":
        weights = compute_sample_weight(class_weight="balanced", y=labels)
        pipeline.fit(texts, labels, classifier__sample_weight=weights)
    else:
        pipeline.fit(texts, labels)
    return pipeline


def save_classical(pipeline: Pipeline, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model.joblib"
    joblib.dump(pipeline, path)
    metadata = {
        "model_name": output_dir.name,
        "model_version": "v1",
        "artifact": "model.joblib",
        "preprocessing": "stnlp.preprocessing.transformer.VietnameseTextTransformer",
        "vectorizer": "TfidfVectorizer",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
