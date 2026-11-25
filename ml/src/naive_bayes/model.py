"""Train, evaluate, persist and serve the TF-IDF Multinomial Naive Bayes baseline."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from src.data_processing.categories import CATEGORIES
from src.embedding import LabelEncoderBundle
from src.nlp_preprocessing import preprocess_vietnamese_text


@dataclass
class NaiveBayesConfig:
    tfidf_dir: str = "ml/artifacts/tfidf"
    output_dir: str = "ml/artifacts/naive_bayes"
    alpha: float = 1.0
    fit_prior: bool = True
    top_terms: int = 20


@dataclass
class NaiveBayesResult:
    train_rows: int
    validation_rows: int
    test_rows: int
    feature_count: int
    trained_class_count: int
    validation_accuracy: float | None
    test_accuracy: float | None
    model_path: str
    output_dir: str


def _validate_config(config: NaiveBayesConfig) -> None:
    if config.alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if config.top_terms < 1:
        raise ValueError("top_terms must be at least 1.")


def _load_matrix(path: Path, *, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"TF-IDF matrix not found: {path}. Run ml/scripts/build_tfidf.py first."
            )
        raise FileNotFoundError(f"TF-IDF matrix not found: {path}")
    return load_npz(path).tocsr()


def _load_labels(path: Path, *, required: bool = True) -> list[str]:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"TF-IDF labels not found: {path}. Run ml/scripts/build_tfidf.py first."
            )
        raise FileNotFoundError(f"TF-IDF labels not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    if "label_code" not in frame.columns:
        raise ValueError(f"{path.name} must contain a label_code column.")
    if "row_index" in frame.columns:
        indices = pd.to_numeric(frame["row_index"], errors="coerce")
        expected = np.arange(len(frame), dtype=np.int64)
        if indices.isna().any() or not np.array_equal(indices.to_numpy(dtype=np.int64), expected):
            raise ValueError(f"{path.name} row_index must be sequential and aligned with the feature matrix.")
    labels = frame["label_code"].astype(str).str.strip().str.upper().tolist()
    if any(not label for label in labels):
        raise ValueError(f"{path.name} contains an empty label_code.")
    return labels


def _validate_split(name: str, matrix: Any, labels: list[str], feature_count: int) -> None:
    if matrix.shape[0] != len(labels):
        raise ValueError(
            f"{name} row mismatch: feature matrix has {matrix.shape[0]} rows "
            f"but label file has {len(labels)} labels."
        )
    if matrix.shape[1] != feature_count:
        raise ValueError(
            f"{name} feature mismatch: matrix has {matrix.shape[1]} columns; "
            f"expected {feature_count} columns from the saved TF-IDF vectorizer."
        )
    if matrix.data.size and np.min(matrix.data) < 0:
        raise ValueError("MultinomialNB requires non-negative feature values.")


def _evaluate(
    model: MultinomialNB,
    matrix: Any,
    actual_ids: np.ndarray,
    label_encoder: LabelEncoder,
    split_name: str,
    output: Path,
) -> dict[str, Any] | None:
    category_codes = label_encoder.classes_.tolist()
    if matrix.shape[0] == 0:
        empty_predictions = pd.DataFrame(columns=[
            "row_index", "actual_label_code", "predicted_label_code",
            "actual_label_name", "predicted_label_name", "confidence", "correct",
        ])
        empty_predictions.to_csv(
            output / f"predictions_{split_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        empty_confusion = pd.DataFrame(
            np.zeros((len(category_codes), len(category_codes)), dtype=np.int64),
            index=category_codes,
            columns=category_codes,
        )
        empty_confusion.index.name = "actual_label_code"
        empty_confusion.to_csv(
            output / f"confusion_matrix_{split_name}.csv",
            encoding="utf-8-sig",
        )
        return None

    predicted_ids = model.predict(matrix).astype(np.int64)
    probabilities = model.predict_proba(matrix)
    confidence = probabilities.max(axis=1)
    predicted_codes = label_encoder.inverse_transform(predicted_ids)
    actual_codes = label_encoder.inverse_transform(actual_ids)

    category_names = {category.code: category.name for category in CATEGORIES}
    prediction_frame = pd.DataFrame({
        "row_index": np.arange(len(actual_ids), dtype=np.int64),
        "actual_label_code": actual_codes,
        "predicted_label_code": predicted_codes,
        "actual_label_name": [category_names[code] for code in actual_codes],
        "predicted_label_name": [category_names[code] for code in predicted_codes],
        "confidence": confidence,
        "correct": actual_ids == predicted_ids,
    })
    prediction_frame.to_csv(
        output / f"predictions_{split_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_class_ids = np.arange(len(label_encoder.classes_), dtype=np.int64)
    report = classification_report(
        actual_ids,
        predicted_ids,
        labels=all_class_ids,
        target_names=label_encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )
    matrix_values = confusion_matrix(
        actual_ids, predicted_ids, labels=all_class_ids
    )
    confusion = pd.DataFrame(
        matrix_values,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )
    confusion.index.name = "actual_label_code"
    confusion.to_csv(output / f"confusion_matrix_{split_name}.csv", encoding="utf-8-sig")

    return {
        "rows": int(matrix.shape[0]),
        "accuracy": float(accuracy_score(actual_ids, predicted_ids)),
        "classification_report": report,
        "confusion_matrix_file": f"confusion_matrix_{split_name}.csv",
        "predictions_file": f"predictions_{split_name}.csv",
    }


def _save_top_terms(
    model: MultinomialNB,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
    output: Path,
    limit: int,
) -> None:
    terms = vectorizer.get_feature_names_out()
    rows: list[dict[str, object]] = []
    for class_id, log_probabilities in zip(model.classes_, model.feature_log_prob_):
        code = str(label_encoder.inverse_transform([int(class_id)])[0])
        category_name = next(
            category.name for category in CATEGORIES if category.code == code
        )
        top_indices = np.argsort(log_probabilities)[::-1][:limit]
        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append({
                "label_code": code,
                "label_name": category_name,
                "rank": rank,
                "term": str(terms[feature_index]),
                "log_probability": float(log_probabilities[feature_index]),
            })
    pd.DataFrame(rows).to_csv(
        output / "top_terms_by_class.csv",
        index=False,
        encoding="utf-8-sig",
    )


def train_naive_bayes(
    config: NaiveBayesConfig | None = None,
) -> NaiveBayesResult:
    """Fit MultinomialNB on Phase 6 train matrix and evaluate held-out matrices."""
    config = config or NaiveBayesConfig()
    _validate_config(config)
    tfidf_dir = Path(config.tfidf_dir)
    vectorizer_path = tfidf_dir / "tfidf_vectorizer.joblib"
    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Saved TF-IDF vectorizer not found: {vectorizer_path}. "
            "Run Phase 6 before training Naive Bayes."
        )
    vectorizer = joblib.load(vectorizer_path)
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError("tfidf_vectorizer.joblib is not a scikit-learn TfidfVectorizer.")

    x_train = _load_matrix(tfidf_dir / "X_train.npz")
    x_validation = _load_matrix(tfidf_dir / "X_validation.npz")
    x_test = _load_matrix(tfidf_dir / "X_test.npz")
    train_codes = _load_labels(tfidf_dir / "y_train.csv")
    validation_codes = _load_labels(tfidf_dir / "y_validation.csv")
    test_codes = _load_labels(tfidf_dir / "y_test.csv")

    feature_count = len(vectorizer.get_feature_names_out())
    if x_train.shape[0] == 0:
        raise ValueError("X_train.npz is empty. At least one training row is required.")
    for name, matrix, labels in (
        ("train", x_train, train_codes),
        ("validation", x_validation, validation_codes),
        ("test", x_test, test_codes),
    ):
        _validate_split(name, matrix, labels, feature_count)

    label_encoder = LabelEncoderBundle().fit_taxonomy().encoder
    try:
        y_train = label_encoder.transform(train_codes).astype(np.int64)
        y_validation = label_encoder.transform(validation_codes).astype(np.int64)
        y_test = label_encoder.transform(test_codes).astype(np.int64)
    except ValueError as exc:
        raise ValueError(
            "TF-IDF label files must use canonical category codes R01-R11."
        ) from exc

    model = MultinomialNB(alpha=config.alpha, fit_prior=config.fit_prior)
    model.fit(x_train, y_train)

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation_metrics = _evaluate(
        model, x_validation, y_validation, label_encoder, "validation", output
    )
    test_metrics = _evaluate(
        model, x_test, y_test, label_encoder, "test", output
    )

    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
        "category_names": {category.code: category.name for category in CATEGORIES},
        "preprocessing": {
            "pipeline": "src.nlp_preprocessing.preprocess_vietnamese_text",
            "remove_stopwords": True,
            "text_column": "processed_text",
        },
        "model_type": "sklearn.naive_bayes.MultinomialNB",
    }
    model_path = output / "naive_bayes_bundle.joblib"
    joblib.dump(bundle, model_path, compress=3)
    _save_top_terms(model, vectorizer, label_encoder, output, config.top_terms)

    summary = {
        "model": "MultinomialNB",
        "parameters": {
            "alpha": config.alpha,
            "fit_prior": config.fit_prior,
        },
        "training_source": "X_train.npz and y_train.csv only",
        "tfidf_vectorizer_source": str(vectorizer_path),
        "feature_count": int(feature_count),
        "class_mapping": {
            str(code): int(index)
            for index, code in enumerate(label_encoder.classes_)
        },
        "trained_class_codes": [
            str(label_encoder.inverse_transform([int(class_id)])[0])
            for class_id in model.classes_
        ],
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "artifacts": [
            "naive_bayes_bundle.joblib",
            "predictions_validation.csv",
            "predictions_test.csv",
            "confusion_matrix_validation.csv",
            "confusion_matrix_test.csv",
            "top_terms_by_class.csv",
            "naive_bayes_summary.json",
        ],
        "notes": [
            "The model is fitted only on the Phase 6 training matrix.",
            "Validation and test data are evaluated but never used to fit the model.",
            "The saved bundle contains the exact fitted TF-IDF vectorizer and canonical label encoder used for training.",
            "Predictions for classes absent from train are impossible for this fitted model; add labeled training examples for those categories.",
            "Small sample datasets are demonstrations only; reported scores are not representative production performance.",
        ],
    }
    (output / "naive_bayes_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return NaiveBayesResult(
        train_rows=int(x_train.shape[0]),
        validation_rows=int(x_validation.shape[0]),
        test_rows=int(x_test.shape[0]),
        feature_count=int(feature_count),
        trained_class_count=int(len(model.classes_)),
        validation_accuracy=(
            None if validation_metrics is None else validation_metrics["accuracy"]
        ),
        test_accuracy=None if test_metrics is None else test_metrics["accuracy"],
        model_path=str(model_path),
        output_dir=str(output),
    )


def load_naive_bayes_bundle(
    model_path: str | Path = "ml/artifacts/naive_bayes/naive_bayes_bundle.joblib",
) -> dict[str, Any]:
    """Load a saved Naive Bayes + vectorizer + label encoder bundle."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Naive Bayes model not found: {path}. Run ml/scripts/train_naive_bayes.py first."
        )
    bundle = joblib.load(path)
    required = {"model", "vectorizer", "label_encoder", "category_names", "preprocessing"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Invalid Naive Bayes model bundle: {path}")
    if not isinstance(bundle["model"], MultinomialNB):
        raise TypeError("The saved model is not a MultinomialNB instance.")
    if not isinstance(bundle["vectorizer"], TfidfVectorizer):
        raise TypeError("The saved vectorizer is not a TfidfVectorizer instance.")
    if not isinstance(bundle["label_encoder"], LabelEncoder):
        raise TypeError("The saved label encoder is not a LabelEncoder instance.")
    return bundle


def predict_reports(
    texts: Iterable[object],
    *,
    model_path: str | Path = "ml/artifacts/naive_bayes/naive_bayes_bundle.joblib",
    already_processed: bool = False,
) -> list[dict[str, Any]]:
    """Predict raw citizen reports or Phase 5 processed_text strings."""
    bundle = load_naive_bayes_bundle(model_path)
    values = ["" if value is None else str(value) for value in texts]
    if already_processed:
        processed = values
    else:
        processed = [
            preprocess_vietnamese_text(
                value,
                remove_stopwords_enabled=bool(
                    bundle["preprocessing"].get("remove_stopwords", True)
                ),
            ).processed_text
            for value in values
        ]

    vectorizer: TfidfVectorizer = bundle["vectorizer"]
    model: MultinomialNB = bundle["model"]
    label_encoder: LabelEncoder = bundle["label_encoder"]
    category_names: dict[str, str] = bundle["category_names"]

    if not processed:
        return []
    matrix = vectorizer.transform(processed).tocsr()
    predicted_ids = model.predict(matrix).astype(np.int64)
    probabilities = model.predict_proba(matrix)
    results: list[dict[str, Any]] = []
    for index, class_id in enumerate(predicted_ids):
        code = str(label_encoder.inverse_transform([int(class_id)])[0])
        has_known_features = bool(matrix[index].nnz)
        results.append({
            "raw_content": values[index],
            "processed_text": processed[index],
            "predicted_label_code": code,
            "predicted_label_name": category_names[code],
            "confidence": float(probabilities[index].max()),
            "has_known_features": has_known_features,
            "warning": (
                None if has_known_features
                else "No input token exists in the trained TF-IDF vocabulary; prediction relies on class priors."
            ),
        })
    return results
