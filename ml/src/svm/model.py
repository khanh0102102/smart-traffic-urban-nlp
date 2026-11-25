"""Train, calibrate, evaluate and serve a linear SVM on TF-IDF features.

LinearSVC is suitable for high-dimensional sparse TF-IDF matrices. Probability
calibration uses cross-validated sigmoid calibration on the training split only.
When training classes are too small for calibration, the module keeps a valid
uncalibrated SVM and explicitly marks probability/confidence as unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.data_processing.categories import CATEGORIES
from src.embedding import LabelEncoderBundle
from src.nlp_preprocessing import preprocess_vietnamese_text


@dataclass
class SVMConfig:
    tfidf_dir: str = "ml/artifacts/tfidf"
    output_dir: str = "ml/artifacts/svm"
    C: float = 1.0
    class_weight: str | None = "balanced"
    max_iter: int = 10000
    random_state: int = 42
    calibrate_probabilities: bool = True
    calibration_method: str = "sigmoid"
    calibration_cv: int = 3
    top_features: int = 20


@dataclass
class SVMResult:
    train_rows: int
    validation_rows: int
    test_rows: int
    feature_count: int
    trained_class_count: int
    probability_calibrated: bool
    calibration_method: str | None
    calibration_cv: int | None
    calibration_reason: str | None
    validation_accuracy: float | None
    test_accuracy: float | None
    model_path: str
    output_dir: str


def _validate_config(config: SVMConfig) -> None:
    if config.C <= 0:
        raise ValueError("C must be greater than zero.")
    if config.class_weight not in {None, "balanced"}:
        raise ValueError("class_weight must be None or 'balanced'.")
    if config.max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if config.calibration_method not in {"sigmoid", "isotonic"}:
        raise ValueError("calibration_method must be 'sigmoid' or 'isotonic'.")
    if config.calibration_cv < 2:
        raise ValueError("calibration_cv must be at least 2.")
    if config.top_features < 1:
        raise ValueError("top_features must be at least 1.")


def _load_matrix(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"TF-IDF matrix not found: {path}. Run ml/scripts/build_tfidf.py first."
        )
    matrix = load_npz(path).tocsr()
    if matrix.data.size and not np.isfinite(matrix.data).all():
        raise ValueError(f"{path.name} contains non-finite feature values.")
    return matrix


def _load_labels(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"TF-IDF labels not found: {path}. Run ml/scripts/build_tfidf.py first."
        )
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    if "label_code" not in frame.columns:
        raise ValueError(f"{path.name} must contain a label_code column.")
    if "row_index" in frame.columns:
        indices = pd.to_numeric(frame["row_index"], errors="coerce")
        expected = np.arange(len(frame), dtype=np.int64)
        if indices.isna().any() or not np.array_equal(
            indices.to_numpy(dtype=np.int64), expected
        ):
            raise ValueError(
                f"{path.name} row_index must be sequential and aligned with the feature matrix."
            )
    labels = frame["label_code"].astype(str).str.strip().str.upper().tolist()
    if any(not label for label in labels):
        raise ValueError(f"{path.name} contains an empty label_code.")
    return labels


def _validate_split(
    name: str,
    matrix: Any,
    labels: list[str],
    feature_count: int,
) -> None:
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


def _make_estimator(config: SVMConfig) -> LinearSVC:
    return LinearSVC(
        C=config.C,
        class_weight=config.class_weight,
        max_iter=config.max_iter,
        random_state=config.random_state,
    )


def _fit_svm(
    x_train: Any,
    y_train: np.ndarray,
    config: SVMConfig,
) -> tuple[Any, bool, int | None, str | None]:
    """Fit SVM and apply sigmoid/isotonic calibration when sample counts allow."""
    unique_classes, counts = np.unique(y_train, return_counts=True)
    if len(unique_classes) < 2:
        raise ValueError(
            "SVM training requires at least two different category labels in y_train."
        )

    base_estimator = _make_estimator(config)
    if not config.calibrate_probabilities:
        base_estimator.fit(x_train, y_train)
        return base_estimator, False, None, "Probability calibration disabled by configuration."

    min_class_count = int(counts.min())
    if min_class_count < 2:
        base_estimator.fit(x_train, y_train)
        reason = (
            "Probability calibration skipped: every training class needs at least "
            "two examples for stratified cross-validation. The SVM decision scores "
            "are retained, but they are not calibrated probabilities."
        )
        return base_estimator, False, None, reason

    if config.calibration_method == "isotonic" and len(y_train) < 3000:
        raise ValueError(
            "Isotonic calibration is not recommended with fewer than 3000 total "
            "training rows (the calibration fold would contain fewer than roughly "
            "1000 rows with three-fold CV). Use sigmoid calibration or add data."
        )

    cv_folds = min(config.calibration_cv, min_class_count)
    calibrated = CalibratedClassifierCV(
        estimator=base_estimator,
        method=config.calibration_method,
        cv=cv_folds,
    )
    calibrated.fit(x_train, y_train)
    return calibrated, True, cv_folds, None


def _model_classes(model: Any) -> np.ndarray:
    return np.asarray(model.classes_, dtype=np.int64)


def _align_estimator_decision_scores(
    estimator: Any,
    matrix: Any,
    target_classes: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(estimator.decision_function(matrix))
    estimator_classes = np.asarray(estimator.classes_, dtype=np.int64)
    if scores.ndim == 1:
        # Binary decision_function is positive for estimator_classes_[1].
        scores = np.column_stack((-scores, scores))
    if scores.shape[1] != len(estimator_classes):
        raise ValueError(
            "SVM decision score dimensions do not match the fitted estimator classes."
        )
    aligned = np.zeros((matrix.shape[0], len(target_classes)), dtype=float)
    target_positions = {int(value): index for index, value in enumerate(target_classes)}
    for source_column, class_id in enumerate(estimator_classes):
        if int(class_id) in target_positions:
            aligned[:, target_positions[int(class_id)]] = scores[:, source_column]
    return aligned


def _decision_scores(model: Any, matrix: Any) -> np.ndarray:
    classes = _model_classes(model)
    if hasattr(model, "decision_function"):
        return _align_estimator_decision_scores(model, matrix, classes)

    # CalibratedClassifierCV exposes predict_proba but not decision_function.
    # Average the underlying LinearSVC margins for a diagnostic decision score.
    calibrated_estimators = getattr(model, "calibrated_classifiers_", None)
    if calibrated_estimators:
        fold_scores = [
            _align_estimator_decision_scores(item.estimator, matrix, classes)
            for item in calibrated_estimators
        ]
        return np.mean(fold_scores, axis=0)
    raise TypeError("The fitted SVM estimator does not expose decision scores.")


def _confidence_values(
    model: Any,
    matrix: Any,
    predicted_ids: np.ndarray,
    probability_calibrated: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return calibrated confidence when available and per-row decision margins."""
    scores = _decision_scores(model, matrix)
    classes = _model_classes(model)
    class_columns = {int(class_id): index for index, class_id in enumerate(classes)}
    margins = np.asarray([
        scores[row_index, class_columns[int(class_id)]]
        for row_index, class_id in enumerate(predicted_ids)
    ], dtype=float)

    if probability_calibrated:
        probabilities = model.predict_proba(matrix)
        confidence = probabilities.max(axis=1).astype(float)
    else:
        # Do not present margins as probabilities or calibrated confidence.
        confidence = np.full(matrix.shape[0], np.nan, dtype=float)
    return confidence, margins


def _evaluate(
    model: Any,
    matrix: Any,
    actual_ids: np.ndarray,
    label_encoder: LabelEncoder,
    split_name: str,
    output: Path,
    probability_calibrated: bool,
) -> dict[str, Any] | None:
    class_codes = label_encoder.classes_.tolist()
    category_names = {category.code: category.name for category in CATEGORIES}
    if matrix.shape[0] == 0:
        pd.DataFrame(columns=[
            "row_index", "actual_label_code", "predicted_label_code",
            "actual_label_name", "predicted_label_name", "confidence",
            "decision_margin", "probability_calibrated", "correct",
        ]).to_csv(
            output / f"predictions_{split_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        empty = pd.DataFrame(
            np.zeros((len(class_codes), len(class_codes)), dtype=np.int64),
            index=class_codes,
            columns=class_codes,
        )
        empty.index.name = "actual_label_code"
        empty.to_csv(
            output / f"confusion_matrix_{split_name}.csv",
            encoding="utf-8-sig",
        )
        return None

    predicted_ids = model.predict(matrix).astype(np.int64)
    confidence, margins = _confidence_values(
        model, matrix, predicted_ids, probability_calibrated
    )
    predicted_codes = label_encoder.inverse_transform(predicted_ids)
    actual_codes = label_encoder.inverse_transform(actual_ids)
    predictions = pd.DataFrame({
        "row_index": np.arange(len(actual_ids), dtype=np.int64),
        "actual_label_code": actual_codes,
        "predicted_label_code": predicted_codes,
        "actual_label_name": [category_names[code] for code in actual_codes],
        "predicted_label_name": [category_names[code] for code in predicted_codes],
        "confidence": confidence,
        "decision_margin": margins,
        "probability_calibrated": probability_calibrated,
        "correct": actual_ids == predicted_ids,
    })
    predictions.to_csv(
        output / f"predictions_{split_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_class_ids = np.arange(len(label_encoder.classes_), dtype=np.int64)
    report = classification_report(
        actual_ids,
        predicted_ids,
        labels=all_class_ids,
        target_names=class_codes,
        output_dict=True,
        zero_division=0,
    )
    matrix_values = confusion_matrix(
        actual_ids, predicted_ids, labels=all_class_ids
    )
    confusion = pd.DataFrame(
        matrix_values,
        index=class_codes,
        columns=class_codes,
    )
    confusion.index.name = "actual_label_code"
    confusion.to_csv(
        output / f"confusion_matrix_{split_name}.csv",
        encoding="utf-8-sig",
    )

    result = {
        "rows": int(matrix.shape[0]),
        "accuracy": float(accuracy_score(actual_ids, predicted_ids)),
        "classification_report": report,
        "confusion_matrix_file": f"confusion_matrix_{split_name}.csv",
        "predictions_file": f"predictions_{split_name}.csv",
        "probability_calibrated": probability_calibrated,
    }
    if probability_calibrated:
        from sklearn.metrics import log_loss
        model_classes = _model_classes(model)
        result["log_loss"] = float(
            log_loss(actual_ids, model.predict_proba(matrix), labels=model_classes)
        ) if set(np.unique(actual_ids)).issubset(set(model_classes)) else None
    return result


def _linear_coefficients(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """Average linear coefficients across calibrated CV estimators if needed."""
    if hasattr(model, "calibrated_classifiers_"):
        estimators = [
            calibrated.estimator
            for calibrated in model.calibrated_classifiers_
        ]
    else:
        estimators = [model]

    class_ids = _model_classes(model)
    all_coefficients = []
    for estimator in estimators:
        coefficients = np.asarray(estimator.coef_, dtype=float)
        estimator_classes = np.asarray(estimator.classes_, dtype=np.int64)
        if len(estimator_classes) == 2 and coefficients.shape[0] == 1:
            coefficients = np.vstack((-coefficients[0], coefficients[0]))
        if coefficients.shape[0] != len(estimator_classes):
            raise ValueError("Cannot align SVM coefficients with estimator classes.")
        aligned = np.zeros((len(class_ids), coefficients.shape[1]), dtype=float)
        class_positions = {int(value): index for index, value in enumerate(estimator_classes)}
        for target_row, class_id in enumerate(class_ids):
            if int(class_id) in class_positions:
                aligned[target_row] = coefficients[class_positions[int(class_id)]]
        all_coefficients.append(aligned)
    return class_ids, np.mean(all_coefficients, axis=0)


def _save_top_features(
    model: Any,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
    output: Path,
    limit: int,
) -> None:
    class_ids, coefficients = _linear_coefficients(model)
    terms = vectorizer.get_feature_names_out()
    rows: list[dict[str, object]] = []
    category_names = {category.code: category.name for category in CATEGORIES}
    for row_index, class_id in enumerate(class_ids):
        code = str(label_encoder.inverse_transform([int(class_id)])[0])
        order = np.argsort(coefficients[row_index])[::-1][:limit]
        for rank, feature_index in enumerate(order, start=1):
            rows.append({
                "label_code": code,
                "label_name": category_names[code],
                "rank": rank,
                "term": str(terms[feature_index]),
                "coefficient": float(coefficients[row_index, feature_index]),
            })
    pd.DataFrame(rows).to_csv(
        output / "top_features_by_class.csv",
        index=False,
        encoding="utf-8-sig",
    )


def train_svm(config: SVMConfig | None = None) -> SVMResult:
    """Fit LinearSVC on Phase 6 train data, calibrate when possible, evaluate holdouts."""
    config = config or SVMConfig()
    _validate_config(config)

    tfidf_dir = Path(config.tfidf_dir)
    vectorizer_path = tfidf_dir / "tfidf_vectorizer.joblib"
    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Saved TF-IDF vectorizer not found: {vectorizer_path}. "
            "Run Phase 6 before training SVM."
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

    model, probability_calibrated, calibration_cv, calibration_reason = _fit_svm(
        x_train, y_train, config
    )

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation_metrics = _evaluate(
        model, x_validation, y_validation, label_encoder,
        "validation", output, probability_calibrated,
    )
    test_metrics = _evaluate(
        model, x_test, y_test, label_encoder,
        "test", output, probability_calibrated,
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
        "model_type": "sklearn.svm.LinearSVC",
        "probability_calibrated": probability_calibrated,
        "calibration_method": config.calibration_method if probability_calibrated else None,
        "calibration_cv": calibration_cv,
        "calibration_reason": calibration_reason,
    }
    model_path = output / "svm_bundle.joblib"
    joblib.dump(bundle, model_path, compress=3)
    _save_top_features(model, vectorizer, label_encoder, output, config.top_features)

    summary = {
        "model": "LinearSVC",
        "parameters": {
            "C": config.C,
            "class_weight": config.class_weight,
            "max_iter": config.max_iter,
            "random_state": config.random_state,
        },
        "probability_calibration": {
            "requested": config.calibrate_probabilities,
            "calibrated": probability_calibrated,
            "method": config.calibration_method if probability_calibrated else None,
            "cv_folds": calibration_cv,
            "reason": calibration_reason,
        },
        "training_source": "X_train.npz and y_train.csv only",
        "tfidf_vectorizer_source": str(vectorizer_path),
        "feature_count": int(feature_count),
        "trained_class_codes": [
            str(label_encoder.inverse_transform([int(class_id)])[0])
            for class_id in _model_classes(model)
        ],
        "class_mapping": {
            str(code): int(index)
            for index, code in enumerate(label_encoder.classes_)
        },
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "artifacts": [
            "svm_bundle.joblib",
            "predictions_validation.csv",
            "predictions_test.csv",
            "confusion_matrix_validation.csv",
            "confusion_matrix_test.csv",
            "top_features_by_class.csv",
            "svm_summary.json",
        ],
        "notes": [
            "LinearSVC is fitted only on the Phase 6 training matrix.",
            "Validation and test data are evaluated but never used for fitting or probability calibration.",
            "When every training class has at least two examples, sigmoid/isotonic calibration is cross-validated on train only.",
            "If a class has fewer than two training examples, calibration is skipped and decision margins are reported instead of probabilities.",
            "Classes absent from train cannot be predicted by the fitted model.",
            "The included sample dataset is a workflow demonstration, not evidence of production performance.",
        ],
    }
    (output / "svm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return SVMResult(
        train_rows=int(x_train.shape[0]),
        validation_rows=int(x_validation.shape[0]),
        test_rows=int(x_test.shape[0]),
        feature_count=int(feature_count),
        trained_class_count=int(len(_model_classes(model))),
        probability_calibrated=probability_calibrated,
        calibration_method=config.calibration_method if probability_calibrated else None,
        calibration_cv=calibration_cv,
        calibration_reason=calibration_reason,
        validation_accuracy=None if validation_metrics is None else validation_metrics["accuracy"],
        test_accuracy=None if test_metrics is None else test_metrics["accuracy"],
        model_path=str(model_path),
        output_dir=str(output),
    )


def load_svm_bundle(
    model_path: str | Path = "ml/artifacts/svm/svm_bundle.joblib",
) -> dict[str, Any]:
    """Load the saved SVM, TF-IDF vectorizer and label encoder."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SVM model not found: {path}. Run ml/scripts/train_svm.py first."
        )
    bundle = joblib.load(path)
    required = {
        "model", "vectorizer", "label_encoder", "category_names",
        "preprocessing", "probability_calibrated",
    }
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Invalid SVM model bundle: {path}")
    if not isinstance(bundle["vectorizer"], TfidfVectorizer):
        raise TypeError("The saved vectorizer is not a TfidfVectorizer instance.")
    if not isinstance(bundle["label_encoder"], LabelEncoder):
        raise TypeError("The saved label encoder is not a LabelEncoder instance.")
    model = bundle["model"]
    is_linear_svc = isinstance(model, LinearSVC)
    is_calibrated = isinstance(model, CalibratedClassifierCV)
    if not hasattr(model, "predict") or not (is_linear_svc or is_calibrated):
        raise TypeError("The saved SVM estimator has an unexpected type.")
    if bundle["probability_calibrated"] and not hasattr(bundle["model"], "predict_proba"):
        raise ValueError("Bundle is marked calibrated but the estimator has no predict_proba.")
    return bundle


def predict_svm_reports(
    texts: Iterable[object],
    *,
    model_path: str | Path = "ml/artifacts/svm/svm_bundle.joblib",
    already_processed: bool = False,
) -> list[dict[str, Any]]:
    """Predict raw citizen reports or preprocessed Phase 5 texts with saved SVM."""
    bundle = load_svm_bundle(model_path)
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
    if not processed:
        return []

    vectorizer: TfidfVectorizer = bundle["vectorizer"]
    model = bundle["model"]
    label_encoder: LabelEncoder = bundle["label_encoder"]
    category_names: dict[str, str] = bundle["category_names"]
    calibrated = bool(bundle["probability_calibrated"])

    matrix = vectorizer.transform(processed).tocsr()
    predicted_ids = model.predict(matrix).astype(np.int64)
    confidence, margins = _confidence_values(
        model, matrix, predicted_ids, calibrated
    )
    probabilities = model.predict_proba(matrix) if calibrated else None
    classes = _model_classes(model)
    results: list[dict[str, Any]] = []
    for row_index, class_id in enumerate(predicted_ids):
        code = str(label_encoder.inverse_transform([int(class_id)])[0])
        has_known_features = bool(matrix[row_index].nnz)
        class_probabilities = None
        if probabilities is not None:
            class_probabilities = {
                str(label_encoder.inverse_transform([int(model_class)])[0]): float(
                    probabilities[row_index, column]
                )
                for column, model_class in enumerate(classes)
            }
        results.append({
            "raw_content": values[row_index],
            "processed_text": processed[row_index],
            "predicted_label_code": code,
            "predicted_label_name": category_names[code],
            "probability_calibrated": calibrated,
            "confidence": (
                float(confidence[row_index]) if calibrated else None
            ),
            "decision_margin": float(margins[row_index]),
            "class_probabilities": class_probabilities,
            "has_known_features": has_known_features,
            "warning": (
                "No input token exists in the trained TF-IDF vocabulary; "
                "prediction is based on model bias/intercepts."
                if not has_known_features else bundle.get("calibration_reason")
            ),
        })
    return results
