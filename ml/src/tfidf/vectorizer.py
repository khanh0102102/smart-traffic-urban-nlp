"""Build and persist TF-IDF matrices from Phase 5 processed datasets.

Only train.csv is used to fit the vocabulary and IDF weights. Validation and
test data are transformed with that fitted vectorizer and never participate
in vocabulary construction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfConfig:
    dataset_dir: str = "ml/data/processed"
    output_dir: str = "ml/artifacts/tfidf"
    text_column: str = "processed_text"
    label_column: str = "label_code"
    max_features: int | None = 20000
    min_df: int = 1
    max_df: float = 1.0
    ngram_min: int = 1
    ngram_max: int = 2
    sublinear_tf: bool = True
    use_idf: bool = True
    norm: str | None = "l2"


@dataclass
class TfidfResult:
    train_rows: int
    validation_rows: int
    test_rows: int
    feature_count: int
    train_nonzero: int
    validation_nonzero: int
    test_nonzero: int
    output_dir: str


def _read_split(path: Path, config: TfidfConfig, *, required: bool) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Dataset split not found: {path}. Run the Phase 4–5 prepare_dataset.py pipeline first."
            )
        raise FileNotFoundError(f"Dataset split not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    missing = [column for column in (config.text_column, config.label_column) if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing required column(s) {missing}. "
            "Run Phase 5 and use processed_text/label_code, or specify the correct columns."
        )
    return frame


def _validate_config(config: TfidfConfig) -> None:
    if config.max_features is not None and config.max_features < 1:
        raise ValueError("max_features must be a positive integer or None.")
    if config.min_df < 1:
        raise ValueError("min_df must be at least 1 for this dataset pipeline.")
    if not 0 < config.max_df <= 1:
        raise ValueError("max_df must be in the interval (0, 1].")
    if config.ngram_min < 1 or config.ngram_max < config.ngram_min:
        raise ValueError("ngram range must satisfy 1 <= ngram_min <= ngram_max.")


def _matrix_stats(matrix: Any) -> dict[str, Any]:
    total = int(matrix.shape[0] * matrix.shape[1])
    nonzero = int(matrix.nnz)
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nonzero_values": nonzero,
        "sparsity": round(1.0 - nonzero / total, 6) if total else 1.0,
    }


def _save_labels(frame: pd.DataFrame, path: Path, config: TfidfConfig) -> None:
    labels = pd.DataFrame({
        "row_index": range(len(frame)),
        "label_code": frame[config.label_column].astype(str),
    })
    if "label_name" in frame.columns:
        labels["label_name"] = frame["label_name"].astype(str)
    labels.to_csv(path, index=False, encoding="utf-8-sig")


def build_tfidf(config: TfidfConfig | None = None) -> TfidfResult:
    """Fit TF-IDF on train only, transform the other splits, and save artifacts."""
    config = config or TfidfConfig()
    _validate_config(config)

    dataset_dir = Path(config.dataset_dir)
    train = _read_split(dataset_dir / "train.csv", config, required=True)
    validation = _read_split(dataset_dir / "validation.csv", config, required=False)
    test = _read_split(dataset_dir / "test.csv", config, required=False)

    if train.empty:
        raise ValueError("train.csv is empty. At least one labeled training row is required.")
    for split_name, frame in (("train", train), ("validation", validation), ("test", test)):
        if frame[config.label_column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{split_name}.csv contains a row without {config.label_column}.")

    train_texts = train[config.text_column].fillna("").astype(str).tolist()
    if not any(text.strip() for text in train_texts):
        raise ValueError(
            f"All values in {config.text_column!r} of train.csv are empty. "
            "Re-run NLP preprocessing or select a populated text column."
        )

    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        strip_accents=None,
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        sublinear_tf=config.sublinear_tf,
        use_idf=config.use_idf,
        norm=config.norm,
    )
    try:
        x_train = vectorizer.fit_transform(train_texts).tocsr()
    except ValueError as exc:
        raise ValueError(
            "Could not build TF-IDF vocabulary from train.csv. Check that processed_text "
            "contains usable segmented tokens and that min_df/max_df are not too restrictive."
        ) from exc

    x_validation = vectorizer.transform(
        validation[config.text_column].fillna("").astype(str).tolist()
    ).tocsr()
    x_test = vectorizer.transform(
        test[config.text_column].fillna("").astype(str).tolist()
    ).tocsr()

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output / "tfidf_vectorizer.joblib", compress=3)
    save_npz(output / "X_train.npz", x_train, compressed=True)
    save_npz(output / "X_validation.npz", x_validation, compressed=True)
    save_npz(output / "X_test.npz", x_test, compressed=True)
    _save_labels(train, output / "y_train.csv", config)
    _save_labels(validation, output / "y_validation.csv", config)
    _save_labels(test, output / "y_test.csv", config)

    feature_names = vectorizer.get_feature_names_out()
    vocabulary = pd.DataFrame({
        "feature_index": range(len(feature_names)),
        "term": feature_names,
        "idf": vectorizer.idf_,
    })
    vocabulary.to_csv(output / "vocabulary.csv", index=False, encoding="utf-8-sig")

    matrices = {
        "train": _matrix_stats(x_train),
        "validation": _matrix_stats(x_validation),
        "test": _matrix_stats(x_test),
    }
    summary = {
        "method": "TfidfVectorizer",
        "fit_source": "train.csv only",
        "transform_sources": ["validation.csv", "test.csv"],
        "text_column": config.text_column,
        "label_column": config.label_column,
        "vectorizer_parameters": {
            "analyzer": "word",
            "tokenizer": "str.split (Phase 5 segmented tokens)",
            "lowercase": False,
            "strip_accents": None,
            "ngram_range": [config.ngram_min, config.ngram_max],
            "min_df": config.min_df,
            "max_df": config.max_df,
            "max_features": config.max_features,
            "sublinear_tf": config.sublinear_tf,
            "use_idf": config.use_idf,
            "norm": config.norm,
        },
        "feature_count": int(len(feature_names)),
        "matrices": matrices,
        "sklearn_version": sklearn.__version__,
        "artifacts": [
            "tfidf_vectorizer.joblib",
            "X_train.npz",
            "X_validation.npz",
            "X_test.npz",
            "y_train.csv",
            "y_validation.csv",
            "y_test.csv",
            "vocabulary.csv",
        ],
        "notes": [
            "The vocabulary and inverse document frequency values are learned from train.csv only.",
            "Validation and test rows are transformed without refitting the vectorizer.",
            "The text is already segmented by Phase 5; whitespace tokenization preserves compound tokens joined by underscores.",
            "Vietnamese diacritics are preserved; strip_accents is disabled.",
            "Sparse CSR matrices are saved as compressed NPZ files to reduce disk usage.",
            "Each y_*.csv row aligns by row_index with the corresponding X_*.npz matrix row.",
        ],
    }
    (output / "tfidf_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return TfidfResult(
        train_rows=int(x_train.shape[0]),
        validation_rows=int(x_validation.shape[0]),
        test_rows=int(x_test.shape[0]),
        feature_count=int(x_train.shape[1]),
        train_nonzero=int(x_train.nnz),
        validation_nonzero=int(x_validation.nnz),
        test_nonzero=int(x_test.nnz),
        output_dir=str(output),
    )


def load_tfidf_vectorizer(path: str | Path) -> TfidfVectorizer:
    """Load a previously saved vectorizer for future inference."""
    vectorizer = joblib.load(path)
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError(f"Artifact at {path} is not a scikit-learn TfidfVectorizer.")
    return vectorizer
