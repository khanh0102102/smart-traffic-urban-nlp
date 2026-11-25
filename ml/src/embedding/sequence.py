"""Train-only tokenizer, integer sequences, padding and Keras embedding helpers.

Vocabulary is fitted on train.csv only. Validation/test words not observed in
training are mapped to OOV_INDEX. Class IDs are defined by the canonical
R01-R11 taxonomy, so class-to-index mappings remain stable across datasets.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.data_processing.categories import CATEGORIES

PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"
PAD_INDEX = 0
OOV_INDEX = 1


@dataclass
class SequenceTokenizer:
    """Small deterministic tokenizer for Phase 5 whitespace-segmented text."""

    max_vocab_size: int | None = 30000
    min_frequency: int = 1
    lowercase: bool = False
    word_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_vocab_size is not None and self.max_vocab_size < 2:
            raise ValueError("max_vocab_size must be at least 2 (PAD and OOV tokens).")
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be at least 1.")

    def _tokens(self, text: object) -> list[str]:
        value = "" if text is None else str(text)
        if self.lowercase:
            value = value.casefold()
        return [token for token in value.split() if token]

    def fit(self, texts: Iterable[object]) -> "SequenceTokenizer":
        """Learn token IDs from training texts only, with deterministic ties."""
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(self._tokens(text))

        eligible = [
            (token, count)
            for token, count in counts.items()
            if count >= self.min_frequency and token not in {PAD_TOKEN, OOV_TOKEN}
        ]
        eligible.sort(key=lambda item: (-item[1], item[0]))

        capacity = None if self.max_vocab_size is None else self.max_vocab_size - 2
        if capacity is not None:
            eligible = eligible[:capacity]

        self.word_index = {PAD_TOKEN: PAD_INDEX, OOV_TOKEN: OOV_INDEX}
        for token, _count in eligible:
            self.word_index[token] = len(self.word_index)
        return self

    @property
    def vocab_size(self) -> int:
        """Embedding input dimension: number of reserved + learned token IDs."""
        return len(self.word_index)

    def texts_to_sequences(self, texts: Iterable[object]) -> list[list[int]]:
        if not self.word_index:
            raise RuntimeError("Tokenizer has not been fitted. Call fit() using train texts first.")
        sequences: list[list[int]] = []
        for text in texts:
            tokens = self._tokens(text)
            sequences.append([self.word_index.get(token, OOV_INDEX) for token in tokens])
        return sequences

    def to_dict(self) -> dict[str, object]:
        return {
            "max_vocab_size": self.max_vocab_size,
            "min_frequency": self.min_frequency,
            "lowercase": self.lowercase,
            "word_index": self.word_index,
            "pad_token": PAD_TOKEN,
            "pad_index": PAD_INDEX,
            "oov_token": OOV_TOKEN,
            "oov_index": OOV_INDEX,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SequenceTokenizer":
        tokenizer = cls(
            max_vocab_size=payload.get("max_vocab_size"),
            min_frequency=int(payload.get("min_frequency", 1)),
            lowercase=bool(payload.get("lowercase", False)),
        )
        word_index = payload.get("word_index")
        if not isinstance(word_index, dict):
            raise ValueError("Tokenizer artifact must contain a word_index object.")
        tokenizer.word_index = {str(token): int(index) for token, index in word_index.items()}
        if tokenizer.word_index.get(PAD_TOKEN) != PAD_INDEX:
            raise ValueError("Tokenizer artifact has an invalid PAD token index.")
        if tokenizer.word_index.get(OOV_TOKEN) != OOV_INDEX:
            raise ValueError("Tokenizer artifact has an invalid OOV token index.")
        ids = sorted(tokenizer.word_index.values())
        if ids != list(range(len(ids))):
            raise ValueError("Tokenizer word IDs must be contiguous and start at zero.")
        return tokenizer


def pad_sequences(
    sequences: Iterable[Sequence[int]],
    max_length: int,
    *,
    padding: str = "post",
    truncating: str = "post",
    value: int = PAD_INDEX,
) -> np.ndarray:
    """Pad/truncate integer sequences into a fixed-width int32 matrix."""
    if max_length < 1:
        raise ValueError("max_length must be a positive integer.")
    if padding not in {"pre", "post"}:
        raise ValueError("padding must be 'pre' or 'post'.")
    if truncating not in {"pre", "post"}:
        raise ValueError("truncating must be 'pre' or 'post'.")

    rows = list(sequences)
    output = np.full((len(rows), max_length), value, dtype=np.int32)
    for row_index, sequence in enumerate(rows):
        tokens = [int(token) for token in sequence]
        if len(tokens) > max_length:
            tokens = tokens[-max_length:] if truncating == "pre" else tokens[:max_length]
        if not tokens:
            continue
        values = np.asarray(tokens, dtype=np.int32)
        if padding == "post":
            output[row_index, :len(values)] = values
        else:
            output[row_index, -len(values):] = values
    return output


@dataclass
class LabelEncoderBundle:
    """Fixed label-to-ID mapping based on the canonical database taxonomy."""

    encoder: LabelEncoder = field(default_factory=LabelEncoder)

    def fit_taxonomy(self) -> "LabelEncoderBundle":
        self.encoder.fit([category.code for category in CATEGORIES])
        return self

    @property
    def classes_(self) -> np.ndarray:
        if not hasattr(self.encoder, "classes_"):
            raise RuntimeError("Label encoder has not been initialized.")
        return self.encoder.classes_

    def transform(self, labels: Iterable[object]) -> np.ndarray:
        if not hasattr(self.encoder, "classes_"):
            self.fit_taxonomy()
        normalized = [str(label).strip().upper() for label in labels]
        known = set(self.encoder.classes_)
        unknown = sorted(set(normalized) - known)
        if unknown:
            raise ValueError(f"Unknown label code(s) {unknown}; expected canonical codes R01-R11.")
        return self.encoder.transform(normalized).astype(np.int64)

    def inverse_transform(self, encoded: Iterable[int]) -> np.ndarray:
        if not hasattr(self.encoder, "classes_"):
            self.fit_taxonomy()
        return self.encoder.inverse_transform(np.asarray(list(encoded), dtype=np.int64))


@dataclass
class EmbeddingConfig:
    dataset_dir: str = "ml/data/processed"
    output_dir: str = "ml/artifacts/embedding"
    text_column: str = "processed_text"
    label_column: str = "label_code"
    max_vocab_size: int | None = 30000
    min_frequency: int = 1
    max_length: int = 200
    padding: str = "post"
    truncating: str = "post"
    embedding_dim: int = 128
    lowercase: bool = False
    mask_zero: bool = True


@dataclass
class EmbeddingResult:
    train_rows: int
    validation_rows: int
    test_rows: int
    vocabulary_size: int
    sequence_length: int
    class_count: int
    embedding_dim: int
    output_dir: str


def _read_split(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}. Run the Phase 4–5 prepare_dataset.py pipeline first."
        )
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    missing = [column for column in (text_column, label_column) if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing required column(s) {missing}. "
            f"Expected Phase 5 columns {text_column!r} and {label_column!r}."
        )
    return frame


def _validate_config(config: EmbeddingConfig) -> None:
    if config.max_vocab_size is not None and config.max_vocab_size < 2:
        raise ValueError("max_vocab_size must be at least 2.")
    if config.min_frequency < 1:
        raise ValueError("min_frequency must be at least 1.")
    if config.max_length < 1:
        raise ValueError("max_length must be at least 1.")
    if config.padding not in {"pre", "post"}:
        raise ValueError("padding must be either 'pre' or 'post'.")
    if config.truncating not in {"pre", "post"}:
        raise ValueError("truncating must be either 'pre' or 'post'.")
    if config.embedding_dim < 1:
        raise ValueError("embedding_dim must be a positive integer.")


def _save_split_labels(
    labels: np.ndarray,
    frame: pd.DataFrame,
    path: Path,
    category_names: dict[str, str],
) -> None:
    codes = frame["label_code"].astype(str).str.strip().str.upper().tolist()
    pd.DataFrame({
        "row_index": np.arange(len(codes), dtype=np.int64),
        "label_code": codes,
        "label_id": labels,
        "label_name": [category_names[code] for code in codes],
    }).to_csv(path, index=False, encoding="utf-8-sig")


def build_embedding_data(config: EmbeddingConfig | None = None) -> EmbeddingResult:
    """Fit tokenizer on train, pad all splits, and encode canonical labels."""
    config = config or EmbeddingConfig()
    _validate_config(config)

    dataset_dir = Path(config.dataset_dir)
    train = _read_split(dataset_dir / "train.csv", config.text_column, config.label_column)
    validation = _read_split(dataset_dir / "validation.csv", config.text_column, config.label_column)
    test = _read_split(dataset_dir / "test.csv", config.text_column, config.label_column)
    if train.empty:
        raise ValueError("train.csv is empty. At least one labeled training row is required.")

    train_texts = train[config.text_column].fillna("").astype(str).tolist()
    if not any(text.strip() for text in train_texts):
        raise ValueError(
            f"All values in {config.text_column!r} of train.csv are empty. "
            "Run Phase 5 NLP preprocessing or use a populated text column."
        )

    tokenizer = SequenceTokenizer(
        max_vocab_size=config.max_vocab_size,
        min_frequency=config.min_frequency,
        lowercase=config.lowercase,
    ).fit(train_texts)
    if tokenizer.vocab_size <= 2:
        raise ValueError("Training text produced no vocabulary tokens after tokenization.")

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    validation_sequences = tokenizer.texts_to_sequences(
        validation[config.text_column].fillna("").astype(str).tolist()
    )
    test_sequences = tokenizer.texts_to_sequences(
        test[config.text_column].fillna("").astype(str).tolist()
    )

    x_train = pad_sequences(
        train_sequences, config.max_length,
        padding=config.padding, truncating=config.truncating,
    )
    x_validation = pad_sequences(
        validation_sequences, config.max_length,
        padding=config.padding, truncating=config.truncating,
    )
    x_test = pad_sequences(
        test_sequences, config.max_length,
        padding=config.padding, truncating=config.truncating,
    )

    label_bundle = LabelEncoderBundle().fit_taxonomy()
    y_train = label_bundle.transform(train[config.label_column])
    y_validation = label_bundle.transform(validation[config.label_column])
    y_test = label_bundle.transform(test[config.label_column])

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "X_train.npy", x_train)
    np.save(output / "X_validation.npy", x_validation)
    np.save(output / "X_test.npy", x_test)
    np.save(output / "y_train.npy", y_train)
    np.save(output / "y_validation.npy", y_validation)
    np.save(output / "y_test.npy", y_test)
    joblib.dump(tokenizer, output / "tokenizer.joblib", compress=3)
    joblib.dump(label_bundle.encoder, output / "label_encoder.joblib", compress=3)
    (output / "tokenizer.json").write_text(
        json.dumps(tokenizer.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    vocabulary = pd.DataFrame(
        [{"token": token, "token_id": index}
         for token, index in tokenizer.word_index.items()]
    )
    vocabulary.to_csv(output / "vocabulary.csv", index=False, encoding="utf-8-sig")

    category_names = {category.code: category.name for category in CATEGORIES}
    class_mapping = pd.DataFrame({
        "label_id": np.arange(len(label_bundle.classes_), dtype=np.int64),
        "label_code": label_bundle.classes_,
        "label_name": [category_names[code] for code in label_bundle.classes_],
    })
    class_mapping.to_csv(output / "class_mapping.csv", index=False, encoding="utf-8-sig")
    for split_name, labels, frame in (
        ("train", y_train, train),
        ("validation", y_validation, validation),
        ("test", y_test, test),
    ):
        _save_split_labels(labels, frame, output / f"labels_{split_name}.csv", category_names)

    embedding_config = {
        "layer": "tensorflow.keras.layers.Embedding",
        "input_dim": tokenizer.vocab_size,
        "output_dim": config.embedding_dim,
        "sequence_length": config.max_length,
        "mask_zero": config.mask_zero,
        "padding_index": PAD_INDEX,
        "oov_index": OOV_INDEX,
        "weights": "random trainable initialization; learned during model training",
    }
    (output / "embedding_config.json").write_text(
        json.dumps(embedding_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "tokenizer_fit_source": "train.csv only",
        "validation_test_behavior": "transform with train vocabulary; unseen tokens map to OOV index 1",
        "vocabulary_size_including_special_tokens": tokenizer.vocab_size,
        "max_vocab_size": config.max_vocab_size,
        "min_frequency": config.min_frequency,
        "lowercase": config.lowercase,
        "special_tokens": {PAD_TOKEN: PAD_INDEX, OOV_TOKEN: OOV_INDEX},
        "sequence_length": config.max_length,
        "padding": config.padding,
        "truncating": config.truncating,
        "embedding_layer": embedding_config,
        "label_encoder_source": "fixed canonical taxonomy R01-R11 from Phase 3",
        "class_mapping": {
            str(code): int(index)
            for index, code in enumerate(label_bundle.classes_)
        },
        "split_shapes": {
            "X_train": list(x_train.shape),
            "X_validation": list(x_validation.shape),
            "X_test": list(x_test.shape),
            "y_train": list(y_train.shape),
            "y_validation": list(y_validation.shape),
            "y_test": list(y_test.shape),
        },
        "artifacts": [
            "X_train.npy", "X_validation.npy", "X_test.npy",
            "y_train.npy", "y_validation.npy", "y_test.npy",
            "tokenizer.joblib", "tokenizer.json", "label_encoder.joblib",
            "vocabulary.csv", "class_mapping.csv",
            "labels_train.csv", "labels_validation.csv", "labels_test.csv",
            "embedding_config.json",
        ],
        "notes": [
            "Tokenizer vocabulary is learned from train.csv only; validation/test never update it.",
            "Padding index 0 is reserved and masked by the Keras Embedding layer when mask_zero=True.",
            "Unknown tokens are mapped to OOV index 1.",
            "The canonical R01-R11 label mapping is fixed and does not depend on which classes happen to appear in train.",
            "Embedding weights are initialized by TensorFlow/Keras and learned during downstream model training.",
            "Sequence arrays are saved as int32; label IDs are saved as int64.",
        ],
    }
    (output / "embedding_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return EmbeddingResult(
        train_rows=int(len(train)),
        validation_rows=int(len(validation)),
        test_rows=int(len(test)),
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=config.max_length,
        class_count=int(len(label_bundle.classes_)),
        embedding_dim=config.embedding_dim,
        output_dir=str(output),
    )


def create_embedding_layer(
    vocab_size: int,
    embedding_dim: int = 128,
    *,
    mask_zero: bool = True,
    name: str = "token_embedding",
):
    """Create a trainable Keras Embedding layer (TensorFlow is optional).

    Install ml/requirements-embedding.txt before calling this function. The
    returned layer is intended to be added to a downstream Keras model.
    """
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2 (PAD and OOV tokens).")
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be a positive integer.")
    try:
        from tensorflow.keras.layers import Embedding
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required to create a Keras Embedding layer. "
            "Install with: python -m pip install -r ml/requirements-embedding.txt"
        ) from exc
    return Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=mask_zero,
        name=name,
    )


def load_embedding_assets(
    output_dir: str | Path = "ml/artifacts/embedding",
) -> tuple[SequenceTokenizer, LabelEncoder]:
    """Load the fitted tokenizer and taxonomy-based label encoder."""
    output = Path(output_dir)
    tokenizer_path = output / "tokenizer.joblib"
    label_encoder_path = output / "label_encoder.joblib"
    if not tokenizer_path.exists() or not label_encoder_path.exists():
        raise FileNotFoundError(
            f"Tokenizer/label encoder artifacts are missing under {output}. "
            "Run ml/scripts/build_embedding.py first."
        )
    tokenizer = joblib.load(tokenizer_path)
    label_encoder = joblib.load(label_encoder_path)
    if not isinstance(tokenizer, SequenceTokenizer):
        raise TypeError("Saved tokenizer artifact has an unexpected type.")
    if not isinstance(label_encoder, LabelEncoder):
        raise TypeError("Saved label encoder artifact has an unexpected type.")
    return tokenizer, label_encoder
