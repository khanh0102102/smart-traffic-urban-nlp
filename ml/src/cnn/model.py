"""Train, evaluate and serve a 1D CNN for Vietnamese incident classification.

The CNN consumes the padded integer sequences produced by Phase 7. The tokenizer
and label mapping are loaded as fitted artifacts; they are never refitted here.
Only the training split updates model weights. Validation is used for early
stopping, while the test split is held out for final evaluation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import LabelEncoder

from src.data_processing.categories import CATEGORIES
from src.embedding import SequenceTokenizer
from src.nlp_preprocessing import preprocess_vietnamese_text


@dataclass
class CNNConfig:
    embedding_dir: str = "ml/artifacts/embedding"
    output_dir: str = "ml/artifacts/cnn"
    epochs: int = 20
    batch_size: int = 32
    filters: int = 128
    kernel_sizes: tuple[int, ...] = (3, 4, 5)
    dense_units: int = 128
    dropout: float = 0.4
    embedding_dropout: float = 0.2
    learning_rate: float = 1e-3
    random_state: int = 42
    early_stopping_patience: int = 4
    class_weight: str | None = "balanced"


@dataclass
class CNNResult:
    train_rows: int
    validation_rows: int
    test_rows: int
    vocabulary_size: int
    sequence_length: int
    trained_class_count: int
    epochs_trained: int
    validation_accuracy: float | None
    test_accuracy: float | None
    model_path: str
    bundle_path: str
    output_dir: str


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for the CNN phase. Install the optional "
            "dependency with: python -m pip install -r ml/requirements-embedding.txt"
        ) from exc
    return tf


def _register_custom_layers(tf):
    """Register custom padding layers before building or loading a model."""
    registered = getattr(_register_custom_layers, "_layers", None)
    if registered is not None:
        return registered

    @tf.keras.utils.register_keras_serializable(package="STNLP")
    class ZeroPadEmbeddingOutput(tf.keras.layers.Layer):
        """Force the embedding vector for PAD token ID 0 to remain zero."""

        def call(self, inputs):
            embeddings, token_ids = inputs
            valid = tf.expand_dims(tf.not_equal(token_ids, 0), axis=-1)
            return tf.where(valid, embeddings, tf.zeros_like(embeddings))

        def get_config(self):
            return super().get_config()

    @tf.keras.utils.register_keras_serializable(package="STNLP")
    class MaskedGlobalMaxPooling1D(tf.keras.layers.Layer):
        """Global max pooling that excludes PAD token positions (token ID 0)."""

        def call(self, inputs):
            features, token_ids = inputs
            valid = tf.not_equal(token_ids, 0)
            expanded = tf.expand_dims(valid, axis=-1)
            floor = tf.cast(-1e9, features.dtype)
            masked = tf.where(expanded, features, floor)
            pooled = tf.reduce_max(masked, axis=1)
            has_tokens = tf.reduce_any(valid, axis=1, keepdims=True)
            return tf.where(has_tokens, pooled, tf.zeros_like(pooled))

        def get_config(self):
            return super().get_config()

    _register_custom_layers._layers = (
        MaskedGlobalMaxPooling1D,
        ZeroPadEmbeddingOutput,
    )
    return _register_custom_layers._layers


def _validate_config(config: CNNConfig) -> None:
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if config.filters < 1:
        raise ValueError("filters must be at least 1.")
    if not config.kernel_sizes or any(size < 1 for size in config.kernel_sizes):
        raise ValueError("kernel_sizes must contain positive integers.")
    if len(set(config.kernel_sizes)) != len(config.kernel_sizes):
        raise ValueError("kernel_sizes must not contain duplicate values.")
    if config.dense_units < 1:
        raise ValueError("dense_units must be at least 1.")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in the interval [0, 1).")
    if not 0 <= config.embedding_dropout < 1:
        raise ValueError("embedding_dropout must be in the interval [0, 1).")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative.")
    if config.class_weight not in {None, "balanced"}:
        raise ValueError("class_weight must be None or 'balanced'.")


def build_cnn_model(
    vocab_size: int,
    sequence_length: int,
    class_count: int = 11,
    *,
    embedding_dim: int = 128,
    filters: int = 128,
    kernel_sizes: tuple[int, ...] = (3, 4, 5),
    dense_units: int = 128,
    dropout: float = 0.4,
    embedding_dropout: float = 0.2,
    learning_rate: float = 1e-3,
):
    """Build a parallel-kernel TextCNN with padding-aware global max pooling."""
    tf = _tensorflow()
    pooling_layer, zero_pad_layer = _register_custom_layers(tf)
    if vocab_size < 2:
        raise ValueError("vocab_size must include at least PAD and OOV tokens.")
    if sequence_length < 1 or class_count < 2 or embedding_dim < 1:
        raise ValueError("sequence_length, class_count and embedding_dim are invalid.")
    if not kernel_sizes or any(size < 1 for size in kernel_sizes):
        raise ValueError("kernel_sizes must contain positive integers.")

    token_ids = tf.keras.Input(
        shape=(sequence_length,), dtype="int32", name="token_ids"
    )
    embedded = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="token_embedding",
    )(token_ids)
    embedded = zero_pad_layer(name="zero_pad_embedding_output")(
        [embedded, token_ids]
    )
    embedded = tf.keras.layers.SpatialDropout1D(
        embedding_dropout, name="embedding_spatial_dropout"
    )(embedded)

    pooled_branches = []
    for kernel_size in kernel_sizes:
        features = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            name=f"conv1d_kernel_{kernel_size}",
        )(embedded)
        pooled = pooling_layer(name=f"masked_global_max_{kernel_size}")(
            [features, token_ids]
        )
        pooled_branches.append(pooled)

    if len(pooled_branches) == 1:
        combined = pooled_branches[0]
    else:
        combined = tf.keras.layers.Concatenate(name="multi_kernel_features")(
            pooled_branches
        )
    hidden = tf.keras.layers.Dense(
        dense_units, activation="relu", name="classifier_dense"
    )(combined)
    hidden = tf.keras.layers.Dropout(dropout, name="classifier_dropout")(hidden)
    probabilities = tf.keras.layers.Dense(
        class_count, activation="softmax", name="category_probabilities"
    )(hidden)

    model = tf.keras.Model(
        inputs=token_ids, outputs=probabilities, name="stnlp_text_cnn_1d"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _load_array(path: Path, *, dtype: Any) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding artifact not found: {path}. Run "
            "ml/scripts/build_embedding.py before training the CNN."
        )
    array = np.load(path, allow_pickle=False)
    return np.asarray(array, dtype=dtype)


def _load_embedding_assets(embedding_dir: Path):
    tokenizer_path = embedding_dir / "tokenizer.joblib"
    encoder_path = embedding_dir / "label_encoder.joblib"
    config_path = embedding_dir / "embedding_config.json"
    for path in (tokenizer_path, encoder_path, config_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Phase 7 asset is missing: {path}. Rebuild the embedding artifacts first."
            )
    tokenizer = joblib.load(tokenizer_path)
    label_encoder = joblib.load(encoder_path)
    if not isinstance(tokenizer, SequenceTokenizer):
        raise TypeError("tokenizer.joblib is not a Phase 7 SequenceTokenizer.")
    if not isinstance(label_encoder, LabelEncoder):
        raise TypeError("label_encoder.joblib is not a scikit-learn LabelEncoder.")
    if not np.array_equal(
        label_encoder.classes_, np.asarray([category.code for category in CATEGORIES])
    ):
        raise ValueError("The saved label encoder must use canonical categories R01-R11.")
    embedding_config = json.loads(config_path.read_text(encoding="utf-8"))
    summary_path = embedding_dir / "embedding_summary.json"
    if summary_path.exists():
        embedding_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        embedding_config["padding"] = embedding_summary.get("padding", "post")
        embedding_config["truncating"] = embedding_summary.get("truncating", "post")
    else:
        embedding_config["padding"] = "post"
        embedding_config["truncating"] = "post"
    return tokenizer, label_encoder, embedding_config


def _load_split(embedding_dir: Path, split_name: str):
    x = _load_array(embedding_dir / f"X_{split_name}.npy", dtype=np.int32)
    y = _load_array(embedding_dir / f"y_{split_name}.npy", dtype=np.int64).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"X_{split_name}.npy must be a 2D padded sequence matrix.")
    if x.shape[0] != len(y):
        raise ValueError(
            f"{split_name} row mismatch: sequence matrix has {x.shape[0]} rows "
            f"but label array has {len(y)} values."
        )
    if x.size and (x.min() < 0):
        raise ValueError(f"X_{split_name}.npy contains a negative token ID.")
    return x, y


def _class_names() -> dict[str, str]:
    return {category.code: category.name for category in CATEGORIES}


def _evaluate_split(
    model,
    x: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    split_name: str,
    output: Path,
) -> dict[str, Any] | None:
    class_codes = label_encoder.classes_.tolist()
    category_names = _class_names()
    prediction_columns = [
        "row_index", "actual_label_code", "predicted_label_code",
        "actual_label_name", "predicted_label_name", "confidence",
        "has_known_tokens", "correct",
    ]
    if len(y) == 0:
        pd.DataFrame(columns=prediction_columns).to_csv(
            output / f"predictions_{split_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        empty_matrix = pd.DataFrame(
            np.zeros((len(class_codes), len(class_codes)), dtype=np.int64),
            index=class_codes,
            columns=class_codes,
        )
        empty_matrix.index.name = "actual_label_code"
        empty_matrix.to_csv(
            output / f"confusion_matrix_{split_name}.csv",
            encoding="utf-8-sig",
        )
        return None

    probabilities = np.asarray(
        model.predict(x, batch_size=256, verbose=0), dtype=float
    )
    predicted_ids = probabilities.argmax(axis=1).astype(np.int64)
    confidence = probabilities.max(axis=1)
    actual_codes = label_encoder.inverse_transform(y)
    predicted_codes = label_encoder.inverse_transform(predicted_ids)
    predictions = pd.DataFrame({
        "row_index": np.arange(len(y), dtype=np.int64),
        "actual_label_code": actual_codes,
        "predicted_label_code": predicted_codes,
        "actual_label_name": [category_names[code] for code in actual_codes],
        "predicted_label_name": [category_names[code] for code in predicted_codes],
        "confidence": confidence,
        "has_known_tokens": np.any(x > 1, axis=1),
        "correct": y == predicted_ids,
    })
    predictions.to_csv(
        output / f"predictions_{split_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_class_ids = np.arange(len(class_codes), dtype=np.int64)
    report = classification_report(
        y,
        predicted_ids,
        labels=all_class_ids,
        target_names=class_codes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y, predicted_ids, labels=all_class_ids)
    confusion = pd.DataFrame(matrix, index=class_codes, columns=class_codes)
    confusion.index.name = "actual_label_code"
    confusion.to_csv(
        output / f"confusion_matrix_{split_name}.csv",
        encoding="utf-8-sig",
    )
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted_ids)),
        "log_loss": float(log_loss(y, probabilities, labels=all_class_ids)),
        "classification_report": report,
        "confusion_matrix_file": f"confusion_matrix_{split_name}.csv",
        "predictions_file": f"predictions_{split_name}.csv",
    }


def train_cnn(config: CNNConfig | None = None) -> CNNResult:
    """Train a 1D CNN from Phase 7 arrays and evaluate held-out splits."""
    config = config or CNNConfig()
    _validate_config(config)
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(config.random_state)

    embedding_dir = Path(config.embedding_dir)
    tokenizer, label_encoder, embedding_config = _load_embedding_assets(embedding_dir)
    x_train, y_train = _load_split(embedding_dir, "train")
    x_validation, y_validation = _load_split(embedding_dir, "validation")
    x_test, y_test = _load_split(embedding_dir, "test")
    if len(y_train) == 0:
        raise ValueError("The CNN requires at least one labeled training row.")
    if x_train.shape[1] < 1:
        raise ValueError("The sequence length must be at least one.")
    if any(x.shape[1] != x_train.shape[1] for x in (x_validation, x_test)):
        raise ValueError("Train, validation and test sequence lengths must match.")
    for split_name, x, y in (
        ("train", x_train, y_train),
        ("validation", x_validation, y_validation),
        ("test", x_test, y_test),
    ):
        if y.size and (y.min() < 0 or y.max() >= len(label_encoder.classes_)):
            raise ValueError(f"{split_name} labels fall outside canonical R01-R11 IDs.")
        if x.size and x.max() >= tokenizer.vocab_size:
            raise ValueError(f"{split_name} contains token IDs outside the saved vocabulary.")

    model = build_cnn_model(
        vocab_size=tokenizer.vocab_size,
        sequence_length=x_train.shape[1],
        class_count=len(label_encoder.classes_),
        embedding_dim=int(embedding_config.get("output_dim", 128)),
        filters=config.filters,
        kernel_sizes=config.kernel_sizes,
        dense_units=config.dense_units,
        dropout=config.dropout,
        embedding_dropout=config.embedding_dropout,
        learning_rate=config.learning_rate,
    )

    callbacks = []
    if len(y_validation) > 0 and config.early_stopping_patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            mode="min",
        ))

    class_weights = None
    if config.class_weight == "balanced":
        unique_ids, counts = np.unique(y_train, return_counts=True)
        total = float(len(y_train))
        class_weights = {
            int(class_id): total / (len(unique_ids) * float(count))
            for class_id, count in zip(unique_ids, counts)
        }

    fit_kwargs: dict[str, Any] = {
        "x": x_train,
        "y": y_train,
        "epochs": config.epochs,
        "batch_size": min(config.batch_size, len(y_train)),
        "shuffle": True,
        "callbacks": callbacks,
        "class_weight": class_weights,
        "verbose": 0,
    }
    if len(y_validation) > 0:
        fit_kwargs["validation_data"] = (x_validation, y_validation)
    history = model.fit(**fit_kwargs)

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "cnn_model.keras"
    model.save(model_path)

    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    history_frame.to_csv(
        output / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_metrics = _evaluate_split(
        model, x_validation, y_validation, label_encoder, "validation", output
    )
    test_metrics = _evaluate_split(
        model, x_test, y_test, label_encoder, "test", output
    )

    bundle = {
        "model_type": "TensorFlow/Keras 1D TextCNN",
        "model_file": model_path.name,
        "tokenizer": tokenizer,
        "label_encoder": label_encoder,
        "category_names": _class_names(),
        "sequence": {
            "max_length": int(x_train.shape[1]),
            "padding": embedding_config.get("padding", "post"),
            "truncating": embedding_config.get("truncating", "post"),
            "pad_token_id": 0,
            "oov_token_id": 1,
        },
        "preprocessing": {
            "pipeline": "src.nlp_preprocessing.preprocess_vietnamese_text",
            "remove_stopwords": True,
            "text_column": "processed_text",
        },
        "parameters": asdict(config),
        "trained_class_ids": sorted(int(value) for value in np.unique(y_train)),
    }
    bundle_path = output / "cnn_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    summary = {
        "model": "1D TextCNN",
        "architecture": {
            "input": "fixed-length integer token sequence",
            "embedding": {
                "input_dim": tokenizer.vocab_size,
                "output_dim": int(embedding_config.get("output_dim", 128)),
                "trainable": True,
            },
            "embedding_dropout": config.embedding_dropout,
            "convolution": {
                "type": "parallel Conv1D branches",
                "filters_per_branch": config.filters,
                "kernel_sizes": list(config.kernel_sizes),
                "padding": "same",
                "activation": "relu",
            },
            "pooling": "MaskedGlobalMaxPooling1D excludes PAD token ID 0",
            "classifier": [
                {"type": "Dense", "units": config.dense_units, "activation": "relu"},
                {"type": "Dropout", "rate": config.dropout},
                {"type": "Dense", "units": len(label_encoder.classes_), "activation": "softmax"},
            ],
        },
        "parameters": asdict(config),
        "training_source": "X_train.npy and y_train.npy from Phase 7 only",
        "validation_usage": (
            "Early stopping and validation evaluation only."
            if len(y_validation) else
            "Validation split is empty; early stopping was not used."
        ),
        "test_usage": "Final held-out evaluation only; never used for model fitting.",
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_validation)),
        "test_rows": int(len(y_test)),
        "vocabulary_size": tokenizer.vocab_size,
        "sequence_length": int(x_train.shape[1]),
        "trained_class_codes": [
            str(label_encoder.inverse_transform([int(class_id)])[0])
            for class_id in sorted(np.unique(y_train))
        ],
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "epochs_trained": int(len(history_frame)),
        "artifacts": [
            "cnn_model.keras",
            "cnn_bundle.joblib",
            "training_history.csv",
            "predictions_validation.csv",
            "predictions_test.csv",
            "confusion_matrix_validation.csv",
            "confusion_matrix_test.csv",
            "cnn_summary.json",
        ],
        "notes": [
            "Tokenizer and label encoder are reused from Phase 7 and are not refitted.",
            "Only training sequences update model weights; validation is used for early stopping when non-empty.",
            "The test split is reserved for final evaluation.",
            "Confidence is the maximum softmax score and is not guaranteed to be calibrated.",
            "CNN performance is not meaningful on the tiny synthetic sample; use a larger manually verified dataset.",
        ],
    }
    (output / "cnn_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    final_validation_accuracy = (
        validation_metrics["accuracy"] if validation_metrics is not None else None
    )
    final_test_accuracy = test_metrics["accuracy"] if test_metrics is not None else None
    return CNNResult(
        train_rows=int(len(y_train)),
        validation_rows=int(len(y_validation)),
        test_rows=int(len(y_test)),
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=int(x_train.shape[1]),
        trained_class_count=int(len(np.unique(y_train))),
        epochs_trained=int(len(history_frame)),
        validation_accuracy=final_validation_accuracy,
        test_accuracy=final_test_accuracy,
        model_path=str(model_path),
        bundle_path=str(bundle_path),
        output_dir=str(output),
    )


def predict_cnn_reports(
    texts: Iterable[object],
    *,
    model_path: str | Path = "ml/artifacts/cnn/cnn_bundle.joblib",
    already_processed: bool = False,
) -> list[dict[str, Any]]:
    """Predict raw or Phase 5 processed reports with a saved CNN bundle."""
    tf = _tensorflow()
    bundle_path = Path(model_path)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"CNN bundle not found: {bundle_path}. Train the model first."
        )
    bundle = joblib.load(bundle_path)
    if bundle.get("model_type") != "TensorFlow/Keras 1D TextCNN":
        raise ValueError("The selected bundle is not a Phase 10 CNN model.")

    tokenizer = bundle["tokenizer"]
    label_encoder = bundle["label_encoder"]
    sequence_config = bundle["sequence"]
    pooling_layer, zero_pad_layer = _register_custom_layers(tf)
    keras_path = bundle_path.parent / bundle["model_file"]
    if not keras_path.exists():
        raise FileNotFoundError(f"Saved Keras model file is missing: {keras_path}.")
    model = tf.keras.models.load_model(
        keras_path,
        custom_objects={
            "MaskedGlobalMaxPooling1D": pooling_layer,
            "ZeroPadEmbeddingOutput": zero_pad_layer,
        },
        compile=False,
    )

    raw_values = ["" if value is None else str(value) for value in texts]
    processed_values = []
    for value in raw_values:
        if already_processed:
            processed_values.append(value)
        else:
            processed_values.append(
                preprocess_vietnamese_text(value).processed_text
            )

    sequences = tokenizer.texts_to_sequences(processed_values)
    from src.embedding import pad_sequences
    matrix = pad_sequences(
        sequences,
        int(sequence_config["max_length"]),
        padding=sequence_config.get("padding", "post"),
        truncating=sequence_config.get("truncating", "post"),
    )
    if len(matrix) == 0:
        return []

    probabilities = np.asarray(model.predict(matrix, batch_size=256, verbose=0), dtype=float)
    predicted_ids = probabilities.argmax(axis=1)
    category_names = bundle["category_names"]
    results: list[dict[str, Any]] = []
    for row_index, class_id in enumerate(predicted_ids):
        code = str(label_encoder.inverse_transform([int(class_id)])[0])
        class_probabilities = {
            str(label_encoder.inverse_transform([int(index)])[0]): float(probability)
            for index, probability in enumerate(probabilities[row_index])
        }
        has_known_tokens = bool(np.any(matrix[row_index] > 1))
        results.append({
            "raw_content": raw_values[row_index],
            "processed_text": processed_values[row_index],
            "predicted_label_code": code,
            "predicted_label_name": category_names[code],
            "confidence": float(probabilities[row_index, class_id]),
            "class_probabilities": class_probabilities,
            "has_known_tokens": has_known_tokens,
            "warning": (
                "No token from the report was found in the Phase 7 training vocabulary; "
                "the prediction may be unreliable."
                if not has_known_tokens else None
            ),
        })
    return results
