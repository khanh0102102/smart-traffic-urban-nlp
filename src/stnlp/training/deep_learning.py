"""Tokenizer + Embedding + CNN/LSTM training utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from stnlp.labels import LABEL_CODES
from stnlp.preprocessing.text import preprocess_text


def _make_model(model_name: str, vocab_size: int, max_len: int, num_classes: int):
    inputs = tf.keras.Input(shape=(max_len,), dtype="int32", name="token_ids")
    x = tf.keras.layers.Embedding(
        input_dim=max(vocab_size, 2), output_dim=128, name="word_embedding"
    )(inputs)
    if model_name == "cnn":
        x = tf.keras.layers.Conv1D(128, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
    elif model_name == "lstm":
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    else:
        raise ValueError(f"Deep model không được hỗ trợ: {model_name}")
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_deep_model(
    model_name: str,
    train_texts,
    train_labels,
    validation_texts,
    validation_labels,
    output_dir: Path,
    epochs: int = 12,
    batch_size: int = 32,
    max_words: int = 30_000,
):
    """Fit tokenizer on train only, use validation for early stopping, save all inference assets."""
    if model_name not in {"cnn", "lstm"}:
        raise ValueError("model_name phải là cnn hoặc lstm")

    train_clean = [preprocess_text(t) for t in train_texts]
    validation_clean = [preprocess_text(t) for t in validation_texts]
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>", filters="", lower=False)
    tokenizer.fit_on_texts(train_clean)

    train_sequences = tokenizer.texts_to_sequences(train_clean)
    validation_sequences = tokenizer.texts_to_sequences(validation_clean)
    lengths = [len(row) for row in train_sequences if row]
    max_len = max(8, min(120, int(np.percentile(lengths, 95)) if lengths else 8))
    x_train = pad_sequences(train_sequences, maxlen=max_len, padding="post", truncating="post")
    x_validation = pad_sequences(
        validation_sequences, maxlen=max_len, padding="post", truncating="post"
    )

    label_to_index = {code: index for index, code in enumerate(LABEL_CODES)}
    y_train = np.asarray([label_to_index[label] for label in train_labels], dtype="int32")
    y_validation = np.asarray(
        [label_to_index[label] for label in validation_labels], dtype="int32"
    )
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    model = _make_model(
        model_name,
        min(max_words, len(tokenizer.word_index) + 1),
        max_len,
        len(LABEL_CODES),
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.keras")
    (output_dir / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    (output_dir / "labels.json").write_text(
        json.dumps(list(LABEL_CODES), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "model_name": model_name,
        "max_len": max_len,
        "max_words": max_words,
        "embedding_dim": 128,
        "epochs_completed": len(history.history.get("loss", [])),
        "preprocessing": "stnlp.preprocessing.text.preprocess_text",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model, tokenizer, max_len
