"""Tests for sequence tokenization, padding, labels and Keras embedding setup."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedding import (
    EmbeddingConfig,
    LabelEncoderBundle,
    SequenceTokenizer,
    build_embedding_data,
    create_embedding_layer,
    load_embedding_assets,
    pad_sequences,
)


def _write_split(path: Path, rows: list[tuple[str, str]]) -> None:
    pd.DataFrame(
        [{"processed_text": text, "label_code": label}
         for text, label in rows],
        columns=["processed_text", "label_code"],
    ).to_csv(path, index=False, encoding="utf-8-sig")


def test_tokenizer_learns_deterministic_vocab_from_fit_texts_only():
    tokenizer = SequenceTokenizer(max_vocab_size=6).fit([
        "đường đèn_đường",
        "đường ngập",
    ])
    assert tokenizer.word_index["<PAD>"] == 0
    assert tokenizer.word_index["<OOV>"] == 1
    assert tokenizer.word_index["đường"] == 2
    assert tokenizer.texts_to_sequences(["đường từ_mới"]) == [[2, 1]]
    assert "từ_mới" not in tokenizer.word_index


def test_padding_supports_pre_post_padding_and_truncation():
    sequences = [[4, 5, 6], [7]]
    post = pad_sequences(sequences, 4, padding="post", truncating="post")
    pre = pad_sequences(sequences, 4, padding="pre", truncating="pre")
    assert post.tolist() == [[4, 5, 6, 0], [7, 0, 0, 0]]
    assert pre.tolist() == [[0, 4, 5, 6], [0, 0, 0, 7]]
    assert pad_sequences([[1, 2, 3, 4]], 2, truncating="pre").tolist() == [[3, 4]]


def test_label_encoder_uses_stable_canonical_taxonomy():
    encoder = LabelEncoderBundle().fit_taxonomy()
    encoded = encoder.transform(["R11", "R01", "R08"])
    assert encoded.tolist() == [10, 0, 7]
    assert encoder.inverse_transform(encoded).tolist() == ["R11", "R01", "R08"]
    with pytest.raises(ValueError, match="Unknown label code"):
        encoder.transform(["R99"])


def test_build_embedding_data_creates_sequences_and_label_artifacts(tmp_path):
    dataset = tmp_path / "processed"
    dataset.mkdir()
    _write_split(dataset / "train.csv", [
        ("đường ổ_gà hỏng", "R01"),
        ("đèn_đường không_sáng", "R02"),
        ("cống ngập nước", "R03"),
    ])
    _write_split(dataset / "validation.csv", [
        ("đường từ_lạ", "R04"),
    ])
    _write_split(dataset / "test.csv", [
        ("rác_thải khu_vực", "R09"),
    ])
    output = tmp_path / "embedding"

    result = build_embedding_data(EmbeddingConfig(
        dataset_dir=str(dataset),
        output_dir=str(output),
        max_vocab_size=20,
        max_length=5,
        embedding_dim=64,
    ))

    assert result.train_rows == 3
    assert result.validation_rows == 1
    assert result.test_rows == 1
    assert result.class_count == 11
    assert result.embedding_dim == 64

    x_train = np.load(output / "X_train.npy")
    x_validation = np.load(output / "X_validation.npy")
    x_test = np.load(output / "X_test.npy")
    y_train = np.load(output / "y_train.npy")
    y_validation = np.load(output / "y_validation.npy")
    y_test = np.load(output / "y_test.npy")
    assert x_train.shape == (3, 5)
    assert x_validation.shape == (1, 5)
    assert x_test.shape == (1, 5)
    assert y_train.tolist() == [0, 1, 2]
    assert y_validation.tolist() == [3]
    assert y_test.tolist() == [8]

    tokenizer, label_encoder = load_embedding_assets(output)
    assert tokenizer.vocab_size == result.vocabulary_size
    assert label_encoder.classes_.tolist() == [f"R{i:02d}" for i in range(1, 12)]
    assert x_validation[0, -1] == 0
    assert 1 in x_validation[0]  # token not seen in train is OOV

    for name in (
        "tokenizer.json", "tokenizer.joblib", "label_encoder.joblib",
        "vocabulary.csv", "class_mapping.csv", "embedding_config.json",
        "embedding_summary.json", "labels_train.csv", "labels_validation.csv",
        "labels_test.csv",
    ):
        assert (output / name).exists()


def test_build_embedding_data_rejects_empty_training_corpus(tmp_path):
    dataset = tmp_path / "processed"
    dataset.mkdir()
    _write_split(dataset / "train.csv", [("", "R01")])
    _write_split(dataset / "validation.csv", [])
    _write_split(dataset / "test.csv", [])

    with pytest.raises(ValueError, match="All values"):
        build_embedding_data(EmbeddingConfig(
            dataset_dir=str(dataset),
            output_dir=str(tmp_path / "embedding"),
        ))


def test_create_embedding_layer_when_tensorflow_is_installed():
    pytest.importorskip("tensorflow")
    layer = create_embedding_layer(vocab_size=20, embedding_dim=32, mask_zero=True)
    assert layer.input_dim == 20
    assert layer.output_dim == 32
    assert layer.mask_zero is True
