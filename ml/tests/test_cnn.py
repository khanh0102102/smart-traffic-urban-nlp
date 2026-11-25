"""Tests for Phase 10 CNN model creation, training and prediction."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

tf = pytest.importorskip("tensorflow")

from src.cnn import CNNConfig, build_cnn_model, predict_cnn_reports, train_cnn
from src.embedding import SequenceTokenizer, LabelEncoderBundle


def _prepare_embedding_artifacts(directory: Path) -> None:
    directory.mkdir(parents=True)
    train_texts = [
        "duong o_ga hu_hong",
        "duong mat_duong nut",
        "duong sat lun",
        "den_duong chieu_sang",
        "den_tin_hieu giao_thong",
        "den_bi_hong",
    ]
    tokenizer = SequenceTokenizer(max_vocab_size=100).fit(train_texts)
    label_encoder = LabelEncoderBundle().fit_taxonomy().encoder
    joblib.dump(tokenizer, directory / "tokenizer.joblib")
    joblib.dump(label_encoder, directory / "label_encoder.joblib")
    (directory / "embedding_config.json").write_text(
        '{"input_dim": 20, "output_dim": 12, "sequence_length": 8, "mask_zero": true}',
        encoding="utf-8",
    )

    splits = {
        "train": (train_texts, ["R01", "R01", "R01", "R02", "R02", "R02"]),
        "validation": (["duong o_ga", "den_duong"], ["R01", "R02"]),
        "test": (["duong nut", "den_tin_hieu"], ["R01", "R02"]),
    }
    from src.embedding import pad_sequences
    for split_name, (texts, codes) in splits.items():
        sequences = tokenizer.texts_to_sequences(texts)
        x = pad_sequences(sequences, 8)
        y = label_encoder.transform(codes).astype(np.int64)
        np.save(directory / f"X_{split_name}.npy", x)
        np.save(directory / f"y_{split_name}.npy", y)


def test_cnn_architecture_has_eleven_outputs_and_padding_aware_pooling():
    model = build_cnn_model(
        vocab_size=30,
        sequence_length=12,
        class_count=11,
        embedding_dim=8,
        filters=6,
        kernel_sizes=(3, 5),
        dense_units=10,
        dropout=0.1,
        embedding_dropout=0.1,
    )
    assert model.output_shape == (None, 11)
    assert model.name == "stnlp_text_cnn_1d"
    assert "masked_global_max_3" in [layer.name for layer in model.layers]
    sample = np.zeros((2, 12), dtype=np.int32)
    sample[0, :2] = [2, 3]
    sample[1, :3] = [4, 5, 6]
    probabilities = model.predict(sample, verbose=0)
    assert probabilities.shape == (2, 11)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_cnn_trains_saves_evaluates_and_predicts_processed_text(tmp_path):
    embedding_dir = tmp_path / "embedding"
    _prepare_embedding_artifacts(embedding_dir)
    output = tmp_path / "cnn"

    result = train_cnn(CNNConfig(
        embedding_dir=str(embedding_dir),
        output_dir=str(output),
        epochs=2,
        batch_size=2,
        filters=6,
        kernel_sizes=(3, 4),
        dense_units=8,
        dropout=0.1,
        embedding_dropout=0.1,
        early_stopping_patience=1,
    ))
    assert result.train_rows == 6
    assert result.validation_rows == 2
    assert result.test_rows == 2
    assert result.trained_class_count == 2
    assert result.epochs_trained >= 1
    assert Path(result.model_path).exists()
    assert Path(result.bundle_path).exists()

    for filename in (
        "cnn_model.keras",
        "cnn_bundle.joblib",
        "training_history.csv",
        "cnn_summary.json",
        "predictions_validation.csv",
        "predictions_test.csv",
        "confusion_matrix_validation.csv",
        "confusion_matrix_test.csv",
    ):
        assert (output / filename).exists()

    predictions = predict_cnn_reports(
        ["duong o_ga", "tu_khoa_khong_co_trong_train"],
        model_path=output / "cnn_bundle.joblib",
        already_processed=True,
    )
    assert len(predictions) == 2
    assert predictions[0]["predicted_label_code"] in {"R01", "R02"}
    assert 0 <= predictions[0]["confidence"] <= 1
    assert sum(predictions[0]["class_probabilities"].values()) == pytest.approx(1.0)
    assert predictions[0]["has_known_tokens"] is True
    assert predictions[1]["has_known_tokens"] is False
    assert predictions[1]["warning"] is not None


def test_cnn_rejects_sequence_label_row_mismatch(tmp_path):
    embedding_dir = tmp_path / "embedding"
    _prepare_embedding_artifacts(embedding_dir)
    np.save(embedding_dir / "y_train.npy", np.asarray([0], dtype=np.int64))
    with pytest.raises(ValueError, match="row mismatch"):
        train_cnn(CNNConfig(
            embedding_dir=str(embedding_dir),
            output_dir=str(tmp_path / "cnn"),
            epochs=1,
        ))


def test_cnn_validates_hyperparameters():
    with pytest.raises(ValueError, match="epochs"):
        train_cnn(CNNConfig(epochs=0))
    with pytest.raises(ValueError, match="learning_rate"):
        train_cnn(CNNConfig(learning_rate=0))


def test_cnn_requires_embedding_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="tokenizer.joblib"):
        train_cnn(CNNConfig(
            embedding_dir=str(tmp_path / "missing"),
            output_dir=str(tmp_path / "cnn"),
        ))
