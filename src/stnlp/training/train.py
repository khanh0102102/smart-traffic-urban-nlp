"""CLI to train and compare Naive Bayes, SVM, CNN and LSTM."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

from stnlp.labels import LABEL_CODES, LABEL_TO_NAME
from stnlp.training.classical import build_classical_models, fit_classical, save_classical
from stnlp.training.data import load_dataset, stratified_split
from stnlp.training.deep_learning import train_deep_model
from stnlp.training.metrics import calculate_metrics, save_evaluation
from stnlp.training.error_analysis import save_error_analysis
from stnlp.preprocessing.text import preprocess_text


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def _evaluate_deep(model, tokenizer, max_len, texts, labels):
    clean_texts = [preprocess_text(text) for text in texts]
    sequences = tokenizer.texts_to_sequences(clean_texts)
    x_values = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    probabilities = model.predict(x_values, verbose=0)
    predicted_indices = probabilities.argmax(axis=1)
    predictions = [LABEL_CODES[int(index)] for index in predicted_indices]
    return predictions, probabilities


def run_training(args) -> dict:
    _set_seed(args.random_state)
    frame = load_dataset(args.data)
    train_df, validation_df, test_df = stratified_split(frame, args.random_state)
    output_root = Path(args.artifacts_dir)
    evaluation_dir = output_root / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    x_val, y_val = validation_df["text"].tolist(), validation_df["label"].tolist()
    x_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()

    label_distribution = frame["label"].value_counts().reindex(LABEL_CODES, fill_value=0)
    pd.DataFrame({
        "label": list(LABEL_CODES),
        "label_name": [LABEL_TO_NAME[code] for code in LABEL_CODES],
        "count": label_distribution.astype(int).tolist(),
    }).to_csv(evaluation_dir / "dataset_label_distribution.csv", index=False, encoding="utf-8-sig")

    summary = {
        "dataset": str(args.data),
        "dataset_rows_after_exact_deduplication": int(len(frame)),
        "removed_exact_duplicates": int(frame.attrs.get("removed_exact_duplicates", 0)),
        "split_sizes": {
            "train": len(train_df), "validation": len(validation_df), "test": len(test_df)
        },
        "random_state": args.random_state,
        "label_order": list(LABEL_CODES),
        "models": {},
    }
    validation_scores = {}

    # Classical baselines: both use the same raw split and same preprocessing implementation.
    for model_name, pipeline in build_classical_models(args.random_state).items():
        fit_classical(model_name, pipeline, x_train, y_train)
        val_predictions = pipeline.predict(x_val)
        val_metrics = calculate_metrics(y_val, val_predictions)
        validation_scores[model_name] = val_metrics["macro_f1"]
        save_classical(pipeline, output_root / model_name)
        test_predictions = pipeline.predict(x_test)
        test_metrics = save_evaluation(evaluation_dir, model_name, y_test, test_predictions)
        save_error_analysis(evaluation_dir, model_name, x_test, y_test, test_predictions)
        summary["models"][model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "artifact": f"{model_name}/model.joblib",
        }

    # Neural models: tokenizer is fitted on training texts only; validation is for early stopping.
    for model_name in ("cnn", "lstm"):
        model, tokenizer, max_len = train_deep_model(
            model_name,
            x_train,
            y_train,
            x_val,
            y_val,
            output_root / model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        val_predictions, _ = _evaluate_deep(model, tokenizer, max_len, x_val, y_val)
        test_predictions, _ = _evaluate_deep(model, tokenizer, max_len, x_test, y_test)
        val_metrics = calculate_metrics(y_val, val_predictions)
        test_metrics = save_evaluation(evaluation_dir, model_name, y_test, test_predictions)
        save_error_analysis(evaluation_dir, model_name, x_test, y_test, test_predictions)
        validation_scores[model_name] = val_metrics["macro_f1"]
        summary["models"][model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "artifact": f"{model_name}/model.keras",
        }

    selected_model = max(validation_scores, key=validation_scores.get)
    summary["selection"] = {
        "selected_by_validation_macro_f1": selected_model,
        "validation_macro_f1": validation_scores[selected_model],
        "note": "Test metrics are for final reporting; selection is based on validation only.",
    }
    (evaluation_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    comparison = []
    for name, result in summary["models"].items():
        comparison.append({"model": name, **result["test"]})
    pd.DataFrame(comparison).to_csv(evaluation_dir / "comparison.csv", index=False)
    print(json.dumps(summary["selection"], ensure_ascii=False, indent=2))
    print(f"Artifacts và báo cáo đã lưu tại: {output_root.resolve()}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train ST-NLP classifiers.")
    parser.add_argument("--data", required=True, help="CSV có các cột text,label.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
