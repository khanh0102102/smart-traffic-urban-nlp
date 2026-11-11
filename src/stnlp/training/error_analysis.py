"""Export misclassified examples and frequent confusion pairs for manual review."""

from pathlib import Path

import pandas as pd

from stnlp.labels import LABEL_TO_NAME


def save_error_analysis(
    output_dir: Path,
    model_name: str,
    texts: list[str],
    y_true: list[str],
    y_pred: list[str],
) -> tuple[Path, Path]:
    """Write row-level errors and aggregated true/predicted class confusion pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "text": texts,
        "true_label": y_true,
        "predicted_label": y_pred,
    })
    errors = frame[frame["true_label"] != frame["predicted_label"]].copy()
    errors["true_label_name"] = errors["true_label"].map(LABEL_TO_NAME)
    errors["predicted_label_name"] = errors["predicted_label"].map(LABEL_TO_NAME)
    errors_path = output_dir / f"{model_name}_misclassified_examples.csv"
    errors.to_csv(errors_path, index=False, encoding="utf-8-sig")

    if errors.empty:
        pairs = pd.DataFrame(
            columns=["true_label", "true_label_name", "predicted_label",
                     "predicted_label_name", "count"]
        )
    else:
        pairs = (
            errors.groupby(["true_label", "predicted_label"])
            .size()
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
        )
        pairs["true_label_name"] = pairs["true_label"].map(LABEL_TO_NAME)
        pairs["predicted_label_name"] = pairs["predicted_label"].map(LABEL_TO_NAME)
        pairs = pairs[
            ["true_label", "true_label_name", "predicted_label",
             "predicted_label_name", "count"]
        ]
    pairs_path = output_dir / f"{model_name}_confusion_pairs.csv"
    pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    return errors_path, pairs_path
