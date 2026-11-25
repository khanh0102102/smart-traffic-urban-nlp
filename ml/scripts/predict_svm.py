"""Predict citizen reports with a saved SVM + TF-IDF bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from src.svm import predict_svm_reports


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict incident categories using a saved LinearSVC model."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="Raw report text; repeat for multiple reports.")
    source.add_argument("--input", help="CSV file containing reports.")
    parser.add_argument("--text-column", default="content")
    parser.add_argument("--output", default="predictions_svm.csv")
    parser.add_argument("--model", default="ml/artifacts/svm/svm_bundle.joblib")
    parser.add_argument(
        "--already-processed",
        action="store_true",
        help="Input CSV text is Phase 5 processed_text; skip NLP preprocessing.",
    )
    args = parser.parse_args()

    if args.text is not None:
        predictions = predict_svm_reports(args.text, model_path=args.model)
        print(json.dumps(predictions, ensure_ascii=False, indent=2))
        return 0

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    frame = pd.read_csv(input_path, encoding="utf-8-sig", keep_default_na=False)
    if args.text_column not in frame.columns:
        raise ValueError(
            f"Column {args.text_column!r} not found in {input_path.name}. "
            f"Available columns: {list(frame.columns)}"
        )
    predictions = predict_svm_reports(
        frame[args.text_column].tolist(),
        model_path=args.model,
        already_processed=args.already_processed,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps({
        "input_rows": len(frame),
        "output_file": str(output_path),
        "predictions": len(predictions),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
