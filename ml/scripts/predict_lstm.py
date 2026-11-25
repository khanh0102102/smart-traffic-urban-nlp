"""Predict reports using a trained Phase 11 LSTM bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from src.lstm import predict_lstm_reports


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict R01-R11 incident categories using the saved LSTM."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="Raw Vietnamese report; repeat for multiple.")
    source.add_argument("--input", help="CSV file containing reports.")
    parser.add_argument("--text-column", default="content")
    parser.add_argument("--output", default="predictions_lstm.csv")
    parser.add_argument("--model", default="ml/artifacts/lstm/lstm_bundle.joblib")
    parser.add_argument(
        "--already-processed",
        action="store_true",
        help="Input CSV text is Phase 5 processed_text; skip NLP preprocessing.",
    )
    args = parser.parse_args()

    if args.text is not None:
        predictions = predict_lstm_reports(args.text, model_path=args.model)
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
    predictions = predict_lstm_reports(
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
