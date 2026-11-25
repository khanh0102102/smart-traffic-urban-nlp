"""Run single or batch inference with the selected ST-NLP model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from src.inference import ModelPredictor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify citizen reports using the best packaged ST-NLP model."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="Input text; may be repeated.")
    source.add_argument("--input", help="Input CSV for batch prediction.")
    parser.add_argument("--text-column", default="content")
    parser.add_argument("--bundle", default="ml/artifacts/inference/best_model_bundle.joblib")
    parser.add_argument("--already-processed", action="store_true")
    parser.add_argument("--output", help="Optional CSV output path for batch prediction.")
    args = parser.parse_args()

    predictor = ModelPredictor(args.bundle)
    if args.text is not None:
        values = predictor.predict(args.text, already_processed=args.already_processed)
        print(json.dumps(values, ensure_ascii=False, indent=2))
        return 0

    source_path = Path(args.input)
    if not source_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source_path}")
    frame = pd.read_csv(source_path, encoding="utf-8-sig", keep_default_na=False)
    if args.text_column not in frame.columns:
        raise ValueError(
            f"CSV does not contain text column {args.text_column!r}. "
            f"Available columns: {list(frame.columns)}"
        )
    predictions = predictor.predict(
        frame[args.text_column].tolist(),
        already_processed=args.already_processed,
    )
    result = pd.concat([frame.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(result)} predictions to {output_path}")
    else:
        print(result.to_json(orient="records", force_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
