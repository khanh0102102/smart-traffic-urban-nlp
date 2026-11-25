"""Build the packaged best-model inference artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from src.inference import build_inference_bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select the best classifier by validation Macro F1 and package it for inference."
    )
    parser.add_argument("--evaluation-dir", default="ml/artifacts/evaluation")
    parser.add_argument("--naive-bayes-dir", default="ml/artifacts/naive_bayes")
    parser.add_argument("--svm-dir", default="ml/artifacts/svm")
    parser.add_argument("--cnn-dir", default="ml/artifacts/cnn")
    parser.add_argument("--lstm-dir", default="ml/artifacts/lstm")
    parser.add_argument("--output", default="ml/artifacts/inference")
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    args = parser.parse_args()

    result = build_inference_bundle(
        evaluation_dir=args.evaluation_dir,
        model_dirs={
            "naive_bayes": args.naive_bayes_dir,
            "svm": args.svm_dir,
            "cnn": args.cnn_dir,
            "lstm": args.lstm_dir,
        },
        output_dir=args.output,
        confidence_threshold=args.confidence_threshold,
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
