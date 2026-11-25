"""Evaluate held-out predictions from the four ST-NLP models."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from src.evaluation import EvaluationConfig, evaluate_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Naive Bayes, SVM, CNN and LSTM from saved held-out predictions.")
    parser.add_argument("--naive-bayes-dir", default="ml/artifacts/naive_bayes")
    parser.add_argument("--svm-dir", default="ml/artifacts/svm")
    parser.add_argument("--cnn-dir", default="ml/artifacts/cnn")
    parser.add_argument("--lstm-dir", default="ml/artifacts/lstm")
    parser.add_argument("--output", default="ml/artifacts/evaluation")
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), default=["validation", "test"])
    parser.add_argument("--require-all-models", action="store_true")
    args = parser.parse_args()
    result = evaluate_models(EvaluationConfig(
        model_dirs={"naive_bayes": args.naive_bayes_dir, "svm": args.svm_dir,
                    "cnn": args.cnn_dir, "lstm": args.lstm_dir},
        output_dir=args.output, splits=tuple(args.splits),
        require_all_models=args.require_all_models))
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
