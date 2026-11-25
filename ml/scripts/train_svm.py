"""Train a LinearSVC model and optionally calibrate class probabilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from src.svm import SVMConfig, train_svm


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train LinearSVC from Phase 6 TF-IDF matrices and evaluate held-out data."
    )
    parser.add_argument("--tfidf-dir", default="ml/artifacts/tfidf")
    parser.add_argument("--output", default="ml/artifacts/svm")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--class-weight", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--calibration-method", choices=("sigmoid", "isotonic"), default="sigmoid")
    parser.add_argument("--calibration-cv", type=int, default=3)
    parser.add_argument("--top-features", type=int, default=20)
    args = parser.parse_args()

    result = train_svm(SVMConfig(
        tfidf_dir=args.tfidf_dir,
        output_dir=args.output,
        C=args.C,
        class_weight=None if args.class_weight == "none" else "balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
        calibrate_probabilities=not args.no_calibration,
        calibration_method=args.calibration_method,
        calibration_cv=args.calibration_cv,
        top_features=args.top_features,
    ))
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
