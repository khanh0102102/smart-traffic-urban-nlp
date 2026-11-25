"""Build TF-IDF matrices from the Phase 5 train/validation/test CSV files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from src.tfidf import TfidfConfig, build_tfidf


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit TF-IDF on train.csv and transform validation.csv/test.csv."
    )
    parser.add_argument("--dataset-dir", default="ml/data/processed")
    parser.add_argument("--output", default="ml/artifacts/tfidf")
    parser.add_argument("--text-column", default="processed_text")
    parser.add_argument("--label-column", default="label_code")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=1.0)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--no-sublinear-tf", action="store_true")
    args = parser.parse_args()

    config = TfidfConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output,
        text_column=args.text_column,
        label_column=args.label_column,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        sublinear_tf=not args.no_sublinear_tf,
    )
    result = build_tfidf(config)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
