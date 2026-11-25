"""Train a Multinomial Naive Bayes classifier from Phase 6 TF-IDF artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from src.naive_bayes import NaiveBayesConfig, train_naive_bayes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit MultinomialNB on X_train only and evaluate validation/test."
    )
    parser.add_argument("--tfidf-dir", default="ml/artifacts/tfidf")
    parser.add_argument("--output", default="ml/artifacts/naive_bayes")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--no-fit-prior", action="store_true")
    parser.add_argument("--top-terms", type=int, default=20)
    args = parser.parse_args()

    result = train_naive_bayes(NaiveBayesConfig(
        tfidf_dir=args.tfidf_dir,
        output_dir=args.output,
        alpha=args.alpha,
        fit_prior=not args.no_fit_prior,
        top_terms=args.top_terms,
    ))
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
