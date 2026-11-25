"""Inspect every NLP preprocessing stage for a single Vietnamese report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from src.nlp_preprocessing import preprocess_vietnamese_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Show cleaning, abbreviation normalization, segmentation and stopword filtering.")
    parser.add_argument("--text", required=True, help="Vietnamese report text to inspect.")
    parser.add_argument("--abbreviations", default=None, help="Optional JSON abbreviation dictionary.")
    parser.add_argument("--stopwords", default=None, help="Optional stopword file (one token per line).")
    parser.add_argument("--keep-stopwords", action="store_true", help="Keep all segmented tokens.")
    args = parser.parse_args()
    result = preprocess_vietnamese_text(
        args.text,
        remove_stopwords_enabled=not args.keep_stopwords,
        abbreviations_path=args.abbreviations,
        stopwords_path=args.stopwords,
    )
    print(json.dumps({
        "raw_cleaned_text": result.raw_cleaned_text,
        "standardized_text": result.standardized_text,
        "normalized_text": result.normalized_text,
        "segmented_text": result.segmented_text,
        "tokens": list(result.tokens),
        "processed_text": result.processed_text,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
