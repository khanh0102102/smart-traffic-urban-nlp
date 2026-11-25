"""Create padded token sequences and encoded labels from Phase 5 dataset splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from src.embedding import EmbeddingConfig, build_embedding_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit a train-only vocabulary, pad sequences and encode R01-R11 labels."
    )
    parser.add_argument("--dataset-dir", default="ml/data/processed")
    parser.add_argument("--output", default="ml/artifacts/embedding")
    parser.add_argument("--text-column", default="processed_text")
    parser.add_argument("--label-column", default="label_code")
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--padding", choices=("pre", "post"), default="post")
    parser.add_argument("--truncating", choices=("pre", "post"), default="post")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--no-mask-zero", action="store_true")
    args = parser.parse_args()

    config = EmbeddingConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output,
        text_column=args.text_column,
        label_column=args.label_column,
        max_vocab_size=args.max_vocab_size,
        min_frequency=args.min_frequency,
        max_length=args.max_length,
        padding=args.padding,
        truncating=args.truncating,
        embedding_dim=args.embedding_dim,
        lowercase=args.lowercase,
        mask_zero=not args.no_mask_zero,
    )
    result = build_embedding_data(config)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
