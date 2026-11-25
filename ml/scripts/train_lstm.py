"""Train and evaluate the Phase 11 LSTM model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from src.lstm import LSTMConfig, train_lstm


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train an LSTM classifier from Phase 7 padded token sequences."
    )
    parser.add_argument("--embedding-dir", default="ml/artifacts/embedding")
    parser.add_argument("--output", default="ml/artifacts/lstm")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lstm-units", type=int, default=128)
    parser.add_argument("--dense-units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--embedding-dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--class-weight", choices=("balanced", "none"), default="balanced")
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use a Bidirectional(LSTM) encoder instead of a one-way LSTM.",
    )
    args = parser.parse_args()

    result = train_lstm(LSTMConfig(
        embedding_dir=args.embedding_dir,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        embedding_dropout=args.embedding_dropout,
        learning_rate=args.learning_rate,
        random_state=args.seed,
        early_stopping_patience=args.patience,
        class_weight=None if args.class_weight == "none" else "balanced",
        bidirectional=args.bidirectional,
    ))
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
