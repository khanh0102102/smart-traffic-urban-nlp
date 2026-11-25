"""Phase 10 1D CNN text classification."""
from .model import (
    CNNConfig,
    CNNResult,
    build_cnn_model,
    train_cnn,
    predict_cnn_reports,
)

__all__ = [
    "CNNConfig",
    "CNNResult",
    "build_cnn_model",
    "train_cnn",
    "predict_cnn_reports",
]
