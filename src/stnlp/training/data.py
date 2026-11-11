"""Dataset loading, validation and stratified split helpers."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from stnlp.labels import LABEL_CODES, normalize_label


def _dedupe_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a UTF-8 CSV with text,label columns.

    Exact duplicate texts are removed only when their labels agree. Conflicting
    duplicate labels are reported instead of silently selecting a ground truth.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {path}")
    frame = pd.read_csv(path, encoding="utf-8")
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset thiếu cột bắt buộc: {sorted(missing)}")

    frame = frame[["text", "label"]].copy()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame = frame[frame["text"] != ""]
    if frame.empty:
        raise ValueError("Dataset không có dòng văn bản hợp lệ.")

    try:
        frame["label"] = frame["label"].astype(str).map(normalize_label)
    except ValueError as exc:
        raise ValueError(f"Dataset có nhãn không hợp lệ: {exc}") from exc

    frame["_key"] = frame["text"].map(_dedupe_key)
    conflicts = frame.groupby("_key")["label"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(
            f"Có {len(conflicts)} nội dung trùng nhưng nhãn khác nhau. "
            "Cần rà soát và adjudicate trước khi huấn luyện."
        )

    duplicate_count = int(frame.duplicated("_key").sum())
    frame = frame.drop_duplicates("_key").drop(columns="_key").reset_index(drop=True)
    present = set(frame["label"])
    missing_labels = sorted(set(LABEL_CODES) - present)
    if missing_labels:
        raise ValueError(f"Dataset chưa có đủ 11 lớp; đang thiếu: {missing_labels}")

    counts = frame["label"].value_counts()
    too_small = counts[counts < 7]
    if not too_small.empty:
        raise ValueError(
            "Mỗi lớp cần tối thiểu 7 mẫu để thử chia Train/Validation/Test phân tầng. "
            f"Lớp thiếu mẫu: {too_small.to_dict()}"
        )

    frame.attrs["removed_exact_duplicates"] = duplicate_count
    return frame


def stratified_split(frame: pd.DataFrame, random_state: int = 42):
    """Return train, validation and test frames using approximately 70/15/15."""
    train_val, test = train_test_split(
        frame,
        test_size=0.15,
        random_state=random_state,
        stratify=frame["label"],
    )
    validation_fraction = 0.15 / 0.85
    train, validation = train_test_split(
        train_val,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=train_val["label"],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
