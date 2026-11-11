"""Add keyword-based label suggestions to a CSV for human review."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stnlp.data.label_suggestions import suggest_label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gợi ý nhãn theo từ khóa. Kết quả luôn yêu cầu kiểm duyệt thủ công."
    )
    parser.add_argument("--input", required=True, help="CSV có cột text.")
    parser.add_argument("--output", required=True, help="CSV kết quả cần annotator rà soát.")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source, encoding="utf-8")
    if "text" not in frame.columns:
        raise ValueError("CSV đầu vào phải có cột text.")

    suggestions = frame["text"].fillna("").astype(str).map(suggest_label)
    result = pd.concat([frame, pd.json_normalize(suggestions)], axis=1)
    result["review_status"] = "pending"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Đã tạo {len(result)} gợi ý tại {output.resolve()}")
    print("Lưu ý: không dùng suggested_label thay cho label đã được người gán nhãn xác nhận.")


if __name__ == "__main__":
    main()
