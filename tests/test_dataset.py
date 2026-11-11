import pandas as pd
import pytest

from stnlp.labels import LABEL_CODES
from stnlp.training.data import load_dataset, stratified_split


def test_load_and_stratified_split_all_classes(tmp_path):
    rows = [
        {"text": f"Phản ánh {code} ví dụ số {index}", "label": code}
        for code in LABEL_CODES
        for index in range(8)
    ]
    path = tmp_path / "reports.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    frame = load_dataset(path)
    train, validation, test = stratified_split(frame)
    assert len(frame) == 88
    assert set(train.label) == set(LABEL_CODES)
    assert set(validation.label) == set(LABEL_CODES)
    assert set(test.label) == set(LABEL_CODES)


def test_conflicting_duplicate_text_is_rejected(tmp_path):
    rows = [
        {"text": f"Phản ánh {code} ví dụ số {index}", "label": code}
        for code in LABEL_CODES
        for index in range(7)
    ]
    rows.append({"text": rows[0]["text"], "label": "R02"})
    path = tmp_path / "conflict.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    with pytest.raises(ValueError, match="nhãn khác nhau"):
        load_dataset(path)
