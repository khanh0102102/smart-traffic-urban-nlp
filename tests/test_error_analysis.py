import pandas as pd

from stnlp.training.error_analysis import save_error_analysis


def test_error_analysis_exports_misclassified_rows_and_pairs(tmp_path):
    errors_path, pairs_path = save_error_analysis(
        tmp_path,
        "svm",
        ["Câu phản ánh một", "Câu phản ánh hai", "Câu đúng"],
        ["R01", "R02", "R03"],
        ["R02", "R02", "R03"],
    )
    errors = pd.read_csv(errors_path)
    pairs = pd.read_csv(pairs_path)

    assert len(errors) == 1
    assert errors.iloc[0]["true_label"] == "R01"
    assert errors.iloc[0]["predicted_label"] == "R02"
    assert len(pairs) == 1
    assert pairs.iloc[0]["count"] == 1
