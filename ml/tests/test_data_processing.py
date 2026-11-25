from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_processing.cleaning import clean_text, normalize_text
from src.data_processing.categories import canonical_label, keyword_label
from src.data_processing.pipeline import ProcessingConfig, run_pipeline

def test_cleaning_preserves_vietnamese_and_removes_markup():
    assert clean_text("  <b>ĐƯỜNG</b>   hỏng &amp; nguy hiểm https://example.com ") == "ĐƯỜNG hỏng & nguy hiểm"
    assert normalize_text("ĐƯỜNG, HỎNG!") == "đường hỏng"

def test_category_mapping_accepts_codes_and_names():
    assert canonical_label("R03")[0] == "R03"
    assert canonical_label("Cống/Hố ga/Thoát nước")[0] == "R03"
    assert canonical_label("unknown")[0] is None

def test_keyword_label_only_when_unambiguous():
    assert keyword_label("Đèn đường bị hỏng")[0] == "R02"
    assert keyword_label("Phản ánh cần xử lý")[0] is None

def test_pipeline_deduplicates_and_exports_review_and_splits(tmp_path):
    source=tmp_path/"reports.csv"
    source.write_text('content,category\n"Đường có ổ gà nguy hiểm",R01\n"Đường có ổ gà nguy hiểm!",R01\n"Đèn đường bị hỏng",R02\n"Rác thải gây mùi hôi",R09\n"Phản ánh chưa rõ nhóm",\n',encoding="utf-8")
    output=tmp_path/"out"
    result=run_pipeline(ProcessingConfig(input_path=str(source),output_dir=str(output),use_keyword_rules=False))
    assert result.input_rows == 5
    assert result.valid_rows == 4
    assert result.duplicate_rows == 1
    assert result.review_rows == 1
    assert (output/"train.csv").exists()
    assert (output/"validation.csv").exists()
    assert (output/"test.csv").exists()
    assert (output/"needs_review.csv").exists()
    assert (output/"duplicates.csv").exists()
    assert (output/"dataset_summary.json").exists()
