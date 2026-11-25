# Machine Learning workspace

## Phase 4 – Data processing

The pipeline reads CSV, TSV, Excel, JSON, JSONL and TXT files; cleans Vietnamese text; normalizes Unicode; removes exact duplicates; maps known labels to the canonical 11-category taxonomy; optionally creates weak keyword labels; tokenizes Vietnamese with Underthesea; and creates deterministic train/validation/test CSV files.

### Install
Run from the repository root:
```powershell
python -m pip install -r ml/requirements.txt
```

### Prepare the included sample
```powershell
python ml/scripts/prepare_dataset.py --input ml/data/sample_reports.csv --text-column content --label-column category
```

### Process your own data
```powershell
python ml/scripts/prepare_dataset.py --input path/to/reports.xlsx --text-column noi_dung --label-column nhom_phan_anh --output ml/data/processed
```
If your file has no label column, omit `--label-column`; unique, unambiguous keyword hits become weak labels and all other rows go to manual review. To disable keyword labeling use `--no-keyword-rules`.

### Outputs
- `train.csv`, `validation.csv`, `test.csv`: labeled records split 70/15/15.
- `labeled_all.csv`: all deduplicated labeled rows before splitting.
- `needs_review.csv`: rows without a reliable label; not included in model splits.
- `duplicates.csv`: repeated rows detected after text normalization.
- `rejected_rows.csv`: empty or too-short rows.
- `label_distribution.csv`, `dataset_summary.json`: class counts and run statistics.

### Data quality rules
- Original text is retained in `raw_content`; cleaned/normalized/tokenized text is stored separately.
- Vietnamese accents are preserved. Unicode is normalized to NFC.
- Exact duplicates are removed before splitting to prevent train/test leakage. Similar but non-identical reports are not automatically removed.
- Existing labels are accepted only when they map to R01–R11. Unrecognized labels are sent for review.
- Keyword labels are weak supervision, not verified ground truth. Review them before model training or evaluation.
- At least three labeled unique rows are required to generate non-empty train/validation/test sets.
- Keep real citizen data private; raw input files and generated processed data are ignored by Git.
