# Machine Learning workspace

## Phase 4 – Data processing

The pipeline reads CSV, TSV, Excel, JSON, JSONL and TXT files; cleans Vietnamese text; normalizes Unicode; removes exact duplicates; maps known labels to the canonical 11-category taxonomy; optionally creates weak keyword labels; tokenizes Vietnamese with Underthesea; and creates deterministic train/validation/test CSV files.

### Phase 5 – Vietnamese NLP preprocessing

The preprocessing module is reusable and integrated into the Phase 4 dataset pipeline. Its stages are:

1. **Clean text:** handles HTML entities/tags, URLs, emails, control characters and repeated whitespace while retaining the original source value in raw_content.
2. **Unicode NFC:** normalizes Vietnamese characters to NFC without removing diacritics.
3. **Abbreviation normalization:** expands common chat shorthand and administrative address forms (ko → không, đg. → đường, Q.1 → quận 1) using ml/data/abbreviations_vi.json.
4. **Word segmentation:** Underthesea creates Vietnamese word tokens, including compound tokens joined by underscores.
5. **Stopword removal:** a conservative list in ml/data/stopwords_vi.txt filters grammatical filler words. Negations and incident-relevant terms such as không, chưa, bị, được and hỏng are protected.

The exports preserve separate representations: cleaned_text, standardized_text, normalized_text, segmented_text and processed_text. Deduplication uses abbreviation-aware normalized text. Stopword removal only affects processed_text; it does not overwrite the source, cleaned or segmented text.

Inspect one sample text and see each stage:
```powershell
python ml/scripts/inspect_nlp.py --text "Đg. này ko thông thoáng tại Q.1"
```

To use custom dictionaries:
```powershell
python ml/scripts/inspect_nlp.py --text "Đg. bị hỏng" --abbreviations path/to/abbreviations.json --stopwords path/to/stopwords.txt
```
## Install
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

The dataset CLI also accepts `--abbreviations path/to/file.json` and `--stopwords path/to/file.txt`. Use `--keep-stopwords` to keep all segmented tokens in `processed_text` while still performing word segmentation.

### Outputs
- `train.csv`, `validation.csv`, `test.csv`: labeled records split 70/15/15.
- `labeled_all.csv`: all deduplicated labeled rows before splitting.
- `needs_review.csv`: rows without a reliable label; not included in model splits.
- `duplicates.csv`: repeated rows detected after text normalization.
- `rejected_rows.csv`: empty or too-short rows.
- `label_distribution.csv`, `dataset_summary.json`: class counts and run statistics.

### Data quality rules
- Original text is retained in `raw_content`; cleaned, standardized, normalized, segmented and stopword-filtered representations are stored separately.
- Vietnamese accents are preserved. Unicode is normalized to NFC.
- Exact duplicates are removed before splitting to prevent train/test leakage. Similar but non-identical reports are not automatically removed.
- If identical normalized text appears with conflicting source labels, the retained row is marked `duplicate_label_conflict` and sent to manual review.
- Existing labels are accepted only when they map to R01–R11. Unrecognized labels are sent for review.
- Keyword labels are weak supervision, not verified ground truth. Review them before model training or evaluation.
- At least three labeled unique rows are required to generate non-empty train/validation/test sets.
- Keep real citizen data private; raw input files and generated processed data are ignored by Git.
