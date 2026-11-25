# Machine Learning workspace

## Phase 4 – Data processing

The pipeline reads CSV, TSV, Excel, JSON, JSONL and TXT files; removes invalid rows and exact duplicates; maps known labels to the canonical 11-category taxonomy; optionally creates weak keyword labels; and creates deterministic train/validation/test CSV files. Vietnamese text preparation is described in Phase 5 below.

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

## Phase 6 – TF-IDF feature extraction

The module in `ml/src/tfidf/` converts the Phase 5 `processed_text` field into sparse numerical feature matrices using scikit-learn `TfidfVectorizer`.

### Build TF-IDF matrices

Run Phase 4–5 first so that `ml/data/processed/train.csv`, `validation.csv` and `test.csv` exist. Then run from the repository root:

```powershell
python ml/scripts/build_tfidf.py --dataset-dir ml/data/processed --output ml/artifacts/tfidf
```

Default settings use unigram + bigram features (`ngram_range=(1, 2)`), sublinear term frequency, IDF weighting, L2 normalization, and a maximum vocabulary size of 20,000 terms. Vietnamese diacritics are preserved. Since Phase 5 already performs Vietnamese word segmentation, TF-IDF splits the prepared text on whitespace and keeps compound segmented tokens such as `đèn_đường` intact.

### Important data-splitting rule

**The vectorizer is fitted only on `train.csv`.** The same fitted vocabulary and IDF weights are applied to `validation.csv` and `test.csv` using `transform()`. Validation/test words that are absent from the training vocabulary are ignored. This prevents information from held-out splits leaking into feature construction.

### Useful options

```powershell
python ml/scripts/build_tfidf.py --dataset-dir ml/data/processed --output ml/artifacts/tfidf --max-features 10000 --min-df 2 --ngram-min 1 --ngram-max 2
```

Available options: `--text-column`, `--label-column`, `--max-features`, `--min-df`, `--max-df`, `--ngram-min`, `--ngram-max`, `--no-sublinear-tf`, and `--no-idf`.

### Generated artifacts

- `tfidf_vectorizer.joblib`: fitted vectorizer for later inference.
- `X_train.npz`, `X_validation.npz`, `X_test.npz`: compressed sparse TF-IDF matrices.
- `y_train.csv`, `y_validation.csv`, `y_test.csv`: labels aligned to matrix rows by `row_index`.
- `vocabulary.csv`: feature index, token/ngram and learned IDF weight.
- `tfidf_summary.json`: configuration, library version, matrix dimensions and sparsity statistics.

Artifacts are written under `ml/artifacts/tfidf/` and remain local because the `ml/artifacts/*` folder is excluded from Git.

### Run tests

```powershell
python -m pytest ml/tests/test_tfidf.py -q
```
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
