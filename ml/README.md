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
## Phase 7 – Tokenizer, Padding, Embedding and Label Encoder

The module in `ml/src/embedding/` converts Phase 5 `processed_text` into fixed-length integer sequences and converts canonical incident labels R01–R11 into stable numeric IDs.

### Build token sequences and label IDs

Run Phase 4–5 first, then run from the repository root:

```powershell
python ml/scripts/build_embedding.py --dataset-dir ml/data/processed --output ml/artifacts/embedding
```

Default settings are a maximum vocabulary of 30,000 entries (including PAD/OOV), minimum token frequency 1, sequence length 200, post-padding, post-truncation, and embedding dimension 128.

### Data leakage and token IDs

- The tokenizer vocabulary is fitted on `train.csv` only. Validation/test words not seen in train map to the OOV token ID `1`.
- Padding uses token ID `0`; the Keras Embedding layer can mask it with `mask_zero=True`.
- The vocabulary is deterministic: tokens are ordered by descending frequency and then alphabetically for ties.
- The label encoder uses the fixed canonical Phase 3 taxonomy R01–R11, not labels discovered from validation/test. This keeps numeric class IDs stable even when a category is absent from a small training split.

### Use a trainable Keras Embedding layer

The sequence and label data pipeline does not require TensorFlow. TensorFlow is an optional dependency only when a downstream neural model needs an actual Keras layer:

```powershell
python -m pip install -r ml/requirements-embedding.txt
```

Example in a future Keras model:

```python
from src.embedding import create_embedding_layer

embedding = create_embedding_layer(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=128,
    mask_zero=True,
)
```

The layer starts with trainable random weights. Those weights are learned when the downstream model is trained; this phase prepares the layer configuration and integer inputs, not a pretrained embedding model.
On native Windows, the optional TensorFlow package runs on CPU; TensorFlow GPU support for newer releases requires WSL2. See the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip) before choosing a GPU setup.


### Generated artifacts

- `X_train.npy`, `X_validation.npy`, `X_test.npy`: padded integer token sequences.
- `y_train.npy`, `y_validation.npy`, `y_test.npy`: numeric class IDs aligned with the corresponding X rows.
- `tokenizer.joblib`, `tokenizer.json`: fitted train-only vocabulary and token ID mapping.
- `label_encoder.joblib`, `class_mapping.csv`: stable mapping between R01–R11 and class IDs.
- `labels_train.csv`, `labels_validation.csv`, `labels_test.csv`: row index, label code, numeric ID and category name.
- `vocabulary.csv`: token and token ID; `embedding_config.json`: Keras Embedding layer settings.
- `embedding_summary.json`: dataset sizes, sequence shapes, vocabulary and class mapping.

Artifacts are written under `ml/artifacts/embedding/` and remain local because `ml/artifacts/*` is excluded from Git.

### Useful options

```powershell
python ml/scripts/build_embedding.py --dataset-dir ml/data/processed --output ml/artifacts/embedding --max-vocab-size 15000 --min-frequency 2 --max-length 150 --embedding-dim 100
```

Additional options include `--padding pre|post`, `--truncating pre|post`, `--lowercase`, and `--no-mask-zero`.

### Run tests

```powershell
python -m pytest ml/tests/test_embedding.py -q
```
## Phase 8 – Naive Bayes classification

The module in `ml/src/naive_bayes/` trains a `MultinomialNB` classifier from Phase 6 TF-IDF matrices. It saves a single inference bundle containing the trained model, the exact TF-IDF vectorizer, and the fixed R01–R11 label encoder.

### Train and evaluate

Run Phase 4–6 first, then train from the repository root:

```powershell
python ml/scripts/train_naive_bayes.py --tfidf-dir ml/artifacts/tfidf --output ml/artifacts/naive_bayes
```

The default additive smoothing parameter is `alpha=1.0`, and class priors are learned from train. The model is fitted only on `X_train.npz` and `y_train.csv`. Validation and test matrices are evaluated separately; neither is used to fit the model.

### Predict a new citizen report

Use raw Vietnamese text. The CLI applies the saved Phase 5 preprocessing pipeline before TF-IDF transformation:

```powershell
python ml/scripts/predict_naive_bayes.py --text "Đèn đường trước nhà bị hỏng nhiều ngày, buổi tối không có ánh sáng"
```

Predict several rows from a CSV file (default source column `content`):

```powershell
python ml/scripts/predict_naive_bayes.py --input reports_to_classify.csv --text-column content --output predictions.csv
```

If the CSV already contains Phase 5 `processed_text`, skip preprocessing explicitly:

```powershell
python ml/scripts/predict_naive_bayes.py --input ml/data/processed/test.csv --text-column processed_text --already-processed --output ml/artifacts/naive_bayes/test_predictions.csv
```

Each prediction contains the category code/name, maximum class probability (`confidence`), processed text, and `has_known_features`. The probability is a model score, not a guarantee or necessarily a calibrated confidence. If no input token is present in the training TF-IDF vocabulary, the model still returns a prior-based prediction but sets `has_known_features=false` and adds a warning; treat that prediction as unreliable.

### Evaluation and saved artifacts

- `naive_bayes_bundle.joblib`: trained MultinomialNB + fitted TF-IDF vectorizer + canonical label encoder + preprocessing metadata.
- `predictions_validation.csv`, `predictions_test.csv`: actual/predicted code and name, confidence and correctness per row.
- `confusion_matrix_validation.csv`, `confusion_matrix_test.csv`: fixed R01–R11 confusion matrices.
- `top_terms_by_class.csv`: terms with the largest class-conditional log probabilities.
- `naive_bayes_summary.json`: hyperparameters, trained classes, feature count, validation/test accuracy and precision/recall/F1 reports.

The artifacts are written under `ml/artifacts/naive_bayes/`; they are local and excluded from Git.

**Data limitation:** the included sample has very few reports per category and is only a pipeline demonstration. A useful production classifier requires a larger, manually verified, balanced dataset. A category absent from the training split cannot be predicted by that fitted model.

### Tune smoothing

```powershell
python ml/scripts/train_naive_bayes.py --tfidf-dir ml/artifacts/tfidf --output ml/artifacts/naive_bayes --alpha 0.5 --top-terms 30
```

Use validation metrics to compare settings. Keep the test split as a final holdout and avoid repeatedly tuning against it.

### Run tests

```powershell
python -m pytest ml/tests/test_naive_bayes.py -q
```
## Phase 9 – SVM classification and probability calibration

The module in `ml/src/svm/` uses `LinearSVC` on Phase 6 sparse TF-IDF matrices. It optionally wraps the linear SVM with `CalibratedClassifierCV` (sigmoid by default) so inference can return calibrated class probabilities.

### Train and evaluate

Run Phase 4–6 first, then train from the repository root:

```powershell
python ml/scripts/train_svm.py --tfidf-dir ml/artifacts/tfidf --output ml/artifacts/svm
```

Default settings: `C=1.0`, `class_weight=balanced`, `max_iter=10000`, sigmoid probability calibration and up to 3 stratified folds. The estimator and calibration folds use only `X_train.npz` and `y_train.csv`; validation/test are held out for evaluation.

### Probability calibration and small datasets

- If every training class has at least two examples, the module calibrates SVM scores using stratified cross-validation on train only. The number of folds is lowered when needed to fit the smallest class count.
- If any training class has only one example, SVM training continues without calibration. In that case, prediction outputs `confidence=null`, `probability_calibrated=false`, and a decision margin; a margin is not a probability.
- Isotonic calibration is accepted only when every class has at least 1,000 training examples and each cross-validation calibration fold has roughly 1,000 rows or more. For smaller datasets, use sigmoid calibration.
- Validation/test are never used to fit the SVM or the probability calibrator.

### Predict a new report

Raw Vietnamese text is passed through the same Phase 5 preprocessing before applying the saved TF-IDF vectorizer:

```powershell
python ml/scripts/predict_svm.py --text "Đèn đường trước nhà bị hỏng nhiều ngày, buổi tối không có ánh sáng"
```

Predict a CSV file:

```powershell
python ml/scripts/predict_svm.py --input reports_to_classify.csv --text-column content --output svm_predictions.csv
```

For an input column that already contains Phase 5 `processed_text`, pass `--already-processed`.

Prediction results include the predicted R01–R11 code/name, `confidence` (only when calibrated), `class_probabilities`, `decision_margin`, `probability_calibrated`, and `has_known_features`. An all-OOV report receives a warning because the model is then relying on its intercepts rather than learned TF-IDF terms.

### Tune the SVM

```powershell
python ml/scripts/train_svm.py --tfidf-dir ml/artifacts/tfidf --output ml/artifacts/svm --C 0.5 --class-weight balanced --calibration-method sigmoid --calibration-cv 3
```

Try `--class-weight none` to disable class balancing, `--no-calibration` to train without probability calibration, and `--top-features 30` to export more influential positive features per class. Compare settings on validation metrics; keep test as a final holdout.

### Generated artifacts

- `svm_bundle.joblib`: fitted SVM, exact TF-IDF vectorizer, label encoder and calibration metadata.
- `predictions_validation.csv`, `predictions_test.csv`: prediction and evaluation details.
- `confusion_matrix_validation.csv`, `confusion_matrix_test.csv`: fixed R01–R11 confusion matrices.
- `top_features_by_class.csv`: highest positive linear coefficients per trained category.
- `svm_summary.json`: hyperparameters, calibration status, trained classes and metrics.

Artifacts are written under `ml/artifacts/svm/` and remain local because `ml/artifacts/*` is excluded from Git.

**Data limitation:** the included sample dataset is only a pipeline demonstration. Reliable SVM performance and probability calibration require a larger, manually verified dataset with enough examples in each class.

### Run tests

```powershell
python -m pytest ml/tests/test_svm.py -q
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


## Phase 10 – 1D CNN text classification

Phase 10 adds a trainable 1D TextCNN using the padded integer sequences from Phase 7. It does not use the TF-IDF matrices from Phase 6. The network has a trainable Embedding layer, parallel Conv1D branches (default kernel sizes 3, 4 and 5), padding-aware global max pooling, a Dense/Dropout classifier and an 11-unit softmax output for R01–R11.

### Install the optional neural-network dependency

TensorFlow is deliberately kept separate from the base ML requirements so that Phases 4–9 do not need to install it:

```powershell
python -m pip install -r ml/requirements-embedding.txt
```

### Build Phase 7 sequences and train the CNN

Run the data preparation and Phase 5 preprocessing first. Then build the sequence artifacts and train:

```powershell
python ml/scripts/build_embedding.py --dataset-dir ml/data/processed --output ml/artifacts/embedding
python ml/scripts/train_cnn.py --embedding-dir ml/artifacts/embedding --output ml/artifacts/cnn
```

Default configuration: 20 epochs, batch size 32, 128 filters per convolution branch, kernel sizes 3/4/5, 128-dimensional Dense layer, 0.2 embedding spatial dropout, 0.4 classifier dropout, Adam learning rate 0.001 and balanced class weights. Early stopping monitors validation loss when the validation split is non-empty and restores the best weights. When validation is empty, training proceeds without validation-based early stopping.

### Data separation and padding

- The CNN reuses the tokenizer and fixed R01–R11 label encoder saved by Phase 7; it does not refit vocabulary or label IDs.
- Model weights and class weights are learned from the training split only.
- Validation is used only for early stopping and validation metrics. The test split is reserved for final evaluation.
- A custom layer zeros the PAD embedding vectors before convolution, and masked global max pooling excludes token ID 0 positions. Token ID 1 remains the OOV token.
- The reported maximum softmax score is a model confidence score, not a separately calibrated probability.

### Predict a new report

Raw Vietnamese text is passed through the saved Phase 5 preprocessing flow:

```powershell
python ml/scripts/predict_cnn.py --text "Đèn đường trước nhà bị hỏng nhiều ngày, buổi tối không có ánh sáng"
```

Predict a CSV file (default input column `content`):

```powershell
python ml/scripts/predict_cnn.py --input reports_to_classify.csv --text-column content --output cnn_predictions.csv
```

When the input CSV already contains Phase 5 `processed_text`, use `--already-processed` to skip preprocessing.

### Generated artifacts

- `cnn_model.keras`: trained Keras 1D CNN model.
- `cnn_bundle.joblib`: fitted Phase 7 tokenizer, label encoder, model filename and inference metadata.
- `training_history.csv`: per-epoch loss and accuracy values.
- `predictions_validation.csv`, `predictions_test.csv`: actual/predicted category, confidence, known-token indicator and correctness.
- `confusion_matrix_validation.csv`, `confusion_matrix_test.csv`: fixed R01–R11 confusion matrices.
- `cnn_summary.json`: architecture, training parameters, split sizes, epoch count and evaluation metrics.

Artifacts are written under `ml/artifacts/cnn/` and remain local because `ml/artifacts/*` is excluded from Git.

### Tune the CNN

```powershell
python ml/scripts/train_cnn.py --embedding-dir ml/artifacts/embedding --output ml/artifacts/cnn --epochs 30 --batch-size 32 --filters 96 --kernel-sizes 3 4 5 --dropout 0.5 --learning-rate 0.0005 --patience 5
```

Use validation metrics to compare settings. Keep the test split as a final holdout and do not repeatedly tune against it.

**Data limitation:** the included synthetic sample is for checking the pipeline, not for training a useful deep-learning classifier. CNNs need a substantially larger, manually reviewed dataset with representative examples for each incident category. If a category is absent from train, its output neuron receives no positive training examples and the model cannot learn that category reliably.

### Run tests

```powershell
python -m pytest ml/tests/test_cnn.py -q
```

The CNN tests are skipped automatically when TensorFlow is not installed.
