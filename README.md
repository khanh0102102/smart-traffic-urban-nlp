# Smart Traffic & Urban Incident Classifier (ST-NLP)

ST-NLP is a Vietnamese NLP system for classifying citizen reports about traffic and urban infrastructure.

## Project workspaces
| Folder | Responsibility |
|---|---|
| `backend/` | FastAPI API, SQLAlchemy models, PostgreSQL and application services |
| `frontend/` | Separate citizen and admin portals |
| `ml/` | Dataset preparation, NLP preprocessing, training and evaluation |
| `database/` | PostgreSQL schema, Alembic migrations and seed data |
| `tests/` | Automated backend and integration tests |

## Initialize local database (PowerShell)
```powershell
Copy-Item .env.example .env
docker compose up -d postgres
python -m pip install -r backend/requirements-dev.txt
alembic upgrade head
python scripts/seed_categories.py
```
Wait until `stnlp-postgres` is healthy before running migrations. The default database is `stnlp` on port `5432`.

## Phases 4–8 – Data processing, Vietnamese NLP preprocessing, TF-IDF, Embedding and Naive Bayes
Install the ML dependencies and run the included sample. The pipeline cleans source text, applies Unicode NFC and abbreviation normalization, segments Vietnamese words, removes conservative stopwords, deduplicates, labels and splits the dataset:

```powershell
python -m pip install -r ml/requirements.txt
python ml/scripts/prepare_dataset.py --input ml/data/sample_reports.csv --text-column content --label-column category
```

Inspect the NLP stages independently with `python ml/scripts/inspect_nlp.py --text "Đg. này ko thông thoáng tại Q.1"`. After Phase 4–5 completes, build TF-IDF matrices with `python ml/scripts/build_tfidf.py --dataset-dir ml/data/processed --output ml/artifacts/tfidf`. The vectorizer is fitted on train only and transforms validation/test with the same vocabulary. Next, create padded sequences and label IDs with `python ml/scripts/build_embedding.py --dataset-dir ml/data/processed --output ml/artifacts/embedding`. The tokenizer is fitted on train only; the canonical R01–R11 label mapping remains fixed. Train the Naive Bayes baseline with `python ml/scripts/train_naive_bayes.py --tfidf-dir ml/artifacts/tfidf --output ml/artifacts/naive_bayes`, then predict raw reports with `python ml/scripts/predict_naive_bayes.py --text "Đèn đường bị hỏng"`. See [`ml/README.md`](ml/README.md) for details. Processed data and model artifacts remain local and are excluded from Git.

## Run backend
```powershell
uvicorn app.main:app --app-dir backend --reload
```
Open http://127.0.0.1:8000/docs. Health checks are at `/api/v1/health` and `/api/v1/health/database`; the citizen and admin portals remain separate at `/citizen/` and `/admin/`.

## Tests
```powershell
python -m pytest -q
```
Never commit secrets, private citizen data, uploaded images, raw datasets or generated model artifacts.
