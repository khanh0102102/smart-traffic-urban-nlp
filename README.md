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

## Phases 4–10 – Data processing, Vietnamese NLP preprocessing, TF-IDF, Embedding, Naive Bayes, SVM and 1D CNN
Install the base ML dependencies and prepare the sample dataset:

```powershell
python -m pip install -r ml/requirements.txt
python ml/scripts/prepare_dataset.py --input ml/data/sample_reports.csv --text-column content --label-column category
```

Run Phase 5 preprocessing through the dataset pipeline, then build TF-IDF matrices with `python ml/scripts/build_tfidf.py --dataset-dir ml/data/processed --output ml/artifacts/tfidf`. Build the Phase 7 padded token sequences with `python ml/scripts/build_embedding.py --dataset-dir ml/data/processed --output ml/artifacts/embedding`. The vocabulary is fitted on train only, and the R01–R11 label mapping remains fixed. Train the Naive Bayes and SVM baselines with `python ml/scripts/train_naive_bayes.py --tfidf-dir ml/artifacts/tfidf --output ml/artifacts/naive_bayes` and `python ml/scripts/train_svm.py --tfidf-dir ml/artifacts/tfidf --output ml/artifacts/svm`.

For the neural model, install the optional TensorFlow dependency and train the 1D CNN from Phase 7 sequences:

```powershell
python -m pip install -r ml/requirements-embedding.txt
python ml/scripts/train_cnn.py --embedding-dir ml/artifacts/embedding --output ml/artifacts/cnn
python ml/scripts/predict_cnn.py --text "Đèn đường bị hỏng"
```

The CNN uses parallel Conv1D kernels, padding-aware max pooling and an 11-class softmax layer. The CNN model and all generated ML artifacts remain local and are excluded from Git. See [`ml/README.md`](ml/README.md) for the commands, options and data-splitting details.

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
