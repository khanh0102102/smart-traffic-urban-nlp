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

## Phases 4–11 – Data processing, Vietnamese NLP, TF-IDF, Embedding, Naive Bayes, SVM, CNN and LSTM
Install the base ML dependencies and prepare the dataset:

```powershell
python -m pip install -r ml/requirements.txt
python ml/scripts/prepare_dataset.py --input ml/data/sample_reports.csv --text-column content --label-column category
```

After Phases 4–5, build TF-IDF features for Naive Bayes/SVM with `python ml/scripts/build_tfidf.py --dataset-dir ml/data/processed --output ml/artifacts/tfidf`. Build Phase 7 integer token sequences with `python ml/scripts/build_embedding.py --dataset-dir ml/data/processed --output ml/artifacts/embedding`. The tokenizer is fitted on train only and the R01–R11 label map remains fixed. Train the Naive Bayes and SVM models using the scripts documented in [`ml/README.md`](ml/README.md).

For the neural models, install the optional TensorFlow dependency. Train the CNN and LSTM from the Phase 7 sequence artifacts:

```powershell
python -m pip install -r ml/requirements-embedding.txt
python ml/scripts/train_cnn.py --embedding-dir ml/artifacts/embedding --output ml/artifacts/cnn
python ml/scripts/train_lstm.py --embedding-dir ml/artifacts/embedding --output ml/artifacts/lstm
python ml/scripts/predict_lstm.py --text "Đèn đường bị hỏng"
```

CNN uses parallel Conv1D branches; LSTM reads token sequences recurrently and explicitly masks PAD token ID 0. Both produce 11-class softmax outputs. Model artifacts remain local and are excluded from Git. See [`ml/README.md`](ml/README.md) for detailed configuration, evaluation and prediction commands.

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
