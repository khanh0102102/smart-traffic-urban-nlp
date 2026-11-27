# Smart Traffic & Urban Incident Classifier (ST-NLP)

ST-NLP is a Vietnamese NLP system for receiving citizen reports about traffic and urban infrastructure, suggesting one of 11 incident categories with machine learning, and supporting staff review.

## Docker-first / Docker-only runtime

**You do not need to install Python, pip, Node.js, PostgreSQL or any backend libraries on the host.** Docker builds the Python runtime and installs the API, Vietnamese NLP, scikit-learn and TensorFlow dependencies inside the API image. PostgreSQL runs in its own container, and FastAPI serves both web portals and the API.

### Requirements

- Git
- Docker Desktop with Docker Compose v2
- A .env file copied from .env.example

### First run (Windows PowerShell)

```powershell
git clone https://github.com/khanh0102102/smart-traffic-urban-nlp.git
cd smart-traffic-urban-nlp
Copy-Item .env.example .env
docker compose up -d --build
```

### First run (macOS / Linux)

```bash
git clone https://github.com/khanh0102102/smart-traffic-urban-nlp.git
cd smart-traffic-urban-nlp
cp .env.example .env
docker compose up -d --build
```

On the first run Docker downloads the base images, builds the API image and installs the Python dependencies inside that image. The API container then waits for PostgreSQL, applies Alembic migrations, inserts any missing R01–R11 categories, and starts Uvicorn. You do not run pip, Alembic or the seed script on the host.

### Open the application

- Citizen portal: http://localhost:8000/citizen/
- Admin portal: http://localhost:8000/admin/
- API documentation: http://localhost:8000/docs
- API health: http://localhost:8000/api/v1/health
- Database health: http://localhost:8000/api/v1/health/database

### Create the first staff account

After the containers are healthy, create an administrator from inside the API container. This does not require Python on the host:

```powershell
docker compose exec api python -m app.core.staff_provision --email admin@example.com --full-name "System Admin" --role ADMIN
```

Enter the password when prompted. Repeat this only when you need another staff account. Staff accounts are not created automatically with a shared default password.

### Optional Google sign-in

Set GOOGLE_CLIENT_ID in .env to your Google OAuth Web Client ID, then recreate the API container:

```powershell
docker compose up -d --force-recreate api
```

The Citizen portal reads this setting from the API at request time; do not edit the HTML file. In Google Cloud Console, add http://localhost:8000 and http://127.0.0.1:8000 as authorized JavaScript origins.

### AI model artifacts

The trained inference bundle is not committed to Git because generated model files can be large and are specific to the training run. Place the selected model bundle at:

```text
ml/artifacts/inference/best_model_bundle.joblib
```

The ml/artifacts folder is mounted into the API container, so adding or replacing the bundle does not require rebuilding the image. CNN/LSTM inference dependencies are already included in the image. Without a valid trained bundle, report submission still works, but AI prediction responds with the existing truthful “model unavailable” fallback.

Training and ML scripts can also be invoked inside the container, for example:

```powershell
docker compose run --rm api python ml/scripts/prepare_dataset.py --input ml/data/sample_reports.csv --text-column content --label-column category
```

See ml/README.md for the model preparation, training, evaluation and packaging workflow. Generated data and artifacts are persisted in the mounted ml/data and ml/artifacts folders.

## Daily commands

```powershell
# Start or rebuild the whole application
docker compose up -d --build

# View service status
docker compose ps

# Follow API logs (migrations and seed messages are shown here)
docker compose logs -f api

# Follow PostgreSQL logs
docker compose logs -f postgres

# Restart API after changing environment values
docker compose up -d --force-recreate api

# Stop containers but preserve database and uploaded images
docker compose down
```

To update the source code later:

```powershell
git pull origin main
docker compose up -d --build
```

**Data safety:** docker compose down preserves named volumes. docker compose down -v deletes PostgreSQL data and uploaded report images; use it only when you intentionally want a full reset.

## Docker services and persistence

| Service / volume | Responsibility |
|---|---|
| api | FastAPI, both portals, authentication, NLP and ML inference |
| postgres | PostgreSQL 17 database |
| stnlp_postgres_data | Persistent PostgreSQL data |
| stnlp_report_uploads | Uploaded citizen report images |
| ./ml/artifacts | Local trained models and generated artifacts |
| ./ml/data/{raw,interim,processed} | Dataset workflow files |

The database is available to the API on the internal Docker network as postgres:5432. It is not published to the host by default.

## Tests

The automated CI pipeline runs JavaScript syntax/client tests, Python unit/API/integration tests, the Playwright browser journey with PostgreSQL, and a Docker Compose build/smoke test.

For local test execution without installing Python on the host, use the container:

```powershell
docker compose run --rm api python -m pip install -r backend/requirements-dev.txt
docker compose run --rm api python -m pytest -q
```

The test command above installs development-only test packages into a disposable container; it does not alter the runtime image. The normal application start does not install development dependencies.

## Security notes

- The committed .env.example secret is for local development only. Replace JWT_SECRET_KEY, database passwords and OAuth settings before exposing the application outside your machine.
- Never commit .env, real citizen data, uploaded images or trained model artifacts.
- Staff accounts are provisioned explicitly; there is no public staff registration.
