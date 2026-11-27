#!/bin/sh
set -eu

cd /app

echo "[ST-NLP] Applying database migrations..."
alembic upgrade head

echo "[ST-NLP] Ensuring the 11 incident categories exist..."
python scripts/seed_categories.py

mkdir -p "${REPORT_UPLOAD_DIR:-/app/backend/storage}"

echo "[ST-NLP] Starting API..."
exec "$@"
