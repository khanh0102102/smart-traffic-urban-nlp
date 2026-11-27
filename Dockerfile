# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime libraries for NumPy, scikit-learn and TensorFlow CPU inference.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 libstdc++6 libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement manifests separately so source changes do not invalidate the dependency layer.
COPY backend/requirements.txt /tmp/backend-requirements.txt
COPY ml/requirements.txt /tmp/ml/requirements.txt
COPY ml/requirements-embedding.txt /tmp/ml/requirements-embedding.txt

# Install API, Vietnamese NLP, classical ML and optional CNN/LSTM inference dependencies.
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
        -r /tmp/backend-requirements.txt \
        -r /tmp/ml/requirements-embedding.txt

COPY . /app

COPY docker/entrypoint.sh /usr/local/bin/stnlp-entrypoint
RUN chmod 0755 /usr/local/bin/stnlp-entrypoint \
    && mkdir -p /app/backend/storage /app/ml/artifacts /app/ml/data/raw \
        /app/ml/data/interim /app/ml/data/processed

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/stnlp-entrypoint"]
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
