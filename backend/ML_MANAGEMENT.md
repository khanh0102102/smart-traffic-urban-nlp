# Phase 19 – ML Management API

All endpoints in this module are under `/api/v1/ml` and require a Bearer JWT for an active staff user with role `ADMIN` or `RECEIVER`. Citizen accounts receive HTTP 403. The API manages the existing `datasets`, `dataset_samples`, `training_runs` and `training_metrics` tables; it adds no table or migration.

## 1. Dataset management

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/datasets` | Create dataset version |
| GET | `/datasets` | Search and paginate datasets |
| GET | `/datasets/{dataset_id}` | Dataset details and sample counts |
| PATCH | `/datasets/{dataset_id}` | Update metadata |
| DELETE | `/datasets/{dataset_id}` | Delete dataset when no training run references it |

Dataset `name + version` must be unique. The creator is assigned from the authenticated staff user; clients cannot impersonate another creator. The detail response includes total samples, TRAIN/VALIDATION/TEST counts and verified sample count.

Example:

```http
POST /api/v1/ml/datasets
Authorization: Bearer <staff_access_token>
Content-Type: application/json

{
  "name": "Urban incident reports",
  "version": "v1",
  "description": "Reviewed Vietnamese urban incident dataset",
  "source": "staff-reviewed-reports"
}
```

## 2. Dataset sample management

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/datasets/{dataset_id}/samples` | Add 1–500 samples atomically |
| GET | `/datasets/{dataset_id}/samples` | Filter and paginate samples |
| PATCH | `/dataset-samples/{sample_id}` | Correct text, label, split or verification flag |
| DELETE | `/dataset-samples/{sample_id}` | Delete one sample |

Each sample has `raw_text`, optional `processed_text`, `category_code`, `split` (TRAIN, VALIDATION or TEST), `is_verified` and an optional linked `report_id`. Category codes are normalized and must refer to active incident categories R01–R11. If a report ID is supplied, the report must exist. Bulk insertion is all-or-nothing.

Example:

```http
POST /api/v1/ml/datasets/<dataset_id>/samples
Authorization: Bearer <staff_access_token>
Content-Type: application/json

{
  "items": [
    {
      "raw_text": "Mặt đường có ổ gà sâu.",
      "processed_text": "mặt đường ổ gà sâu",
      "category_code": "R01",
      "split": "TRAIN",
      "is_verified": true
    },
    {
      "raw_text": "Đèn đường không hoạt động.",
      "category_code": "R02",
      "split": "VALIDATION"
    }
  ]
}
```

Sample listing supports `split`, `category_code`, `is_verified`, `page` and `page_size` filters.

## 3. Training run management

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/training-runs` | Register a pending training run |
| GET | `/training-runs` | Filter and paginate training runs |
| GET | `/training-runs/{run_id}` | Run details including metrics |
| PATCH | `/training-runs/{run_id}/status` | Advance training lifecycle |

A run may use algorithms `NAIVE_BAYES`, `SVM`, `CNN`, `LSTM`, `OTHER`; representations `TF_IDF`, `WORD_EMBEDDING`, `OTHER`. Creation requires at least one **verified** sample in the dataset's TRAIN split. A report can be enrolled directly from the staff review endpoint by passing both `dataset_id` and `dataset_split`; this creates or refreshes one verified sample per report and dataset.

Allowed lifecycle:

- `PENDING → RUNNING`
- `PENDING → FAILED`
- `RUNNING → COMPLETED`
- `RUNNING → FAILED`

Starting a run assigns `started_at`; completing or failing it assigns `finished_at`. Terminal runs cannot be restarted or changed to another status. Optional `model_version_id` must reference an existing model version; `artifact_path` stores the training artifact location.

Example:

```http
POST /api/v1/ml/training-runs
Authorization: Bearer <staff_access_token>
Content-Type: application/json

{
  "dataset_id": "<dataset_id>",
  "algorithm": "SVM",
  "representation": "TF_IDF",
  "hyperparameters": {"C": 1.0, "kernel": "linear"},
  "imbalance_method": "class_weight"
}
```

**Execution boundary:** this API registers and records training jobs. It does not itself execute Python training scripts or start background workers. The ML pipeline/worker must perform training and call the status and metric endpoints with actual results.

## 4. Training metrics

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/training-runs/{run_id}/metrics` | Save a batch of metrics |
| GET | `/training-runs/{run_id}/metrics` | Retrieve metrics |

Metrics can be added when the run is RUNNING or COMPLETED. A batch accepts up to 200 unique `(metric_name, class_label)` pairs. Values are decimal numbers from 0 to 1, matching the existing database constraint. `class_label` is `ALL` or one of R01–R11. Duplicate metric keys return HTTP 409.

Example:

```http
POST /api/v1/ml/training-runs/<run_id>/metrics
Authorization: Bearer <staff_access_token>
Content-Type: application/json

{
  "items": [
    {"metric_name": "accuracy", "metric_value": 0.93, "class_label": "ALL"},
    {"metric_name": "macro_f1", "metric_value": 0.87, "class_label": "ALL"},
    {"metric_name": "precision", "metric_value": 0.91, "class_label": "R01"}
  ]
}
```

## 5. Integrity and database notes

- The API uses the four existing ML management tables and the existing `incident_categories`, `reports`, `model_versions` and `users` tables.
- No schema change, new table or migration is required.
- Dataset deletion is blocked with HTTP 409 after a training run references it, preserving the existing foreign-key constraint.
- Dataset/sample/run management is restricted to staff; citizens cannot access training data or model management.
- A reviewed report may be synchronized into a selected dataset from `POST /api/v1/reviews` with `dataset_id` and `dataset_split` supplied together. The sample uses the report's original text, the preprocessing output used by the prediction (when available), and the staff-confirmed category.
- Only verified TRAIN examples qualify a dataset for a new training run. If a `model_version_id` is attached to a run, its algorithm and representation must match the run.
- Training metrics remain in the range supported by the current database check constraint (0–1); this endpoint is for scalar scores such as accuracy, precision, recall and F1, not raw confusion-matrix counts.
