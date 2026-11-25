# Prediction Review API – Phase 17

## Purpose

Phase 17 lets authorized staff confirm an AI prediction or correct its category. Every decision is appended to the existing `prediction_reviews` table. No new table or database migration is required.

The original `predictions.predicted_category_id` is never changed. The latest staff-confirmed category is written to `reports.final_category_id`, while all older review decisions remain available as an audit trail.

## Permissions

- `ADMIN` and `RECEIVER`: may create review decisions and inspect review history for any report.
- `CITIZEN`: may not create or change a review; may read review decisions only for their own reports.
- The reviewer identity is always taken from the authenticated Bearer JWT. A caller cannot provide `reviewer_user_id`.

## Endpoints

All endpoints are under `/api/v1` and require `Authorization: Bearer <access_token>`.

| Method | Endpoint | Permission | Purpose |
|---|---|---|---|
| POST | `/reviews` | ADMIN, RECEIVER | Confirm or correct a prediction label |
| GET | `/reviews/{review_id}` | Report owner or staff | Retrieve one review decision |
| GET | `/reviews/prediction/{prediction_id}` | Report owner or staff | List review decisions for a prediction |
| GET | `/reviews/report/{report_id}` | Report owner or staff | List all review decisions for a report |

### Confirm or correct a label

```http
POST /api/v1/reviews
Authorization: Bearer <staff_access_token>
Content-Type: application/json
```

```json
{
  "prediction_id": "UUID-OF-A-PREDICTION",
  "confirmed_category_code": "R02",
  "notes": "Đối chiếu nội dung và hình ảnh, phân loại lại sang nhóm điện chiếu sáng."
}
```

The API verifies that the prediction exists and that the category code is active in the R01–R11 taxonomy. It derives the action rather than trusting a client-supplied action:

- `CONFIRMED`: the selected category is the same as the AI prediction.
- `CORRECTED`: the selected category differs from the AI prediction.

The API then inserts `prediction_reviews` with the reviewer, confirmed category, action and optional notes, and updates the report's current `final_category_id` in the same database transaction.

### Review history

`GET /api/v1/reviews/prediction/{prediction_id}` returns decisions for one prediction. `GET /api/v1/reviews/report/{report_id}` returns decisions across all predictions for a report. Both are ordered newest first.

Each response includes:
- AI-predicted category and staff-confirmed category;
- reviewer ID, display name and current role;
- action (`CONFIRMED` or `CORRECTED`);
- optional notes and creation timestamp;
- report ID and tracking code.

## Database

Phase 17 reuses the original `prediction_reviews`, `predictions`, `reports`, `incident_categories` and `users` tables. The database remains at 12 tables. The operation does not overwrite the original AI prediction or delete previous review records.

## Tests

```powershell
python -m pytest backend/tests/test_reviews.py -q
python -m pytest -q
```

The SQLite tests cover staff confirmation, label correction, append-only history, final-category updates, owner/staff permissions, inactive/unknown categories and authentication.
