# Phase 18 – Dashboard API

The dashboard endpoints are intended for the separate staff/admin portal. Both endpoints require a valid Bearer JWT and a current database role of `ADMIN` or `RECEIVER`. Citizens receive HTTP 403.

## 1. Summary statistics

`GET /api/v1/dashboard/summary`

Optional query parameters:

- `date_from=YYYY-MM-DD`: inclusive start date, interpreted as UTC.
- `date_to=YYYY-MM-DD`: inclusive end date, interpreted as UTC.

Example:

```http
GET /api/v1/dashboard/summary?date_from=2026-11-01&date_to=2026-11-30
Authorization: Bearer <access_token>
```

The response contains:
- `total_reports`, `categorized_reports`, `unclassified_reports`, and `geolocated_reports`.
- `pending_reports` (RECEIVED + PROCESSING), `resolved_reports`, and `rejected_reports`.
- `categories`: one entry for every configured incident category, including zero-count categories, with count, percentage, active flag and description.
- `statuses`: counts for RECEIVED, PROCESSING, RESOLVED and REJECTED.

### Category-counting rule

Each report is assigned to at most one dashboard category. The API uses `reports.final_category_id` when a staff member has confirmed/corrected the label. Otherwise it uses the most recent row in `predictions`. A report with neither a confirmed category nor a prediction is counted as unclassified. Percentages use `categorized_reports` as the denominator.

## 2. Map data

`GET /api/v1/dashboard/map`

Optional query parameters:

- `date_from`, `date_to`: same inclusive UTC date rules as summary.
- `category_code`: one of the configured codes R01–R11 (case-insensitive).
- `status`: RECEIVED, PROCESSING, RESOLVED or REJECTED.
- `limit`: 1–1000, default 500.
- `offset`: zero-based offset, default 0.

Example:

```http
GET /api/v1/dashboard/map?category_code=R03&status=RECEIVED&limit=100
Authorization: Bearer <access_token>
```

The response includes paginated `items`, `total`, `limit`, `offset`, and `has_more`. Each map item contains a report ID, tracking code, numeric latitude/longitude, address, status, effective category, submission time and a 240-character content preview. Reports without both coordinates are excluded from map results.

## 3. Database and privacy notes

- Uses the existing `reports`, `incident_categories` and `predictions` tables; no new table or migration is required.
- The original AI prediction is not modified when a staff review changes the final category.
- No citizen email, name, attachment URL or raw user identity is exposed in map markers.
- Counts and map markers are read from the database at request time; no dashboard cache or fabricated sample data is used.
