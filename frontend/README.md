# Frontend

The frontend contains two independent portal entry points:

- Citizen portal: `/citizen/` (Google sign-in, report submission, map picker and AI result).
- Admin portal: `/admin/` (staff login, dashboard, report queue, case review and map).

The pages use static HTML/CSS/JavaScript and existing FastAPI endpoints. No frontend package installation or build step is required.

## Admin portal

Staff accounts are provisioned only from the trusted backend environment; there is no public staff registration route. See [Authentication – Staff provisioning](../backend/AUTHENTICATION.md).

The Admin portal uses:
- `POST /api/v1/auth/staff/login` and `GET /api/v1/auth/staff/me`.
- `GET /api/v1/dashboard/summary` for status counters and all configured incident categories.
- `GET /api/v1/reports` and `GET /api/v1/reports/{report_id}` for the report queue and details.
- `GET /api/v1/predictions/report/{report_id}` for AI predictions.
- `POST /api/v1/reviews` and `GET /api/v1/reviews/report/{report_id}` for confirmation/correction and audit history.
- `PATCH /api/v1/reports/{report_id}/status` for the operational status.
- `GET /api/v1/dashboard/map` for geolocated report markers.
- `GET /api/v1/reports/{report_id}/attachments/{attachment_id}` to fetch protected image attachments using the staff bearer token.

The UI only permits ADMIN/RECEIVER sessions and the backend enforces the same roles on every API request. AI predictions remain immutable; review decisions append to `prediction_reviews` and the review API updates the report's official category.

## Run locally

From the repository root:

```powershell
Copy-Item .env.example .env
# Configure DATABASE_URL/DB_* and a strong JWT_SECRET_KEY in .env.
docker compose up -d postgres
python -m pip install -r backend/requirements-dev.txt
alembic upgrade head
python scripts/seed_categories.py
Push-Location backend
python -m app.core.staff_provision --email admin@example.com --full-name "System Admin" --role ADMIN
python -m app.core.staff_provision --email receiver@example.com --full-name "Report Receiver" --role RECEIVER
Pop-Location
uvicorn app.main:app --app-dir backend --reload
```

The staff provisioning command prompts for a password (minimum 12 characters). Open `http://127.0.0.1:8000/admin/`. Use the account email and password created by the provisioning command.

The map uses Leaflet 1.9.4 and OpenStreetMap tiles from public CDNs and requires internet access. Map filters use only reports that contain both latitude and longitude. Detail attachments are fetched with the authenticated bearer token and rendered as local object URLs.

## Google Sign-In for citizens

Create an OAuth 2.0 Web Client ID, set `GOOGLE_CLIENT_ID` in backend `.env`, and update the `google-client-id` meta tag in `citizen/index.html`. Add the exact local origin to Authorized JavaScript origins. Never expose a Google client secret or JWT signing secret in frontend code.
