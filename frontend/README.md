# Frontend Integration – Phase 22

The frontend contains two independent portals that share one Axios API client:

- Citizen portal: `/citizen/` (Google sign-in, report submission, attachments, map picker, AI result and report history).
- Admin portal: `/admin/` (staff login, dashboard, report queue, case details, review history and map).

## Shared Axios API client

Both pages load a pinned Axios browser build and `/assets/js/api-client.js` before their application script. The shared client:

- Uses the same-origin API base path `/api/v1`.
- Adds the Bearer token from the current portal's session storage using a request interceptor.
- Returns response data consistently to the application code.
- Normalizes network, timeout, HTTP and FastAPI validation errors into one `ApiError` shape with HTTP status and field-level details.
- Calls the portal's logout handler when a protected request receives HTTP 401. HTTP 403 is displayed as an authorization error and does not erase the current session.
- Supports JSON and multipart FormData requests; Axios generates the multipart boundary.
- Supports protected image downloads with `responseType: "blob"`.

The shared client does not replace backend authorization. Every protected endpoint still verifies the JWT and resolves the active user's role from the database. The frontend also checks the role from `/auth/me` or `/auth/staff/me` before showing a portal.

## Authentication and role separation

- Citizens sign in through Google Identity Services and `POST /api/v1/auth/google`; the backend verifies the Google ID token and issues the ST-NLP JWT.
- Staff use `POST /api/v1/auth/staff/login` and verify their current role through `GET /api/v1/auth/staff/me`.
- Citizen sessions accept only `CITIZEN`.
- Admin sessions accept only `ADMIN` or `RECEIVER`.
- JWTs are stored in `sessionStorage`, with distinct keys for each portal. Sessions end when the browser tab session ends. Old Phase 20 citizen token keys in localStorage are removed during initialization.
- There is no staff self-registration or role elevation in the frontend.

## API workflows

Citizen:
- `POST /auth/google`, `GET /auth/me`
- `POST /reports`
- `POST /reports/{report_id}/attachments`
- `POST /predictions`
- `GET /reports/mine`

Admin:
- `POST /auth/staff/login`, `GET /auth/staff/me`
- `GET /dashboard/summary`, `GET /dashboard/map`
- `GET /reports`, `GET /reports/{report_id}`
- `GET /predictions/report/{report_id}`
- `POST /reviews`, `GET /reviews/report/{report_id}`
- `PATCH /reports/{report_id}/status`
- Protected attachment download through the shared bearer-token interceptor.

The report is saved before AI inference. If the trained inference bundle is missing, the Citizen UI retains the tracking code and displays the API error rather than fabricating an AI result. Review decisions append to `prediction_reviews`; original AI prediction records are not overwritten.

## Run locally

From the repository root:

```powershell
Copy-Item .env.example .env
# Configure DB_* / DATABASE_URL, JWT_SECRET_KEY and GOOGLE_CLIENT_ID in .env.
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

Open `http://127.0.0.1:8000/citizen/` or `http://127.0.0.1:8000/admin/`. Google OAuth requires a Web Client ID in backend `.env` and the Citizen page's `google-client-id` meta tag. Staff credentials must be provisioned from the backend; public staff registration is intentionally unavailable.

Leaflet and OpenStreetMap tiles require internet access. Axios is pinned to browser build `1.21.1` from jsDelivr; those CDN assets must also be reachable by the browser.
