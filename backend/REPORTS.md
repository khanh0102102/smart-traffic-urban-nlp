# Report API – Phase 15

## Data model
Phase 15 reuses the existing `reports` and `report_attachments` tables. It does not add a database table or migration. Report images are stored under the configured upload directory; PostgreSQL stores only the relative path, original filename, MIME type and byte size.

A citizen may create a report and access only their own reports. `ADMIN` and `RECEIVER` users may browse the report queue, inspect any report, update its workflow status and assign/clear its final incident category. All endpoints except the category list require a Bearer JWT. The category list also requires authentication.

## Configuration
Add these optional settings to `.env` (defaults shown):
```dotenv
REPORT_UPLOAD_DIR=backend/storage
REPORT_MAX_UPLOAD_SIZE_BYTES=5242880
REPORT_MAX_ATTACHMENTS=5
```
Use a persistent volume for `backend/storage` in deployed/containerized environments. Back up this directory together with the database because attachment rows reference files stored there. Do not expose the upload directory as a public static folder.

Install dependencies from the repository root:
```powershell
python -m pip install -r backend/requirements-dev.txt
```

## Endpoints
All paths are under `/api/v1`.

| Method | Endpoint | Permission | Purpose |
|---|---|---|---|
| GET | `/reports/categories` | Authenticated | List active R01–R11 categories |
| POST | `/reports` | CITIZEN | Create a report with text, address and optional coordinates |
| GET | `/reports/mine` | CITIZEN | Paginated list of own reports |
| GET | `/reports/tracking/{tracking_code}` | Owner or staff | Retrieve a report by tracking code |
| GET | `/reports/{report_id}` | Owner or staff | Retrieve report detail and attachments |
| GET | `/reports` | ADMIN, RECEIVER | Paginated staff queue; optional `status`, `category_code`, `search` filters |
| PATCH | `/reports/{report_id}/status` | ADMIN, RECEIVER | Update status and optionally set/clear final category |
| POST | `/reports/{report_id}/attachments` | Owner or staff | Upload one or more JPEG, PNG or WebP images |
| GET | `/reports/{report_id}/attachments/{attachment_id}` | Owner or staff | View an uploaded image |

### Create a report
```http
POST /api/v1/reports
Authorization: Bearer <citizen_access_token>
Content-Type: application/json
```
```json
{
  "raw_content": "Mặt đường trước số 20 có ổ gà lớn, gây nguy hiểm cho xe máy.",
  "address": "20 Nguyễn Huệ, phường trung tâm",
  "latitude": 12.6816,
  "longitude": 108.0372
}
```
Latitude and longitude must either both be supplied or both omitted. Latitude is limited to -90..90 and longitude to -180..180. New reports start with status `RECEIVED`. The response includes a unique tracking code.

### Upload images
Send a multipart/form-data request to `POST /api/v1/reports/{report_id}/attachments`; repeat the `files` field for multiple images. Maximum file count and size are configurable. The API checks image signatures and MIME type, accepts JPEG/PNG/WebP only, and generates server-side filenames. Do not send a filesystem path from the client.

### Manage status and category
```http
PATCH /api/v1/reports/{report_id}/status
Authorization: Bearer <staff_access_token>
Content-Type: application/json
```
```json
{
  "status": "PROCESSING",
  "final_category_code": "R01"
}
```
Supported statuses: `RECEIVED`, `PROCESSING`, `RESOLVED`, `REJECTED`. Omit `final_category_code` to keep the current category; use `null` to clear it. Category codes must be active R01–R11 values.

## Test
```powershell
python -m pytest backend/tests/test_reports.py -q
python -m pytest -q
```
Tests use an isolated SQLite database and a temporary upload directory. Configure real PostgreSQL, JWT and persistent file storage before deployment.
