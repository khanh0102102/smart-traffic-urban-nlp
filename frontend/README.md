# Frontend

- Citizen portal: `/citizen/` (Phase 20).
- Admin portal: `/admin/` (separate route and page; developed independently).

The citizen portal uses static HTML/CSS/JavaScript and existing FastAPI endpoints. No frontend build step is required.

## Google Sign-In setup

1. Create an OAuth 2.0 Client ID of type **Web application** in Google Cloud Console.
2. Add the exact local origin (for example `http://127.0.0.1:8000`) to Authorized JavaScript origins.
3. Set the same client ID in backend `GOOGLE_CLIENT_ID`.
4. Replace `YOUR_GOOGLE_WEB_CLIENT_ID.apps.googleusercontent.com` in the `google-client-id` meta tag in `citizen/index.html`.
5. Restart the backend and open `http://127.0.0.1:8000/citizen/`.

Google Identity Services returns an ID token to the browser. The browser posts it to `POST /api/v1/auth/google`; the backend verifies signature, expiry, issuer and audience before issuing the ST-NLP JWT. Never expose a Google client secret or JWT signing secret in frontend code.

## Citizen workflow

- Google login creates/resolves a CITIZEN account.
- Submit text and optional address.
- Choose a map point or use browser geolocation (permission required; geolocation requires HTTPS or localhost).
- Preview/remove up to 5 JPEG, PNG or WebP images (5 MB each).
- The portal creates the report, uploads images, then calls the prediction API.
- Tracking codes and previous reports are loaded from `GET /api/v1/reports/mine`.

The report is persisted before inference. If the trained inference bundle is missing, the UI keeps the report and tracking code and displays the AI error; it never fabricates a prediction.

## Browser assets

Google Identity Services, Leaflet 1.9.4, OpenStreetMap tiles and Be Vietnam Pro font are loaded from public CDNs. Map display needs internet access. Address is entered by the citizen; map clicks save latitude/longitude.
