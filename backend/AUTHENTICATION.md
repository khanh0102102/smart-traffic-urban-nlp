# Authentication – Phase 14

## Environment
Copy `.env.example` to `.env`. Generate a JWT secret with `python -c "import secrets; print(secrets.token_hex(32))"` and set it as `JWT_SECRET_KEY`. Create a Google OAuth 2.0 Client ID (Web application) and set `GOOGLE_CLIENT_ID`. Never commit `.env`.

## Staff provisioning
There is no public staff registration route. From the repository root:

```powershell
Push-Location backend
python -m app.core.staff_provision --email admin@example.com --full-name "System Admin" --role ADMIN
python -m app.core.staff_provision --email receiver@example.com --full-name "Report Receiver" --role RECEIVER
Pop-Location
```

The command prompts for a password (minimum 12 characters), hashes it with Argon2 and only creates ADMIN/RECEIVER accounts. It does not promote a citizen account.

## API
- `POST /api/v1/auth/google`: citizen login; JSON body `{"id_token":"..."}` from Google Identity Services.
- `POST /api/v1/auth/staff/login`: staff email/password login.
- `GET /api/v1/auth/me`: current user.
- `GET /api/v1/auth/staff/me`: current ADMIN/RECEIVER only.

Send the returned access token with `Authorization: Bearer <token>`. Default expiry is 30 minutes. Use `require_roles("ADMIN")` or `require_roles("ADMIN", "RECEIVER")` in later protected routes. Roles are reloaded from the database on each request.

## Tests
```powershell
python -m pip install -r backend/requirements-dev.txt
python -m pytest backend/tests/test_auth.py -q
```

Tests use SQLite and mock the Google verifier. Real Google sign-in needs a configured Web Client ID and a valid Google ID token.
