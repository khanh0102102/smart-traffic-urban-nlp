"""Server-side verification for Google Identity Services ID tokens."""
from fastapi import HTTPException
from google.auth.exceptions import TransportError
from google.auth.transport import requests
from google.oauth2 import id_token

from app.core.config import settings


def verify_google_id_token(token: str) -> dict:
    """Verify token signature, expiry, issuer and configured OAuth client audience."""
    if not settings.google_client_id:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured.")
    try:
        claims = id_token.verify_oauth2_token(
            token, requests.Request(), settings.google_client_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Google ID token is invalid or expired.") from exc
    except TransportError as exc:
        raise HTTPException(status_code=503, detail="Google verification is temporarily unavailable.") from exc
    if claims.get("iss") not in {"accounts.google.com", "https://accounts.google.com"}:
        raise HTTPException(status_code=401, detail="Invalid Google token issuer.")
    if not claims.get("sub") or not claims.get("email") or claims.get("email_verified") is not True:
        raise HTTPException(status_code=401, detail="Google account must have a verified email.")
    return claims
