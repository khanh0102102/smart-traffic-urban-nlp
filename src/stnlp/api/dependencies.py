"""Authentication dependencies for demo Admin endpoints."""

import secrets

from fastapi import Header, HTTPException, status

from stnlp.config import settings


def require_admin(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")) -> str:
    configured_key = settings.api_admin_key
    if not x_admin_key or not secrets.compare_digest(x_admin_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Thiếu hoặc sai X-Admin-Key.",
        )
    return "demo-admin"
