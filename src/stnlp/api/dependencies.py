"""JWT authentication and development-only API-key fallback."""

import secrets

import jwt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from stnlp.auth.security import decode_access_token
from stnlp.config import settings
from stnlp.database.models import AdminUser
from stnlp.database.session import get_db


def require_admin(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
    db: Session = Depends(get_db),
) -> str:
    # API key fallback is intentionally available only in development for local testing.
    configured_key = settings.api_admin_key
    if (
        settings.app_env.casefold() != "production"
        and x_admin_key
        and configured_key
        and secrets.compare_digest(x_admin_key, configured_key)
    ):
        return "demo-admin"

    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.casefold() == "bearer" and token:
            try:
                payload = decode_access_token(token)
                username = payload.get("sub")
                if not username or payload.get("role") != "admin":
                    raise ValueError("Invalid token claims.")
                user = (
                    db.query(AdminUser)
                    .filter(
                        AdminUser.username == username,
                        AdminUser.is_active.is_(True),
                        AdminUser.role == "admin",
                    )
                    .first()
                )
                if user is not None:
                    return user.username
            except (jwt.InvalidTokenError, ValueError, TypeError):
                pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Cần đăng nhập Admin hoặc cung cấp token hợp lệ.",
        headers={"WWW-Authenticate": "Bearer"},
    )
