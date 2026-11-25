"""Password hashing and signed JWT access-token helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import jwt
from pwdlib import PasswordHash

from app.core.config import settings

_password_hash = PasswordHash.recommended()
JWT_ALGORITHM = "HS256"


def _jwt_secret() -> str:
    secret = settings.jwt_secret_key
    if len(secret) < 32:
        raise RuntimeError(
            "JWT_SECRET_KEY must be configured with at least 32 characters. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    if settings.jwt_algorithm != JWT_ALGORITHM:
        raise RuntimeError("Only HS256 is enabled by this authentication module.")
    return secret


def hash_password(password: str) -> str:
    """Hash a staff password using the recommended Argon2 password hasher."""
    if len(password) < 12:
        raise ValueError("Staff passwords must contain at least 12 characters.")
    if len(password) > 128:
        raise ValueError("Staff passwords must not exceed 128 characters.")
    return _password_hash.hash(password)


def verify_password(plain_password: str, password_hash: str | None) -> bool:
    """Verify a submitted password; a missing hash is never accepted."""
    if not password_hash or not plain_password:
        return False
    try:
        return _password_hash.verify(plain_password, password_hash)
    except Exception:
        # Malformed or unsupported stored hashes must fail closed, not produce a 500.
        return False


def create_access_token(user_id: UUID | str, role: str) -> tuple[str, int]:
    """Create a signed, expiring access token with issuer and audience claims."""
    secret = _jwt_secret()
    lifetime = settings.jwt_access_token_expire_minutes
    if lifetime < 1 or lifetime > 24 * 60:
        raise RuntimeError("JWT_ACCESS_TOKEN_EXPIRE_MINUTES must be between 1 and 1440.")
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=lifetime)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "type": "access",
        "iat": now,
        "exp": expires,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
    }
    token = jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)
    return token, lifetime * 60


def decode_access_token(token: str) -> dict[str, Any]:
    """Validate signature, fixed algorithm, audience, issuer and expiry."""
    payload = jwt.decode(
        token,
        _jwt_secret(),
        algorithms=[JWT_ALGORITHM],
        audience=settings.jwt_audience,
        issuer=settings.jwt_issuer,
        options={"require": ["sub", "iat", "exp", "iss", "aud", "type"]},
    )
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("Unsupported token type.")
    return payload
