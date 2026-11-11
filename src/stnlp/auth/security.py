"""Password hashing and signed JWT helpers for the Admin demo."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt

from stnlp.config import settings

_HASH_ALGORITHM = "pbkdf2_sha256"
_HASH_ITERATIONS = 310_000


def hash_password(password: str) -> str:
    if len(password) < 10:
        raise ValueError("Mật khẩu Admin phải có ít nhất 10 ký tự.")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _HASH_ITERATIONS
    )
    return "$".join([
        _HASH_ALGORITHM,
        str(_HASH_ITERATIONS),
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    ])


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algorithm, iterations, salt_text, digest_text = encoded_hash.split("$", 3)
        if algorithm != _HASH_ALGORITHM:
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(iterations)
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, UnicodeError):
        return False


def create_access_token(username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": "admin",
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")


def decode_access_token(token: str) -> dict:
    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=["HS256"],
        options={"require": ["exp", "iat", "sub"]},
    )
