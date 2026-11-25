"""Fast unit tests for authentication primitives and JWT invariants."""
from __future__ import annotations

import pytest
import jwt

from app.core.config import settings
from app.core.security import create_access_token, decode_access_token, hash_password, verify_password


def configure_jwt(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret_key", "phase24-test-secret-" * 3)
    monkeypatch.setattr(settings, "jwt_algorithm", "HS256")
    monkeypatch.setattr(settings, "jwt_issuer", "st-nlp-api")
    monkeypatch.setattr(settings, "jwt_audience", "st-nlp-client")
    monkeypatch.setattr(settings, "jwt_access_token_expire_minutes", 30)


def test_password_hash_verifies_only_the_original_password():
    password = "Strong-test-password-2026!"
    encoded = hash_password(password)
    assert encoded != password
    assert verify_password(password, encoded) is True
    assert verify_password("different-password", encoded) is False
    assert verify_password(password, None) is False


@pytest.mark.parametrize("password", ["short", ""])
def test_password_hash_rejects_short_or_blank_password(password):
    with pytest.raises(ValueError):
        hash_password(password)


def test_password_hash_rejects_values_above_the_documented_limit():
    with pytest.raises(ValueError, match="128 characters"):
        hash_password("x" * 129)


def test_jwt_round_trip_keeps_subject_role_issuer_and_audience(monkeypatch):
    configure_jwt(monkeypatch)
    token, expires_in = create_access_token("cdb7a1d6-4a78-4b77-93c9-9fc6d7e8d159", "RECEIVER")
    claims = decode_access_token(token)
    assert expires_in == 1800
    assert claims["sub"] == "cdb7a1d6-4a78-4b77-93c9-9fc6d7e8d159"
    assert claims["role"] == "RECEIVER"
    assert claims["type"] == "access"
    assert claims["iss"] == "st-nlp-api"
    assert claims["aud"] == "st-nlp-client"


def test_jwt_rejects_modified_signature(monkeypatch):
    configure_jwt(monkeypatch)
    token, _ = create_access_token("cdb7a1d6-4a78-4b77-93c9-9fc6d7e8d159", "CITIZEN")
    modified = token[:-1] + ("a" if token[-1] != "a" else "b")
    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token(modified)
