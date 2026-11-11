from stnlp.auth.security import create_access_token, decode_access_token, hash_password, verify_password


def test_password_hash_verifies_without_storing_plaintext():
    password = "A-secure-demo-password!"
    encoded = hash_password(password)
    assert password not in encoded
    assert verify_password(password, encoded) is True
    assert verify_password("incorrect-password", encoded) is False


def test_jwt_token_contains_admin_subject():
    token = create_access_token("stnlp-admin")
    payload = decode_access_token(token)
    assert payload["sub"] == "stnlp-admin"
    assert payload["role"] == "admin"
