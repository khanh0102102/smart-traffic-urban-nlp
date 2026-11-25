"""Create a staff account from a trusted backend environment.

Run with: python -m app.core.staff_provision --email ... --full-name ... --role ADMIN
"""
from __future__ import annotations
import argparse
from getpass import getpass

from sqlalchemy import select

from app.core.security import hash_password
from app.database.models import User
from app.database.session import SessionLocal


def main() -> int:
    parser = argparse.ArgumentParser(description="Provision an ADMIN or RECEIVER account.")
    parser.add_argument("--email", required=True)
    parser.add_argument("--full-name", required=True)
    parser.add_argument("--role", choices=("ADMIN", "RECEIVER"), required=True)
    args = parser.parse_args()
    email = args.email.strip().lower()
    if "@" not in email or len(email) > 255:
        parser.error("--email must be a valid email address.")
    password = getpass("New staff password (minimum 12 characters): ")
    if password != getpass("Confirm password: "):
        parser.error("Passwords do not match.")
    try:
        hashed = hash_password(password)
    except ValueError as exc:
        parser.error(str(exc))

    with SessionLocal() as db:
        if db.scalar(select(User).where(User.email == email)) is not None:
            parser.error("Email already exists. This command does not promote citizen accounts.")
        user = User(
            email=email, full_name=args.full_name.strip()[:160], role=args.role,
            password_hash=hashed, google_sub=None, is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Created {user.role} account: {user.email} ({user.id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
