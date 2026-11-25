"""Authentication account lookup and sign-in services."""
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.core.google_auth import verify_google_id_token
from app.core.security import verify_password
from app.database.models import User


def authenticate_google_citizen(db: Session, credential: str) -> User:
    claims = verify_google_id_token(credential)
    subject, email = str(claims["sub"]), str(claims["email"]).strip().lower()
    name = str(claims.get("name") or "Google User").strip()[:160]
    user = db.scalar(select(User).where(User.google_sub == subject))
    if user is None:
        user = db.scalar(select(User).where(User.email == email))
        if user:
            if user.role != "CITIZEN" or user.password_hash:
                raise HTTPException(409, "This email belongs to a staff account.")
            if user.google_sub and user.google_sub != subject:
                raise HTTPException(409, "Email is linked to another Google identity.")
            user.google_sub, user.full_name = subject, name or user.full_name
        else:
            user = User(email=email, full_name=name or "Google User", role="CITIZEN",
                        google_sub=subject, password_hash=None, is_active=True)
            db.add(user)
    if user.role != "CITIZEN":
        raise HTTPException(403, "Only citizens may use Google sign-in.")
    if not user.is_active:
        raise HTTPException(403, "This account is inactive.")
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(409, "Google identity or email is already registered.") from exc
    return user


def authenticate_staff(db: Session, email: str, password: str) -> User:
    user = db.scalar(select(User).where(User.email == email.strip().lower()))
    valid = verify_password(password, user.password_hash if user else None)
    if user is None or user.role not in {"ADMIN", "RECEIVER"} or not user.is_active or not valid:
        raise HTTPException(401, "Email or password is incorrect.", headers={"WWW-Authenticate": "Bearer"})
    return user
