"""Create tables and seed labels plus an optional initial Admin user."""

from sqlalchemy.orm import Session

from stnlp.config import settings
from stnlp.database.base import Base
from stnlp.database.models import AdminUser, Category
from stnlp.database.session import SessionLocal, engine
from stnlp.labels import LABELS


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        seed_categories(session)
        seed_admin_user(session)


def seed_categories(session: Session) -> None:
    existing = {item.code for item in session.query(Category).all()}
    for label in LABELS:
        if label.code not in existing:
            session.add(Category(code=label.code, name=label.name))
    session.commit()


def seed_admin_user(session: Session) -> None:
    """Create/update the configured initial Admin account when a password hash is supplied."""
    if not settings.admin_username or not settings.admin_password_hash:
        return
    user = (
        session.query(AdminUser)
        .filter(AdminUser.username == settings.admin_username)
        .first()
    )
    if user is None:
        user = AdminUser(
            username=settings.admin_username,
            password_hash=settings.admin_password_hash,
            role="admin",
            is_active=True,
        )
        session.add(user)
    else:
        user.password_hash = settings.admin_password_hash
        user.is_active = True
        user.role = "admin"
    session.commit()
