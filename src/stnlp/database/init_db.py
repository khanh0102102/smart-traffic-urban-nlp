"""Create tables and seed the canonical 11 categories."""

from sqlalchemy.orm import Session

from stnlp.database.base import Base
from stnlp.database.models import Category
from stnlp.database.session import SessionLocal, engine
from stnlp.labels import LABELS


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        seed_categories(session)


def seed_categories(session: Session) -> None:
    existing = {item.code for item in session.query(Category).all()}
    for label in LABELS:
        if label.code not in existing:
            session.add(Category(code=label.code, name=label.name))
    session.commit()
