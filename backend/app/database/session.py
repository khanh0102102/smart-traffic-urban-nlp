"""SQLAlchemy engine, session factory and connection check."""

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

database_url = settings.sqlalchemy_url
connect_args = (
    {"check_same_thread": False}
    if database_url.drivername.startswith("sqlite")
    else {}
)

engine: Engine = create_engine(
    database_url,
    echo=settings.sql_echo,
    pool_pre_ping=True,
    pool_recycle=1800,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Provide one database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> None:
    """Execute a lightweight query to verify database connectivity."""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
