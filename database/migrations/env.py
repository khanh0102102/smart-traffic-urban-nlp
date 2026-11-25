"""Alembic environment for ST-NLP."""
from logging.config import fileConfig
from pathlib import Path
import sys
from alembic import context
from sqlalchemy import engine_from_config, pool
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import settings
from app.database.base import Base
from app.database import models  # noqa: F401

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)
config.set_main_option("sqlalchemy.url", settings.sqlalchemy_url.render_as_string(hide_password=False))
target_metadata = Base.metadata

def run_migrations_offline():
    context.configure(url=settings.sqlalchemy_url.render_as_string(hide_password=False), target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"}, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(config.get_section(config.config_ini_section, {}), prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
