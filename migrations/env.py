# migrations/env.py
import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
from dotenv import load_dotenv

# ------------------------------------------------------------------------
# 1. Add your project root to the python path so imports work
# ------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ------------------------------------------------------------------------
# 2. Load Environment Variables & Import Models
# ------------------------------------------------------------------------
# Import database connection module to ensure environment is loaded
# and to get the canonical database URL used throughout the application
from ml_eval.database.connection import SQLALCHEMY_DATABASE_URL

# Import Base and models so Alembic can detect schema changes
# This imports all models automatically since models.py is imported in connection.py
from ml_eval.database.models import Base

config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ------------------------------------------------------------------------
# 3. Set the Database URL dynamically from connection module
# ------------------------------------------------------------------------
# This ensures Alembic uses the EXACT SAME database URL as the application
# Importing from connection.py guarantees consistency across all environments
# IMPORTANT: Only override if using the dummy placeholder URL from alembic.ini
# This allows tests to set their own database URL without it being overridden
current_url = config.get_main_option("sqlalchemy.url")
if not current_url or current_url == "driver://user:pass@localhost/dbname":
    config.set_main_option("sqlalchemy.url", SQLALCHEMY_DATABASE_URL)

# ------------------------------------------------------------------------
# 4. Set Target Metadata
# ------------------------------------------------------------------------
# This tells Alembic where your "Target" schema is (your Python models)
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()