
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from ml_eval.main import app
from ml_eval.database.connection import get_db, Base
from alembic.config import Config
from alembic import command
import os
from pathlib import Path

# --- Test Database Configuration ---
# Use a PostgreSQL database for testing
# NOTE: This assumes a PostgreSQL server is running and a database named 'ml_eval_db_test' exists.
db_user = os.getenv("POSTGRES_USER", "ml_user")
db_password = os.getenv("POSTGRES_PASSWORD", "ml_password")
db_host = "localhost"
db_port = os.getenv("POSTGRES_PORT", "5433")
db_name = os.getenv("POSTGRES_DB", "ml_eval_db") + "_test"

SQLALCHEMY_DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# --- Fixture for Test Database ---
@pytest.fixture(scope="function")
def db_session():
    """
    Create a new database session for a test, using Alembic migrations for schema.

    IMPORTANT: This fixture uses Alembic migrations to create the schema, ensuring
    tests run against the same schema as dev/prod. This prevents schema drift.
    """
    # Set up Alembic configuration to use test database
    project_root = Path(__file__).parent.parent
    alembic_cfg = Config(str(project_root / "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", SQLALCHEMY_DATABASE_URL)

    # Run migrations to create schema
    command.upgrade(alembic_cfg, "head")

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Clean up: drop all tables after test
        # We use raw SQL to drop all tables in the public schema
        with engine.connect() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
            conn.commit()


# --- Fixture for Test Client ---
@pytest.fixture(scope="function")
def client(db_session):
    """
    Create a new TestClient for a test.
    """

    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    del app.dependency_overrides[get_db]

