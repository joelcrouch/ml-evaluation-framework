"""Fixtures for test_suite tests."""
import pytest
import tempfile
import json
import os
from sqlalchemy.orm import Session
from sqlalchemy import text, create_engine
from alembic.config import Config
from alembic import command
from pathlib import Path

# Test Database Configuration
db_user = os.getenv("POSTGRES_USER", "ml_user")
db_password = os.getenv("POSTGRES_PASSWORD", "ml_password")
db_host = "localhost"
db_port = os.getenv("POSTGRES_PORT", "5433")
db_name = os.getenv("POSTGRES_DB", "ml_eval_db") + "_test"
TEST_DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


@pytest.fixture(scope="function")
def test_db():
    """
    Provides a fresh database session for each test, using Alembic migrations.

    IMPORTANT: Uses Alembic migrations to create schema, ensuring tests run
    against the same schema as dev/prod. This prevents schema drift.
    """
    engine = create_engine(TEST_DATABASE_URL)

    # Get database URL for Alembic
    db_url = TEST_DATABASE_URL

    # Set up Alembic configuration to use test database
    project_root = Path(__file__).parent.parent.parent
    alembic_cfg = Config(str(project_root / "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    # Run migrations to create schema
    command.upgrade(alembic_cfg, "head")

    # Create session
    connection = engine.connect()
    db = Session(bind=connection)

    yield db

    # Cleanup: drop all tables after test
    db.close()
    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.commit()


@pytest.fixture
def sample_test_cases():
    """Sample valid test cases for testing."""
    return [
        {
            "test_case_name": "Test Case 1",
            "model_type": "nlp",
            "input_type": "text",
            "output_type": "classification",
            "input_data": {"text": "Sample text"},
            "ground_truth": {"label": "positive"}
        },
        {
            "test_case_name": "Test Case 2",
            "model_type": "cv",
            "input_type": "image_path",
            "output_type": "classification",
            "input_data": {"path": "data/seeded_images/roses/roses_1.jpg"},
            "ground_truth": {"label": "rose"}
        }
    ]


@pytest.fixture
def temp_json_file(sample_test_cases):
    """Create a temporary JSON file with test cases."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_test_cases, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_yaml_file():
    """Create a temporary YAML file with test cases."""
    yaml_content = """
- test_case_name: "YAML Test 1"
  model_type: "nlp"
  input_type: "text"
  output_type: "classification"
  input_data:
    text: "YAML test"
  ground_truth:
    label: "test"
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def invalid_test_cases():
    """Sample invalid test cases for testing validation."""
    return [
        # Missing required field
        {
            "test_case_name": "Invalid 1",
            "model_type": "nlp",
            "input_type": "text",
            "output_type": "classification",
            "input_data": {"text": "test"}
            # Missing ground_truth
        },
        # Invalid input_data
        {
            "test_case_name": "Invalid 2",
            "model_type": "cv",
            "input_type": "image_path",
            "output_type": "classification",
            "input_data": {"path": "/nonexistent/file.jpg"},
            "ground_truth": {"label": "test"}
        }
    ]
