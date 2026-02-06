"""Fixtures for test_suite tests."""
import pytest
import tempfile
import json
import os
from sqlalchemy.orm import Session

from ml_eval.database.connection import SessionLocal
from ml_eval.database.models import Base


@pytest.fixture(scope="function")
def test_db():
    """Provides a fresh database session for each test."""
    engine = SessionLocal().bind

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create session
    connection = engine.connect()
    db = Session(bind=connection)

    yield db

    # Cleanup
    db.close()
    Base.metadata.drop_all(bind=engine)


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
