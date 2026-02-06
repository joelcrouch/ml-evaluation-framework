"""Tests for TestSuiteManager."""
import pytest
import tempfile
import json
import os

from ml_eval.test_suite import TestSuiteManager
from ml_eval.database import crud


def test_load_json_file(temp_json_file):
    """Test loading a JSON test suite file."""
    manager = TestSuiteManager()
    test_cases = manager.load_from_file(temp_json_file)

    assert isinstance(test_cases, list)
    assert len(test_cases) == 2
    assert test_cases[0]["test_case_name"] == "Test Case 1"
    assert test_cases[1]["test_case_name"] == "Test Case 2"


def test_load_yaml_file(temp_yaml_file):
    """Test loading a YAML test suite file."""
    manager = TestSuiteManager()
    test_cases = manager.load_from_file(temp_yaml_file)

    assert isinstance(test_cases, list)
    assert len(test_cases) == 1
    assert test_cases[0]["test_case_name"] == "YAML Test 1"


def test_load_invalid_file():
    """Test loading a non-existent file raises FileNotFoundError."""
    manager = TestSuiteManager()

    with pytest.raises(FileNotFoundError):
        manager.load_from_file("/nonexistent/file.json")


def test_load_unsupported_format():
    """Test loading an unsupported file format raises ValueError."""
    manager = TestSuiteManager()

    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"test content")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported file format"):
            manager.load_from_file(temp_path)
    finally:
        os.unlink(temp_path)


def test_load_invalid_json():
    """Test loading invalid JSON raises ValueError."""
    manager = TestSuiteManager()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{ invalid json")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Invalid JSON format"):
            manager.load_from_file(temp_path)
    finally:
        os.unlink(temp_path)


def test_validate_suite_valid(sample_test_cases):
    """Test validating a suite with valid test cases."""
    manager = TestSuiteManager()
    report = manager.validate_suite(sample_test_cases)

    assert report.valid_count == 2
    assert report.invalid_count == 0
    assert not report.has_errors()


def test_validate_suite_invalid(invalid_test_cases):
    """Test validating a suite with invalid test cases."""
    manager = TestSuiteManager()
    report = manager.validate_suite(invalid_test_cases)

    assert report.valid_count == 0
    assert report.invalid_count == 2
    assert report.has_errors()


def test_validate_suite_missing_required_fields():
    """Test validation catches missing required fields."""
    manager = TestSuiteManager()
    test_cases = [
        {
            "test_case_name": "Incomplete",
            "model_type": "nlp"
            # Missing other required fields
        }
    ]

    report = manager.validate_suite(test_cases)

    assert report.invalid_count == 1
    errors = report.get_test_case_errors(0)
    assert len(errors) > 0
    # Should have errors for missing fields
    error_messages = [e["message"] for e in errors]
    assert any("input_type" in msg for msg in error_messages)


def test_save_to_database(test_db, sample_test_cases):
    """Test saving test cases to database."""
    manager = TestSuiteManager()
    stats = manager.save_to_database(sample_test_cases, test_db)

    assert stats["saved"] == 2
    assert stats["skipped_invalid"] == 0
    assert stats["skipped_duplicate"] == 0

    # Verify in database
    saved_cases = crud.get_prompts_by_model_type(test_db, "nlp")
    assert len(saved_cases) >= 1


def test_save_to_database_skip_invalid(test_db, invalid_test_cases):
    """Test that invalid test cases are skipped by default."""
    manager = TestSuiteManager()
    stats = manager.save_to_database(
        invalid_test_cases,
        test_db,
        skip_invalid=True
    )

    assert stats["saved"] == 0
    assert stats["skipped_invalid"] == 2


def test_duplicate_detection(test_db, sample_test_cases):
    """Test duplicate detection."""
    manager = TestSuiteManager()

    # Save once
    stats1 = manager.save_to_database(sample_test_cases, test_db)
    assert stats1["saved"] == 2

    # Try to save again with skip_duplicates=True
    stats2 = manager.save_to_database(
        sample_test_cases,
        test_db,
        skip_duplicates=True
    )
    assert stats2["saved"] == 0
    assert stats2["skipped_duplicate"] == 2


def test_check_duplicates_method(test_db):
    """Test the check_duplicates method directly."""
    manager = TestSuiteManager()

    # Create and save a test case
    test_case = {
        "test_case_name": "Duplicate Test",
        "model_type": "nlp",
        "input_type": "text",
        "output_type": "classification",
        "input_data": {"text": "test"},
        "ground_truth": {"label": "test"}
    }

    crud.create_prompt(
        test_db,
        test_case_name=test_case["test_case_name"],
        model_type=test_case["model_type"],
        input_type=test_case["input_type"],
        output_type=test_case["output_type"],
        input_data=test_case["input_data"],
        ground_truth=test_case["ground_truth"]
    )

    # Check for duplicates
    duplicate_indices = manager.check_duplicates([test_case], test_db)
    assert 0 in duplicate_indices


def test_get_suite_metadata(sample_test_cases):
    """Test extracting suite metadata."""
    manager = TestSuiteManager()
    metadata = manager.get_suite_metadata(sample_test_cases)

    assert metadata["total_cases"] == 2
    assert "nlp" in metadata["model_types"]
    assert "cv" in metadata["model_types"]


def test_get_suite_metadata_with_suite_info():
    """Test extracting suite metadata when suite_name/version are present."""
    manager = TestSuiteManager()
    test_cases = [
        {
            "test_case_name": "Test 1",
            "model_type": "nlp",
            "input_type": "text",
            "output_type": "classification",
            "input_data": {"text": "test"},
            "ground_truth": {"label": "test"},
            "test_case_metadata": {
                "suite_name": "My Suite",
                "suite_version": "1.0.0"
            }
        }
    ]

    metadata = manager.get_suite_metadata(test_cases)
    assert metadata["suite_name"] == "My Suite"
    assert metadata["suite_version"] == "1.0.0"


def test_register_custom_validators():
    """Test registering custom validators."""
    manager = TestSuiteManager(register_default_validators=False)

    # Verify no validators registered
    assert len(manager.validators_input) == 0
    assert len(manager.validators_output) == 0

    # Register a custom validator
    from ml_eval.test_suite.validators import TextInputValidator
    manager.register_input_validator("custom_text", TextInputValidator())

    assert "custom_text" in manager.validators_input
