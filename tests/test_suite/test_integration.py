"""Integration tests for test suite loading."""
import pytest
import tempfile
import json

from ml_eval.test_suite import TestSuiteManager


def test_load_suite_end_to_end(test_db):
    """Test complete workflow: load -> validate -> save."""
    # Create test suite file
    test_cases = [
        {
            "test_case_name": "Integration Test 1",
            "model_type": "nlp",
            "input_type": "text",
            "output_type": "classification",
            "input_data": {"text": "Sample text"},
            "ground_truth": {"label": "positive"}
        },
        {
            "test_case_name": "Integration Test 2",
            "model_type": "nlp",
            "input_type": "text",
            "output_type": "classification",
            "input_data": {"text": "Another sample"},
            "ground_truth": {"label": "negative"}
        }
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_cases, f)
        temp_path = f.name

    try:
        manager = TestSuiteManager()

        # Load
        loaded_cases = manager.load_from_file(temp_path)
        assert len(loaded_cases) == 2

        # Validate
        report = manager.validate_suite(loaded_cases)
        assert report.valid_count == 2
        assert report.invalid_count == 0

        # Save
        stats = manager.save_to_database(loaded_cases, test_db)
        assert stats["saved"] == 2
        assert stats["skipped_invalid"] == 0

    finally:
        import os
        os.unlink(temp_path)


def test_validation_errors_reported(test_db):
    """Test that validation errors are properly reported."""
    test_cases = [
        # Valid case
        {
            "test_case_name": "Valid",
            "model_type": "nlp",
            "input_type": "text",
            "output_type": "classification",
            "input_data": {"text": "valid"},
            "ground_truth": {"label": "test"}
        },
        # Invalid - missing ground_truth
        {
            "test_case_name": "Invalid 1",
            "model_type": "nlp",
            "input_type": "text",
            "output_type": "classification",
            "input_data": {"text": "invalid"}
        },
        # Invalid - bad image path
        {
            "test_case_name": "Invalid 2",
            "model_type": "cv",
            "input_type": "image_path",
            "output_type": "classification",
            "input_data": {"path": "/bad/path.jpg"},
            "ground_truth": {"label": "test"}
        }
    ]

    manager = TestSuiteManager()

    # Validate
    report = manager.validate_suite(test_cases)

    assert report.valid_count == 1
    assert report.invalid_count == 2

    # Check errors are reported
    errors_0 = report.get_test_case_errors(0)
    assert len(errors_0) == 0  # First case is valid

    errors_1 = report.get_test_case_errors(1)
    assert len(errors_1) > 0  # Second case has errors

    errors_2 = report.get_test_case_errors(2)
    assert len(errors_2) > 0  # Third case has errors

    # Test report rendering
    report_str = report.render()
    assert "Valid test cases:     1" in report_str
    assert "Invalid test cases:   2" in report_str


def test_duplicate_suite_rejection(test_db):
    """Test that duplicate suites are properly rejected."""
    test_cases = [
        {
            "test_case_name": "Duplicate Test",
            "model_type": "nlp",
            "input_type": "text",
            "output_type": "classification",
            "input_data": {"text": "test"},
            "ground_truth": {"label": "test"}
        }
    ]

    manager = TestSuiteManager()

    # Load first time
    stats1 = manager.save_to_database(test_cases, test_db, skip_duplicates=False)
    assert stats1["saved"] == 1
    assert stats1["skipped_duplicate"] == 0

    # Load second time with skip_duplicates=True
    stats2 = manager.save_to_database(test_cases, test_db, skip_duplicates=True)
    assert stats2["saved"] == 0
    assert stats2["skipped_duplicate"] == 1

    # Load third time with skip_duplicates=False (should save duplicate)
    stats3 = manager.save_to_database(test_cases, test_db, skip_duplicates=False)
    assert stats3["saved"] == 1  # Will save duplicate
    assert stats3["skipped_duplicate"] == 0


def test_mixed_valid_invalid_suite(test_db):
    """Test handling of mixed valid/invalid test cases."""
    test_cases = [
        {"test_case_name": "Valid 1", "model_type": "nlp", "input_type": "text",
         "output_type": "classification", "input_data": {"text": "test1"},
         "ground_truth": {"label": "test1"}},

        {"test_case_name": "Invalid - Missing Field", "model_type": "nlp",
         "input_type": "text", "output_type": "classification",
         "input_data": {"text": "test2"}},  # Missing ground_truth

        {"test_case_name": "Valid 2", "model_type": "nlp", "input_type": "text",
         "output_type": "classification", "input_data": {"text": "test3"},
         "ground_truth": {"label": "test3"}},

        {"test_case_name": "Invalid - Bad Path", "model_type": "cv",
         "input_type": "image_path", "output_type": "classification",
         "input_data": {"path": "/nonexistent.jpg"},
         "ground_truth": {"label": "cat"}},
    ]

    manager = TestSuiteManager()

    # Validate
    report = manager.validate_suite(test_cases)
    assert report.valid_count == 2
    assert report.invalid_count == 2

    # Save with skip_invalid=True
    stats = manager.save_to_database(test_cases, test_db, skip_invalid=True)
    assert stats["saved"] == 2
    assert stats["skipped_invalid"] == 2


def test_yaml_and_json_produce_same_results(test_db):
    """Test that YAML and JSON files produce identical results."""
    test_case_data = [
        {
            "test_case_name": "Format Test",
            "model_type": "nlp",
            "input_type": "text",
            "output_type": "classification",
            "input_data": {"text": "test"},
            "ground_truth": {"label": "test"}
        }
    ]

    # Create JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_case_data, f)
        json_path = f.name

    # Create YAML file
    yaml_content = """
- test_case_name: "Format Test"
  model_type: "nlp"
  input_type: "text"
  output_type: "classification"
  input_data:
    text: "test"
  ground_truth:
    label: "test"
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        manager = TestSuiteManager()

        # Load from JSON
        json_cases = manager.load_from_file(json_path)

        # Load from YAML
        yaml_cases = manager.load_from_file(yaml_path)

        # Both should have same structure
        assert len(json_cases) == len(yaml_cases) == 1
        assert json_cases[0]["test_case_name"] == yaml_cases[0]["test_case_name"]
        assert json_cases[0]["model_type"] == yaml_cases[0]["model_type"]

    finally:
        import os
        os.unlink(json_path)
        os.unlink(yaml_path)
