"""Tests for validators."""
import pytest
import tempfile
import os

from ml_eval.test_suite.validators import (
    TextInputValidator,
    ImagePathValidator,
    TabularInputValidator,
    TimeSeriesInputValidator,
    AudioPathValidator,
    ClassificationOutputValidator,
    BoundingBoxValidator,
    RegressionOutputValidator,
    TextOutputValidator
)


# Input Validator Tests

def test_text_input_validator_valid():
    """Test TextInputValidator with valid input."""
    validator = TextInputValidator()
    test_case = {
        "input_data": {"text": "Sample text"}
    }
    errors = validator.validate(test_case)
    assert len(errors) == 0


def test_text_input_validator_missing_text():
    """Test TextInputValidator with missing text field."""
    validator = TextInputValidator()
    test_case = {
        "input_data": {}
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert "text" in errors[0]


def test_text_input_validator_empty_text():
    """Test TextInputValidator with empty text."""
    validator = TextInputValidator()
    test_case = {
        "input_data": {"text": "   "}
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert "empty" in errors[0].lower()


def test_image_path_validator_valid():
    """Test ImagePathValidator with valid existing image."""
    validator = ImagePathValidator()

    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        f.write(b"fake image data")
        temp_path = f.name

    try:
        test_case = {
            "input_data": {"path": temp_path}
        }
        errors = validator.validate(test_case)
        assert len(errors) == 0
    finally:
        os.unlink(temp_path)


def test_image_path_validator_invalid_extension():
    """Test ImagePathValidator with invalid extension."""
    validator = ImagePathValidator()
    test_case = {
        "input_data": {"path": "/path/to/file.txt"}
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert "extension" in errors[0].lower()


def test_image_path_validator_nonexistent_file():
    """Test ImagePathValidator with non-existent file."""
    validator = ImagePathValidator()
    test_case = {
        "input_data": {"path": "/nonexistent/image.jpg"}
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert "not found" in errors[0].lower()


def test_tabular_input_validator_valid_list():
    """Test TabularInputValidator with valid list of features."""
    validator = TabularInputValidator()
    test_case = {
        "input_data": {"features": [1.0, 2.0, 3.0]}
    }
    errors = validator.validate(test_case)
    assert len(errors) == 0


def test_tabular_input_validator_valid_dict():
    """Test TabularInputValidator with valid dict of features."""
    validator = TabularInputValidator()
    test_case = {
        "input_data": {"features": {"age": 25, "income": 50000}}
    }
    errors = validator.validate(test_case)
    assert len(errors) == 0


def test_tabular_input_validator_missing_features():
    """Test TabularInputValidator with missing features."""
    validator = TabularInputValidator()
    test_case = {
        "input_data": {}
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert "features" in errors[0]


def test_time_series_input_validator_valid_window():
    """Test TimeSeriesInputValidator with valid window."""
    validator = TimeSeriesInputValidator()
    test_case = {
        "input_data": {
            "window": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        }
    }
    errors = validator.validate(test_case)
    assert len(errors) == 0


def test_time_series_input_validator_valid_sequence():
    """Test TimeSeriesInputValidator with valid sequence."""
    validator = TimeSeriesInputValidator()
    test_case = {
        "input_data": {
            "sequence": [1.0, 2.0, 3.0, 4.0, 5.0]
        }
    }
    errors = validator.validate(test_case)
    assert len(errors) == 0


def test_time_series_input_validator_missing_both():
    """Test TimeSeriesInputValidator with neither window nor sequence."""
    validator = TimeSeriesInputValidator()
    test_case = {
        "input_data": {}
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert "window" in errors[0] or "sequence" in errors[0]


def test_time_series_input_validator_uneven_window():
    """Test TimeSeriesInputValidator with uneven window rows."""
    validator = TimeSeriesInputValidator()
    test_case = {
        "input_data": {
            "window": [[1.0, 2.0], [3.0, 4.0, 5.0]]
        }
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert "same length" in errors[0]


def test_audio_path_validator_valid():
    """Test AudioPathValidator with valid audio file."""
    validator = AudioPathValidator()

    # Create a temporary audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(b"fake audio data")
        temp_path = f.name

    try:
        test_case = {
            "input_data": {"path": temp_path}
        }
        errors = validator.validate(test_case)
        assert len(errors) == 0
    finally:
        os.unlink(temp_path)


def test_audio_path_validator_invalid_extension():
    """Test AudioPathValidator with invalid extension."""
    validator = AudioPathValidator()
    test_case = {
        "input_data": {"path": "/path/to/file.txt"}
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert "extension" in errors[0].lower()


# Output Validator Tests

def test_classification_output_validator_valid():
    """Test ClassificationOutputValidator with valid output."""
    validator = ClassificationOutputValidator()
    test_case = {
        "ground_truth": {"label": "positive"}
    }
    errors = validator.validate(test_case)
    assert len(errors) == 0


def test_classification_output_validator_with_confidence():
    """Test ClassificationOutputValidator with valid confidence."""
    validator = ClassificationOutputValidator()
    test_case = {
        "ground_truth": {"label": "positive", "confidence": 0.95}
    }
    errors = validator.validate(test_case)
    assert len(errors) == 0


def test_classification_output_validator_invalid_confidence():
    """Test ClassificationOutputValidator with invalid confidence."""
    validator = ClassificationOutputValidator()
    test_case = {
        "ground_truth": {"label": "positive", "confidence": 1.5}
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert "confidence" in errors[0].lower()


def test_classification_output_validator_missing_label():
    """Test ClassificationOutputValidator with missing label."""
    validator = ClassificationOutputValidator()
    test_case = {
        "ground_truth": {}
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert "label" in errors[0]


def test_bounding_box_validator_valid_dict_format():
    """Test BoundingBoxValidator with valid dict format."""
    validator = BoundingBoxValidator()
    test_case = {
        "ground_truth": {
            "boxes": [
                {"x": 10, "y": 20, "width": 100, "height": 50, "class": "cat"}
            ]
        }
    }
    errors = validator.validate(test_case)
    assert len(errors) == 0


def test_bounding_box_validator_valid_list_format():
    """Test BoundingBoxValidator with valid list format."""
    validator = BoundingBoxValidator()
    test_case = {
        "ground_truth": {
            "boxes": [[10, 20, 110, 70, "cat"]]
        }
    }
    errors = validator.validate(test_case)
    assert len(errors) == 0


def test_bounding_box_validator_missing_boxes():
    """Test BoundingBoxValidator with missing boxes."""
    validator = BoundingBoxValidator()
    test_case = {
        "ground_truth": {}
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert "boxes" in errors[0]


def test_bounding_box_validator_negative_dimensions():
    """Test BoundingBoxValidator with negative dimensions."""
    validator = BoundingBoxValidator()
    test_case = {
        "ground_truth": {
            "boxes": [
                {"x": 10, "y": 20, "width": -100, "height": 50}
            ]
        }
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert "positive" in errors[0].lower()


def test_regression_output_validator_single_value():
    """Test RegressionOutputValidator with single value."""
    validator = RegressionOutputValidator()
    test_case = {
        "ground_truth": {"value": 42.5}
    }
    errors = validator.validate(test_case)
    assert len(errors) == 0


def test_regression_output_validator_multiple_values():
    """Test RegressionOutputValidator with multiple values."""
    validator = RegressionOutputValidator()
    test_case = {
        "ground_truth": {"values": [42.5, 13.2, 99.9]}
    }
    errors = validator.validate(test_case)
    assert len(errors) == 0


def test_regression_output_validator_missing_both():
    """Test RegressionOutputValidator with neither value nor values."""
    validator = RegressionOutputValidator()
    test_case = {
        "ground_truth": {}
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert ("value" in errors[0] or "values" in errors[0])


def test_text_output_validator_valid():
    """Test TextOutputValidator with valid output."""
    validator = TextOutputValidator()
    test_case = {
        "ground_truth": {"text": "Expected output text"}
    }
    errors = validator.validate(test_case)
    assert len(errors) == 0


def test_text_output_validator_with_keywords():
    """Test TextOutputValidator with valid keywords."""
    validator = TextOutputValidator()
    test_case = {
        "ground_truth": {
            "text": "Expected output",
            "keywords": ["keyword1", "keyword2"]
        }
    }
    errors = validator.validate(test_case)
    assert len(errors) == 0


def test_text_output_validator_invalid_keywords():
    """Test TextOutputValidator with invalid keywords type."""
    validator = TextOutputValidator()
    test_case = {
        "ground_truth": {
            "text": "Expected output",
            "keywords": "not a list"
        }
    }
    errors = validator.validate(test_case)
    assert len(errors) > 0
    assert "keywords" in errors[0]
