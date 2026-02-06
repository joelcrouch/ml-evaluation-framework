"""Test Suite Manager for loading, validating, and saving test suites."""
import json
import os
import hashlib
from typing import List, Dict, Optional
from sqlalchemy.orm import Session

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ml_eval.database import crud
from ml_eval.test_suite.validation_report import ValidationReport, Severity
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


class TestSuiteManager:
    """Manages test suite loading, validation, and database persistence."""

    REQUIRED_FIELDS = [
        "test_case_name",
        "model_type",
        "input_type",
        "output_type",
        "input_data",
        "ground_truth"
    ]

    def __init__(self, register_default_validators: bool = True):
        """Initialize the TestSuiteManager.

        Args:
            register_default_validators: If True, register built-in validators
        """
        self.validators_input = {}
        self.validators_output = {}

        if register_default_validators:
            self._register_default_validators()

    def _register_default_validators(self):
        """Register built-in validators for common input/output types."""
        # Input validators
        self.register_input_validator("text", TextInputValidator())
        self.register_input_validator("image_path", ImagePathValidator())
        self.register_input_validator("tabular", TabularInputValidator())
        self.register_input_validator("time_series_window", TimeSeriesInputValidator())
        self.register_input_validator("time_series", TimeSeriesInputValidator())
        self.register_input_validator("audio_path", AudioPathValidator())

        # Output validators
        self.register_output_validator("classification", ClassificationOutputValidator())
        self.register_output_validator("bounding_boxes", BoundingBoxValidator())
        self.register_output_validator("regression", RegressionOutputValidator())
        self.register_output_validator("text", TextOutputValidator())
        self.register_output_validator("temperature_prediction", RegressionOutputValidator())

    def load_from_file(self, file_path: str) -> List[Dict]:
        """Load a test suite from JSON or YAML file.

        Auto-detects file type by extension (.json, .yaml, .yml).

        Args:
            file_path: Path to the test suite file

        Returns:
            List of test case dictionaries

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Test suite file not found: {file_path}")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            with open(file_path, 'r') as f:
                if ext == '.json':
                    data = json.load(f)
                elif ext in ['.yaml', '.yml']:
                    if not YAML_AVAILABLE:
                        raise ValueError(
                            "YAML support not available. Install pyyaml: pip install pyyaml"
                        )
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(
                        f"Unsupported file format: {ext}. "
                        f"Supported formats: .json, .yaml, .yml"
                    )

            # Validate top-level structure
            if not isinstance(data, list):
                raise ValueError(
                    f"Invalid test suite format. Expected a list of test cases, "
                    f"got {type(data).__name__}"
                )

            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {file_path}: {e}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in file {file_path}: {e}")
        except ValueError:
            # Re-raise ValueError as-is (includes unsupported format, invalid structure, etc.)
            raise
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {e}")

    def validate_suite(self, test_cases: List[Dict]) -> ValidationReport:
        """Validate all test cases in a suite.

        Does not fail fast - collects all errors across all test cases.

        Args:
            test_cases: List of test case dictionaries

        Returns:
            ValidationReport with all errors collected
        """
        report = ValidationReport()

        for idx, test_case in enumerate(test_cases):
            errors = self._validate_test_case(test_case, idx)

            if errors:
                report.mark_invalid()
                for error_msg, field in errors:
                    report.add_error(idx, error_msg, Severity.ERROR, field)
            else:
                report.mark_valid()

        return report

    def _validate_test_case(self, test_case: Dict, idx: int) -> List[tuple]:
        """Validate a single test case.

        Args:
            test_case: Test case dictionary
            idx: Index in the suite (for error reporting)

        Returns:
            List of (error_message, field_name) tuples
        """
        errors = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in test_case:
                errors.append((f"Missing required field: '{field}'", field))

        if errors:
            return errors  # Don't validate further if structure is wrong

        # Validate input data
        input_errors = self._validate_input_data(test_case)
        errors.extend(input_errors)

        # Validate output data
        output_errors = self._validate_output_data(test_case)
        errors.extend(output_errors)

        return errors

    def _validate_input_data(self, test_case: Dict) -> List[tuple]:
        """Validate input data based on input_type.

        Args:
            test_case: Test case dictionary

        Returns:
            List of (error_message, field_name) tuples
        """
        errors = []
        input_type = test_case.get("input_type")
        input_data = test_case.get("input_data")

        if not isinstance(input_data, dict):
            errors.append(("input_data must be a dictionary/object", "input_data"))
            return errors

        # Use registered validators if available
        if input_type in self.validators_input:
            validator = self.validators_input[input_type]
            validator_errors = validator.validate(test_case)
            errors.extend([(err, "input_data") for err in validator_errors])
        else:
            # Basic validation: check input_data is not empty
            if not input_data:
                errors.append(("input_data cannot be empty", "input_data"))

        return errors

    def _validate_output_data(self, test_case: Dict) -> List[tuple]:
        """Validate output data (ground_truth) based on output_type.

        Args:
            test_case: Test case dictionary

        Returns:
            List of (error_message, field_name) tuples
        """
        errors = []
        output_type = test_case.get("output_type")
        ground_truth = test_case.get("ground_truth")

        if not isinstance(ground_truth, dict):
            errors.append(("ground_truth must be a dictionary/object", "ground_truth"))
            return errors

        # Use registered validators if available
        if output_type in self.validators_output:
            validator = self.validators_output[output_type]
            validator_errors = validator.validate(test_case)
            errors.extend([(err, "ground_truth") for err in validator_errors])
        else:
            # Basic validation: check ground_truth is not empty
            if not ground_truth:
                errors.append(("ground_truth cannot be empty", "ground_truth"))

        return errors

    def register_input_validator(self, input_type: str, validator):
        """Register a validator for a specific input type.

        Args:
            input_type: The input_type to validate (e.g., 'image_path', 'text')
            validator: Validator instance with validate() method
        """
        self.validators_input[input_type] = validator

    def register_output_validator(self, output_type: str, validator):
        """Register a validator for a specific output type.

        Args:
            output_type: The output_type to validate (e.g., 'classification', 'bounding_boxes')
            validator: Validator instance with validate() method
        """
        self.validators_output[output_type] = validator

    def check_duplicates(
        self,
        test_cases: List[Dict],
        db_session: Session
    ) -> List[int]:
        """Check for duplicate test cases in the database.

        A duplicate is defined as:
        - Same test_case_name + model_type
        - OR same input_data hash

        Args:
            test_cases: List of test case dictionaries
            db_session: Database session

        Returns:
            List of indices of test cases that are duplicates
        """
        duplicate_indices = []

        for idx, test_case in enumerate(test_cases):
            test_case_name = test_case.get("test_case_name")
            model_type = test_case.get("model_type")
            input_data = test_case.get("input_data")

            if not all([test_case_name, model_type, input_data]):
                continue  # Skip invalid test cases

            # Check for name+model_type duplicate
            existing_by_name = crud.get_prompts_by_model_type(db_session, model_type)
            for existing in existing_by_name:
                if existing.test_case_name == test_case_name:
                    duplicate_indices.append(idx)
                    break
            else:
                # Check for input_data hash duplicate
                input_hash = self._hash_input_data(input_data)
                for existing in existing_by_name:
                    if self._hash_input_data(existing.input_data) == input_hash:
                        duplicate_indices.append(idx)
                        break

        return duplicate_indices

    def _hash_input_data(self, input_data: Dict) -> str:
        """Create a hash of input_data for duplicate detection.

        Args:
            input_data: Input data dictionary

        Returns:
            SHA256 hash of the serialized input_data
        """
        # Sort keys for consistent hashing
        serialized = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def save_to_database(
        self,
        test_cases: List[Dict],
        db_session: Session,
        skip_duplicates: bool = False,
        skip_invalid: bool = True
    ) -> Dict[str, int]:
        """Save valid test cases to the database.

        Args:
            test_cases: List of test case dictionaries
            db_session: Database session
            skip_duplicates: If True, skip duplicate test cases
            skip_invalid: If True, skip invalid test cases

        Returns:
            Dictionary with counts:
            - saved: Number of test cases saved
            - skipped_invalid: Number skipped due to validation errors
            - skipped_duplicate: Number skipped as duplicates
        """
        stats = {
            "saved": 0,
            "skipped_invalid": 0,
            "skipped_duplicate": 0
        }

        # Validate first
        validation_report = self.validate_suite(test_cases)

        # Check for duplicates if requested
        duplicate_indices = set()
        if skip_duplicates:
            duplicate_indices = set(self.check_duplicates(test_cases, db_session))

        # Save test cases
        for idx, test_case in enumerate(test_cases):
            # Skip invalid
            if skip_invalid and validation_report.get_test_case_errors(idx):
                stats["skipped_invalid"] += 1
                continue

            # Skip duplicates
            if idx in duplicate_indices:
                stats["skipped_duplicate"] += 1
                continue

            # Save to database
            try:
                crud.create_prompt(
                    db=db_session,
                    test_case_name=test_case["test_case_name"],
                    model_type=test_case["model_type"],
                    input_type=test_case["input_type"],
                    output_type=test_case["output_type"],
                    input_data=test_case["input_data"],
                    ground_truth=test_case["ground_truth"],
                    category=test_case.get("category"),
                    tags=test_case.get("tags"),
                    difficulty=test_case.get("difficulty"),
                    origin=test_case.get("origin", "human"),
                    is_verified=test_case.get("is_verified", True),
                    test_case_metadata=test_case.get("test_case_metadata", {}),
                    created_by=test_case.get("created_by")
                )
                stats["saved"] += 1
            except Exception as e:
                # If save fails, count as invalid
                stats["skipped_invalid"] += 1
                print(f"Warning: Failed to save test case {idx}: {e}")

        return stats

    def get_suite_metadata(self, test_cases: List[Dict]) -> Dict:
        """Extract metadata from a test suite.

        Looks for:
        - suite_name: From first test case metadata or inferred
        - suite_version: From first test case metadata or inferred
        - total_cases: Count of test cases
        - model_types: List of unique model types
        - tags: All unique tags across test cases

        Args:
            test_cases: List of test case dictionaries

        Returns:
            Dictionary with suite metadata
        """
        if not test_cases:
            return {
                "suite_name": None,
                "suite_version": None,
                "total_cases": 0,
                "model_types": [],
                "tags": []
            }

        # Try to extract suite metadata from first test case
        first_tc = test_cases[0]
        tc_metadata = first_tc.get("test_case_metadata", {})

        suite_name = tc_metadata.get("suite_name")
        suite_version = tc_metadata.get("suite_version")

        # Collect model types and tags
        model_types = set()
        all_tags = set()

        for tc in test_cases:
            if "model_type" in tc:
                model_types.add(tc["model_type"])
            if "tags" in tc and tc["tags"]:
                if isinstance(tc["tags"], list):
                    all_tags.update(tc["tags"])

        return {
            "suite_name": suite_name,
            "suite_version": suite_version,
            "total_cases": len(test_cases),
            "model_types": sorted(list(model_types)),
            "tags": sorted(list(all_tags))
        }
