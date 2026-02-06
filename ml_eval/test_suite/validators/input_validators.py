"""Input validators for test suite validation."""
import os
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseInputValidator(ABC):
    """Abstract base class for input validators."""

    @abstractmethod
    def validate(self, test_case: Dict) -> List[str]:
        """Validate the input_data of a test case.

        Args:
            test_case: Test case dictionary with input_data

        Returns:
            List of error messages (empty if valid)
        """
        pass


class TextInputValidator(BaseInputValidator):
    """Validator for text input data."""

    def validate(self, test_case: Dict) -> List[str]:
        """Validate text input.

        Expected format:
        input_data: {
            "text": "some text string"
        }
        """
        errors = []
        input_data = test_case.get("input_data", {})

        if "text" not in input_data:
            errors.append("text input requires 'text' field in input_data")
        elif not isinstance(input_data["text"], str):
            errors.append("'text' field must be a string")
        elif not input_data["text"].strip():
            errors.append("'text' field cannot be empty")

        return errors


class ImagePathValidator(BaseInputValidator):
    """Validator for image path input data."""

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    def validate(self, test_case: Dict) -> List[str]:
        """Validate image path input.

        Expected format:
        input_data: {
            "path": "/path/to/image.jpg"
        }

        Checks:
        - 'path' field exists
        - File extension is valid
        - File exists (if absolute path or relative to project root)
        """
        errors = []
        input_data = test_case.get("input_data", {})

        if "path" not in input_data:
            errors.append("image_path input requires 'path' field in input_data")
            return errors

        path = input_data["path"]
        if not isinstance(path, str):
            errors.append("'path' field must be a string")
            return errors

        # Check file extension
        _, ext = os.path.splitext(path)
        if ext.lower() not in self.VALID_EXTENSIONS:
            errors.append(
                f"Invalid image extension '{ext}'. "
                f"Valid extensions: {', '.join(sorted(self.VALID_EXTENSIONS))}"
            )

        # Check if file exists
        if os.path.isabs(path):
            if not os.path.exists(path):
                errors.append(f"Image file not found: {path}")
        else:
            # Try relative to project root
            project_root = os.getcwd()
            full_path = os.path.join(project_root, path)
            if not os.path.exists(full_path):
                errors.append(
                    f"Image file not found: {path} "
                    f"(checked: {full_path})"
                )

        return errors


class TabularInputValidator(BaseInputValidator):
    """Validator for tabular input data."""

    def validate(self, test_case: Dict) -> List[str]:
        """Validate tabular input.

        Expected format:
        input_data: {
            "features": [...],  # List or dict of features
            "schema": {...}     # Optional schema definition
        }
        """
        errors = []
        input_data = test_case.get("input_data", {})

        if "features" not in input_data:
            errors.append("tabular input requires 'features' field in input_data")
            return errors

        features = input_data["features"]

        # Features can be a list or dict
        if not isinstance(features, (list, dict)):
            errors.append("'features' must be a list or dictionary")
            return errors

        # If list, check it's not empty
        if isinstance(features, list) and len(features) == 0:
            errors.append("'features' list cannot be empty")

        # If dict, check it has at least one key
        if isinstance(features, dict) and len(features) == 0:
            errors.append("'features' dictionary cannot be empty")

        # Validate schema if provided
        if "schema" in input_data:
            schema = input_data["schema"]
            if not isinstance(schema, dict):
                errors.append("'schema' must be a dictionary")

        return errors


class TimeSeriesInputValidator(BaseInputValidator):
    """Validator for time series input data."""

    def validate(self, test_case: Dict) -> List[str]:
        """Validate time series input.

        Expected format:
        input_data: {
            "window": [[...], [...], ...]  # 2D array
            OR
            "sequence": [...]              # 1D array
        }
        """
        errors = []
        input_data = test_case.get("input_data", {})

        has_window = "window" in input_data
        has_sequence = "sequence" in input_data

        if not has_window and not has_sequence:
            errors.append(
                "time series input requires either 'window' or 'sequence' "
                "field in input_data"
            )
            return errors

        # Validate window (2D array)
        if has_window:
            window = input_data["window"]
            if not isinstance(window, list):
                errors.append("'window' must be a list")
            elif len(window) == 0:
                errors.append("'window' cannot be empty")
            else:
                # Check if it's a 2D array
                if not all(isinstance(row, list) for row in window):
                    errors.append("'window' must be a 2D array (list of lists)")
                else:
                    # Check all rows have same length
                    row_lengths = [len(row) for row in window]
                    if len(set(row_lengths)) > 1:
                        errors.append(
                            f"'window' rows must have same length, "
                            f"got lengths: {row_lengths}"
                        )

        # Validate sequence (1D array)
        if has_sequence:
            sequence = input_data["sequence"]
            if not isinstance(sequence, list):
                errors.append("'sequence' must be a list")
            elif len(sequence) == 0:
                errors.append("'sequence' cannot be empty")
            else:
                # Check all elements are numbers
                if not all(isinstance(x, (int, float)) for x in sequence):
                    errors.append("'sequence' must contain only numbers")

        return errors


class AudioPathValidator(BaseInputValidator):
    """Validator for audio path input data."""

    VALID_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

    def validate(self, test_case: Dict) -> List[str]:
        """Validate audio path input.

        Expected format:
        input_data: {
            "path": "/path/to/audio.wav"
        }
        """
        errors = []
        input_data = test_case.get("input_data", {})

        if "path" not in input_data:
            errors.append("audio_path input requires 'path' field in input_data")
            return errors

        path = input_data["path"]
        if not isinstance(path, str):
            errors.append("'path' field must be a string")
            return errors

        # Check file extension
        _, ext = os.path.splitext(path)
        if ext.lower() not in self.VALID_EXTENSIONS:
            errors.append(
                f"Invalid audio extension '{ext}'. "
                f"Valid extensions: {', '.join(sorted(self.VALID_EXTENSIONS))}"
            )

        # Check if file exists
        if os.path.isabs(path):
            if not os.path.exists(path):
                errors.append(f"Audio file not found: {path}")
        else:
            # Try relative to project root
            project_root = os.getcwd()
            full_path = os.path.join(project_root, path)
            if not os.path.exists(full_path):
                errors.append(
                    f"Audio file not found: {path} "
                    f"(checked: {full_path})"
                )

        return errors
