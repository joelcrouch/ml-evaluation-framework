"""Output validators for test suite validation."""
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseOutputValidator(ABC):
    """Abstract base class for output validators."""

    @abstractmethod
    def validate(self, test_case: Dict) -> List[str]:
        """Validate the ground_truth of a test case.

        Args:
            test_case: Test case dictionary with ground_truth

        Returns:
            List of error messages (empty if valid)
        """
        pass


class ClassificationOutputValidator(BaseOutputValidator):
    """Validator for classification output data."""

    def validate(self, test_case: Dict) -> List[str]:
        """Validate classification output.

        Expected format:
        ground_truth: {
            "label": "class_name",
            "confidence": 0.95  # Optional
        }
        """
        errors = []
        ground_truth = test_case.get("ground_truth", {})

        if "label" not in ground_truth:
            errors.append("classification output requires 'label' field in ground_truth")
            return errors

        label = ground_truth["label"]
        if not isinstance(label, str):
            errors.append("'label' must be a string")

        # Validate confidence if present
        if "confidence" in ground_truth:
            confidence = ground_truth["confidence"]
            if not isinstance(confidence, (int, float)):
                errors.append("'confidence' must be a number")
            elif not (0.0 <= confidence <= 1.0):
                errors.append(
                    f"'confidence' must be between 0.0 and 1.0, got {confidence}"
                )

        return errors


class BoundingBoxValidator(BaseOutputValidator):
    """Validator for bounding box output data."""

    def validate(self, test_case: Dict) -> List[str]:
        """Validate bounding box output.

        Expected format:
        ground_truth: {
            "boxes": [
                {"x": 10, "y": 20, "width": 100, "height": 50, "class": "cat"},
                ...
            ]
            OR
            "boxes": [[x1, y1, x2, y2, class], ...]  # Alternative format
        }
        """
        errors = []
        ground_truth = test_case.get("ground_truth", {})

        if "boxes" not in ground_truth:
            errors.append("bounding_boxes output requires 'boxes' field in ground_truth")
            return errors

        boxes = ground_truth["boxes"]
        if not isinstance(boxes, list):
            errors.append("'boxes' must be a list")
            return errors

        if len(boxes) == 0:
            errors.append("'boxes' list cannot be empty")
            return errors

        # Validate each box
        for i, box in enumerate(boxes):
            if isinstance(box, dict):
                # Dict format: {x, y, width, height, class}
                required_fields = ["x", "y", "width", "height"]
                for field in required_fields:
                    if field not in box:
                        errors.append(
                            f"Box {i}: missing required field '{field}'"
                        )
                    elif not isinstance(box[field], (int, float)):
                        errors.append(
                            f"Box {i}: '{field}' must be a number"
                        )

                # Validate coordinates are non-negative
                if "width" in box and box["width"] <= 0:
                    errors.append(f"Box {i}: 'width' must be positive")
                if "height" in box and box["height"] <= 0:
                    errors.append(f"Box {i}: 'height' must be positive")

                # Class is optional but should be string if present
                if "class" in box and not isinstance(box["class"], str):
                    errors.append(f"Box {i}: 'class' must be a string")

            elif isinstance(box, list):
                # List format: [x1, y1, x2, y2] or [x1, y1, x2, y2, class]
                if len(box) < 4:
                    errors.append(
                        f"Box {i}: list format requires at least 4 values [x1, y1, x2, y2]"
                    )
                elif not all(isinstance(v, (int, float)) for v in box[:4]):
                    errors.append(
                        f"Box {i}: coordinates must be numbers"
                    )
                elif len(box) == 5 and not isinstance(box[4], str):
                    errors.append(
                        f"Box {i}: class (5th element) must be a string"
                    )
            else:
                errors.append(
                    f"Box {i}: must be a dictionary or list, got {type(box).__name__}"
                )

        return errors


class RegressionOutputValidator(BaseOutputValidator):
    """Validator for regression output data."""

    def validate(self, test_case: Dict) -> List[str]:
        """Validate regression output.

        Expected format:
        ground_truth: {
            "value": 42.5
            OR
            "values": [42.5, 13.2, ...]  # For multi-output regression
        }
        """
        errors = []
        ground_truth = test_case.get("ground_truth", {})

        has_value = "value" in ground_truth
        has_values = "values" in ground_truth

        if not has_value and not has_values:
            errors.append(
                "regression output requires either 'value' or 'values' "
                "field in ground_truth"
            )
            return errors

        # Validate single value
        if has_value:
            value = ground_truth["value"]
            if not isinstance(value, (int, float)):
                errors.append("'value' must be a number")

        # Validate multiple values
        if has_values:
            values = ground_truth["values"]
            if not isinstance(values, list):
                errors.append("'values' must be a list")
            elif len(values) == 0:
                errors.append("'values' list cannot be empty")
            elif not all(isinstance(v, (int, float)) for v in values):
                errors.append("all elements in 'values' must be numbers")

        return errors


class TextOutputValidator(BaseOutputValidator):
    """Validator for text output data."""

    def validate(self, test_case: Dict) -> List[str]:
        """Validate text output.

        Expected format:
        ground_truth: {
            "text": "expected text output",
            "keywords": ["keyword1", "keyword2"]  # Optional
        }
        """
        errors = []
        ground_truth = test_case.get("ground_truth", {})

        if "text" not in ground_truth:
            errors.append("text output requires 'text' field in ground_truth")
            return errors

        text = ground_truth["text"]
        if not isinstance(text, str):
            errors.append("'text' field must be a string")

        # Validate keywords if present
        if "keywords" in ground_truth:
            keywords = ground_truth["keywords"]
            if not isinstance(keywords, list):
                errors.append("'keywords' must be a list")
            elif not all(isinstance(k, str) for k in keywords):
                errors.append("all 'keywords' must be strings")

        return errors
