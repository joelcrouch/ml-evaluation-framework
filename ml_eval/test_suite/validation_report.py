"""Validation report for test suite loading."""
from typing import List, Dict
from enum import Enum


class Severity(Enum):
    """Severity levels for validation errors."""
    ERROR = "ERROR"
    WARNING = "WARNING"


class ValidationReport:
    """Collects and reports validation errors for test suites."""

    def __init__(self):
        self.errors: List[Dict[str, any]] = []
        self.valid_count: int = 0
        self.invalid_count: int = 0
        self.duplicate_count: int = 0
        self._test_case_errors: Dict[int, List[Dict[str, str]]] = {}

    def add_error(self, test_case_index: int, message: str,
                  severity: Severity = Severity.ERROR, field: str = None):
        """Add a validation error for a specific test case.

        Args:
            test_case_index: Index of the test case in the suite
            message: Error message
            severity: Severity level (ERROR or WARNING)
            field: Optional field name that caused the error
        """
        error = {
            "test_case_index": test_case_index,
            "message": message,
            "severity": severity.value,
            "field": field
        }
        self.errors.append(error)

        if test_case_index not in self._test_case_errors:
            self._test_case_errors[test_case_index] = []
        self._test_case_errors[test_case_index].append(error)

    def mark_valid(self):
        """Increment valid test case count."""
        self.valid_count += 1

    def mark_invalid(self):
        """Increment invalid test case count."""
        self.invalid_count += 1

    def mark_duplicate(self):
        """Increment duplicate test case count."""
        self.duplicate_count += 1

    def has_errors(self) -> bool:
        """Check if any errors were recorded."""
        return len(self.errors) > 0

    def has_critical_errors(self) -> bool:
        """Check if any ERROR-level issues exist."""
        return any(err["severity"] == Severity.ERROR.value for err in self.errors)

    def get_test_case_errors(self, test_case_index: int) -> List[Dict[str, str]]:
        """Get all errors for a specific test case."""
        return self._test_case_errors.get(test_case_index, [])

    def render(self) -> str:
        """Render a formatted report for console output.

        Returns:
            Formatted string with validation results
        """
        lines = []
        lines.append("=" * 70)
        lines.append("Test Suite Validation Report")
        lines.append("=" * 70)
        lines.append(f"✅ Valid test cases:     {self.valid_count}")
        lines.append(f"❌ Invalid test cases:   {self.invalid_count}")
        lines.append(f"⚠️  Duplicate test cases: {self.duplicate_count}")
        lines.append(f"📊 Total processed:      {self.valid_count + self.invalid_count + self.duplicate_count}")

        if self.errors:
            lines.append("")
            lines.append("-" * 70)
            lines.append("Validation Errors:")
            lines.append("-" * 70)

            # Group errors by test case
            for test_case_idx in sorted(self._test_case_errors.keys()):
                errors = self._test_case_errors[test_case_idx]
                lines.append(f"\n[Test Case #{test_case_idx}]")
                for err in errors:
                    severity_icon = "❌" if err["severity"] == "ERROR" else "⚠️ "
                    field_info = f" (field: {err['field']})" if err.get('field') else ""
                    lines.append(f"  {severity_icon} {err['message']}{field_info}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def __str__(self):
        return self.render()
