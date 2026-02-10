"""Test Suite module for loading and validating test suites."""
from ml_eval.test_suite.manager import SuiteManager
from ml_eval.test_suite.validation_report import ValidationReport, Severity

# Backward compatibility alias
TestSuiteManager = SuiteManager

__all__ = ["SuiteManager", "TestSuiteManager", "ValidationReport", "Severity"]
