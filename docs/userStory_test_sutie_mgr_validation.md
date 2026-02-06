# User Stories for Sprint 2: Universal Test Suite Manager & Validation

**Status**: ✅ **COMPLETED** (2026-02-05)

This document breaks down the goals of Sprint 2 into actionable user stories with clear acceptance criteria.

**User Persona:** As a Machine Learning Practitioner...

---

### Story 1: Define and Load a Test Suite ✅

**As an ML Practitioner, I want** a clear, documented format (JSON/YAML) for my test suites and a simple CLI command (`python scripts/load_suite.py <file>`) to upload them, **so that** I can easily provide my Golden Sets to the platform.

**Go/No-Go Criteria:**
-   ✅ **Go:** The command `python scripts/load_suite.py my_tests.json` successfully parses the file and creates test case records in the database.
-   ✅ **Go:** The command `python scripts/load_suite.py my_tests.yaml` successfully parses the file and creates records.
-   ✅ **Go:** The command prints a success message with the number of test cases loaded.
-   ✅ **Go:** The command fails with a clear error if the file does not exist or is not valid JSON/YAML.
-   ✅ **Go:** The command fails if the top-level structure of the file is not a list of test case objects.

**Implementation Notes:**
- Script location: `scripts/load_suite.py`
- Supports both JSON (.json) and YAML (.yaml, .yml) formats
- Uses `TestSuiteManager` class for loading and parsing
- Exit codes: 0 (success), 1 (validation errors), 2 (fatal error)

---

### Story 2: Validate Test Suite Content ✅

**As an ML Practitioner, I want** the platform to validate every test case in my uploaded suite against a defined schema, **so that** I get immediate, clear feedback on all errors in my data.

**Go/No-Go Criteria:**
-   ✅ **Go:** The `load-suite` command rejects a test case if a required field (e.g., `model_type`, `input_data`, `ground_truth`) is missing.
-   ✅ **Go:** The command provides a clear error message indicating the missing field and the test case number or name.
-   ✅ **Go:** The command validates all test cases in the file and returns a consolidated report of all errors found, not just the first one.
-   ✅ **Go:** The command does NOT allow a test case with a missing required field to be loaded into the database (by default).

**Implementation Notes:**
- Validation performed by `TestSuiteManager.validate_suite()` method
- Returns `ValidationReport` object with all errors collected (no fail-fast)
- Required fields: `test_case_name`, `model_type`, `input_type`, `output_type`, `input_data`, `ground_truth`
- Invalid test cases are skipped by default (`skip_invalid=True`)

---

### Story 3: Domain-Specific Validation ✅

**As an ML Practitioner, I want** the system to perform domain-specific validation on my test cases, such as checking if an image file path exists or if bounding box coordinates are valid, **so that** I can prevent runtime errors during evaluation.

**Go/No-Go Criteria:**
-   ✅ **Go:** A test case with `input_type: 'image_path'` is rejected if the file path in `input_data.path` does not exist or has invalid extension.
-   ✅ **Go:** A test case with `output_type: 'bounding_boxes'` is rejected if the coordinates in `ground_truth.boxes` are not in the correct format.
-   ✅ **Go:** A test case with `output_type: 'classification'` is rejected if `ground_truth.label` is not a string.
-   ✅ **Go:** The system does NOT allow a test case with an invalid, non-existent image path to be loaded (by default).

**Implementation Notes:**
- Input validators: `TextInputValidator`, `ImagePathValidator`, `TabularInputValidator`, `TimeSeriesInputValidator`, `AudioPathValidator`
- Output validators: `ClassificationOutputValidator`, `BoundingBoxValidator`, `RegressionOutputValidator`, `TextOutputValidator`
- Validators are pluggable - custom validators can be registered via `TestSuiteManager.register_input_validator()` / `register_output_validator()`
- Default validators are automatically registered on `TestSuiteManager` initialization

---

### Story 4: Manage and Version Test Suites ⚠️ (Partially Complete)

**As an ML Practitioner, I want** to manage my test suites by versioning them and be able to retrieve them via an API, **so that** I can maintain and programmatically access my test data.

**Go/No-Go Criteria:**
-   ✅ **Go:** When loading a suite, I can provide `suite_name` and `suite_version` in the test case's `test_case_metadata` field, which are stored with the test cases.
-   ⏸️ **Partial:** API endpoint for retrieving by suite version not yet implemented (future enhancement).
-   ✅ **Go:** The `load-suite` command with `--skip-duplicates` flag prevents loading duplicate test cases.
-   ✅ **Go:** The system does not silently overwrite - duplicates are reported in the validation report.

**Implementation Notes:**
- Suite metadata extraction via `TestSuiteManager.get_suite_metadata()` method
- Metadata stored in `test_case_metadata` JSONB field
- Duplicate detection via `TestSuiteManager.check_duplicates()` method
- Duplicates detected by: (1) test_case_name + model_type match, OR (2) input_data hash match
- Use `--skip-duplicates` flag to prevent loading duplicates

**Future Enhancement:**
- Add API endpoint `GET /api/v1/prompts/suite/{suite_name}/{version}` to filter by suite version

