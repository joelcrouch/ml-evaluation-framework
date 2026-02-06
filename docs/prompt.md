Context
You are working on the ML Evaluation Framework repository.
Sprint 2 is partially complete (~30%).
Your task is to finish Sprint 2 to 100% completion,  as defined in
docs/SPRINT2_IMPLEMENTATION_PLAN.md.

Implement all Phases 1–5 below.
Follow the method names, file paths, and responsibilities  as written.

🔹 Phase 1: Core Test Suite Manager (Priority 1)
Task 1.1 — Create TestSuiteManager

File: ml_eval/test_suite/manager.py

Implement a class:

class TestSuiteManager:


It must include exactly these public methods:

load_from_file(file_path: str) -> list[dict]

Load a test suite from JSON or YAML

Auto-detect file type by extension

Raise clear errors for unsupported formats or invalid syntax

validate_suite(test_cases: list[dict]) -> ValidationReport

Validate structure, input data, and output data

Must NOT fail fast — collect all errors

save_to_database(test_cases: list[dict], db_session)

Insert valid test cases into the database

Use existing CRUD functions in ml_eval/database/crud.py

Skip invalid or duplicate cases

get_suite_metadata(test_cases: list[dict]) -> dict

Extract:

suite_name

suite_version

optional tags / metadata

Metadata should be stored in JSONB where appropriate

Task 1.2 — Update loader script

File: scripts/load_suite.py

Refactor this script to:

Instantiate TestSuiteManager

Call:

load_from_file()

validate_suite()

save_to_database()

Print a human-readable summary:

total cases

valid cases

invalid cases

duplicates

Exit codes:

0 → success

1 → validation errors

2 → fatal error (file not found, parse error)

Task 1.3 — YAML Support

Add dependency: pyyaml>=6.0

YAML must support:

lists

nested objects

anchors/references

Add example:

data/example_suite.yaml

JSON and YAML must produce identical internal structures

🔹 Phase 2: Validation Framework (Priority 2)
Task 2.1 — Input Validators

File: ml_eval/test_suite/validators/input_validators.py

Implement:

class BaseInputValidator:
    def validate(self, test_case: dict) -> list[str]


Concrete validators:

ImagePathValidator

TextInputValidator

TabularInputValidator

TimeSeriesInputValidator

AudioPathValidator

Each validator:

Returns a list of error messages

Performs domain-specific checks (file exists, schema, shape, etc.)

Task 2.2 — Output Validators

File: ml_eval/test_suite/validators/output_validators.py

Implement:

class BaseOutputValidator:
    def validate(self, test_case: dict) -> list[str]


Concrete validators:

ClassificationOutputValidator

BoundingBoxValidator

RegressionOutputValidator

TextOutputValidator

Task 2.3 — Integrate validators

Inside TestSuiteManager.validate_suite():

Call:

_validate_input_data(test_case)

_validate_output_data(test_case)

Aggregate all errors

Do not stop at first failure

🔹 Phase 3: Advanced Features (Priority 3)
Task 3.1 — Duplicate Detection

Add method to TestSuiteManager:

check_duplicates(test_cases: list[dict], db_session) -> list[int]


Duplicate criteria:

test_case_name + model_type

OR hash of input_data

Behavior:

Duplicates are reported

Duplicates are not inserted

CLI flag --skip-duplicates must be supported

Task 3.2 — Comprehensive Error Reporting

Create class:

class ValidationReport:


Responsibilities:

Collect errors per test case

Support severity levels: ERROR, WARNING

Track:

valid_count

invalid_count

duplicate_count

Render formatted console output

Task 3.3 — Suite Versioning & Metadata

Support suite_name + suite_version

Prevent loading the same suite version twice

Store metadata in JSONB fields

Add helper methods:

get_suite_versions(suite_name)

get_latest_suite_version(suite_name)

🔹 Phase 4: Testing (Priority 1)

Create tests under tests/test_suite/:

Unit Tests

test_manager.py

test_load_json_file

test_load_yaml_file

test_save_to_database

test_duplicate_detection

test_validators.py

Input validators

Output validators

Integration Tests

test_integration.py

End-to-end suite load → validation → DB insert

Mixed valid/invalid suites

Duplicate suite rejection

All tests must pass with:

pytest tests/test_suite/ -v

🔹 Phase 5: Documentation & Polish (Priority 2)

Update documentation:

docs/userStory_test_sutie_mgr_validation.md

Mark Sprint 2 stories as completed

docs/sprint2_recap.md

Summary of features

Examples of JSON/YAML suites

docs/testing_guide.md

How to test validators

Manual validation examples

✅ Acceptance Criteria

Sprint 2 is complete when:

JSON & YAML suites load correctly

All test cases are validated with domain rules

Errors are fully reported (no fail-fast)

Duplicates are detected and skipped

Test cases are persisted to the database

All Sprint 2 tests pass

Documentation is updated

⚠️ Constraints

Use existing DB schema and CRUD APIs

Do not introduce breaking changes

Follow current project structure and style

Prefer clarity and debuggability over clevernes