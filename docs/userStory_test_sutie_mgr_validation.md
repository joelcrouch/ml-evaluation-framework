# User Stories for Sprint 2: Universal Test Suite Manager & Validation

This document breaks down the goals of Sprint 2 into actionable user stories with clear acceptance criteria.

**User Persona:** As a Machine Learning Practitioner...

---

### Story 1: Define and Load a Test Suite

**As an ML Practitioner, I want** a clear, documented format (JSON/YAML) for my test suites and a simple CLI command (`ml-eval load-suite <file>`) to upload them, **so that** I can easily provide my Golden Sets to the platform.

**Go/No-Go Criteria:**
-   **Go:** The command `ml-eval load-suite my_tests.json` successfully parses the file and creates `TestPrompt` records in the database.
-   **Go:** The command `ml-eval load-suite my_tests.yaml` successfully parses the file and creates records.
-   **Go:** The command prints a success message with the number of test cases loaded.
-   **No-Go:** The command fails with a clear error if the file does not exist or is not valid JSON/YAML.
-   **No-Go:** The command fails if the top-level structure of the file is not a list of test case objects.

---

### Story 2: Validate Test Suite Content

**As an ML Practitioner, I want** the platform to validate every test case in my uploaded suite against a defined schema, **so that** I get immediate, clear feedback on all errors in my data.

**Go/No-Go Criteria:**
-   **Go:** The `load-suite` command rejects a test case if a required field (e.g., `model_type`, `input_data`, `ground_truth`) is missing.
-   **Go:** The command provides a clear error message indicating the missing field and the test case number or name.
-   **Go:** The command validates all test cases in the file and returns a consolidated report of all errors found, not just the first one.
-   **No-Go:** The command allows a test case with a missing required field to be loaded into the database.

---

### Story 3: Domain-Specific Validation

**As an ML Practitioner, I want** the system to perform domain-specific validation on my test cases, such as checking if an image file path exists or if bounding box coordinates are valid, **so that** I can prevent runtime errors during evaluation.

**Go/No-Go Criteria:**
-   **Go:** A test case with `input_type: 'image_path'` is rejected if the file path in `input_data.path` does not exist or is outside the project directory.
-   **Go:** A test case with `output_type: 'bounding_boxes'` is rejected if the coordinates in `ground_truth.boxes` are not in the correct format (e.g., a list of `[x, y, w, h]`).
-   **Go:** A test case with `output_type: 'classification'` is rejected if `ground_truth.label` is not a string.
-   **No-Go:** The system allows a test case with an invalid, non-existent image path to be loaded.

---

### Story 4: Manage and Version Test Suites

**As an ML Practitioner, I want** to manage my test suites by versioning them and be able to retrieve them via an API, **so that** I can maintain and programmatically access my test data.

**Go/No-Go Criteria:**
-   **Go:** When loading a suite, I can provide a `version` and `suite_name` in the file's metadata, which are stored with the test cases.
-   **Go:** A new API endpoint `GET /api/v1/prompts/suite/{suite_name}/{version}` returns all test cases associated with that specific suite and version.
-   **Go:** The `load-suite` command warns or errors if I try to upload a suite with the same name and version twice.
-   **No-Go:** The system silently overwrites an existing test suite.

