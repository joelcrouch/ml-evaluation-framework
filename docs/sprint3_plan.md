# Sprint 3 Plan: The First End-to-End Evaluation

## Goal

To run a complete evaluation for a single, simple use case: a text-based "exact match" evaluation. This will involve creating a real model adapter (for a local, dummy model), enhancing the database schema to be closer to the final vision, and building the "Query Engine" and "Evaluation Engine" logic.

## Key Objectives

### 1. Database Schema Enhancement
-   **Task:** Update the database models (`TestPrompt`, `ModelRun`, `Response`, `Evaluation`) to more closely match the `ml_eval_software_description.md`. This includes adding fields like `status`, `model_endpoint`, `config` to `ModelRun`, and so on.
-   **Rationale:** The current schema is too simplistic for the full workflow. This is a necessary step before implementing the core logic.

### 2. Create a simple "local model"
-   **Task:** Create a simple, local "model" that can be used for testing. This could be a simple Python function that takes a string and returns a predefined response, simulating a real model. For example, a function that returns "Paris" when given "What is the capital of France?".
-   **Rationale:** We need a real (but simple) model to test the `IModelAdapter` implementation.

### 3. Implement a `LocalModelAdapter`
-   **Task:** Create a concrete implementation of `IModelAdapter` called `LocalModelAdapter`. This adapter will be responsible for calling the local model function.
-   **Rationale:** This is the first real model adapter, moving beyond the dummy implementation of Sprint 2.

### 4. Implement the Core Evaluation Workflow (`run` command)
-   **Task:** Create a new `run` function/command that orchestrates the evaluation process. This function will:
    -   Take a `model_run_id` as input.
    -   Fetch all `TestPrompt`s for the `ModelRun`'s domain.
    -   For each `TestPrompt`, use the `LocalModelAdapter` to get a model output.
    -   Create a `Response` record to store the output.
    -   Use the `ExactMatchEvaluator` to evaluate the `Response`.
    -   Create an `Evaluation` record to store the result.
    -   Update the `ModelRun` status to "completed".
-   **Rationale:** This is the heart of the framework, the "Query Engine" and "Evaluation Engine" described in the software description.

### 5. Write Integration Tests for the `run` command
-   **Task:** Write tests that call the `run` command and verify that all the database records (`Response`, `Evaluation`) are created correctly and that the `ModelRun` status is updated.
-   **Rationale:** To ensure the end-to-end workflow is working as expected.
