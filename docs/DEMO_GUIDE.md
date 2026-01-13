# How to Demo the ML Evaluation Framework

This guide provides a step-by-step walkthrough for demonstrating the core functionality of the ML Evaluation Framework using the included flower image classification model.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
-   [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management.
-   [Docker](https://docs.docker.com/get-docker/) for running the PostgreSQL database.

## Step 1: Environment Setup

First, set up the project and the Conda environment.

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ml-evaluation-framework
    ```

2.  **Set up the Conda environment:**
    This script creates the `ml-eval-framework` Conda environment from the `environment.yml` file.
    ```bash
    ./setup_env.sh
    ```

3.  **Activate the environment:**
    You must activate the environment in every new terminal session you use for this project.
    ```bash
    conda activate ml-eval-framework
    ```

## Step 2: Start Services

Next, start the required background services: the database and the web server.

1.  **Start the PostgreSQL Database:**
    This command uses Docker Compose to start the database container in the background.
    ```bash
    ./start_db.sh
    ```

2.  **Create Database Tables:**
    This script connects to the newly started database and creates all the necessary tables (e.g., `test_cases`, `model_runs`). **This is a critical step.**
    ```bash
    python scripts/setup_db.py
    ```

3.  **Start the FastAPI Web Server:**
    Open a **new terminal window**, activate the conda environment (`conda activate ml-eval-framework`), and then run the following command. This server will handle API requests from the seeding scripts.
    ```bash
    uvicorn ml_eval.main:app --host 0.0.0.0 --port 8000
    ```
    **Leave this terminal open**, as it will display live logs from the server.

## Step 3: Run the Flower Classification Demo

With the services running, you can now execute the end-to-end evaluation for the flower classification model. Use a **third terminal window** with the conda environment activated.

1.  **Seed the Database with Test Cases:**
    This script downloads the `tf_flowers` dataset, extracts the test images, and populates the `test_cases` table in the database by calling the running API server.
    ```bash
    python scripts/seed_cv_test_cases.py
    ```
    You should see output indicating that it has successfully created 734 test cases.

2.  **Create a Model Run:**
    This `curl` command tells the platform that you want to start a new evaluation run for the `image_classification` model type.
    ```bash
    curl -X 'POST' \
      'http://localhost:8000/api/v1/runs/' \
      -H 'Content-Type: application/json' \
      -d '{
        "model_name": "FlowerClassifier-Full-Test-734",
        "model_version": "1.0",
        "model_type": "image_classification"
      }'
    ```
    The server will respond with a JSON object. Note the **`"id"`** value from the response (e.g., `"id":1`). This is your **`<run_id>`**.

3.  **Run the Evaluation:**
    Now, execute the main evaluation script, replacing `<run_id>` with the ID you received from the previous step.
    ```bash
    python scripts/run_evaluation.py <run_id>
    ```
    The script will find the `ModelRun`, load the appropriate model (`cv_flower_classifier.keras`), and run the evaluation for all 734 test cases. This may take a minute or two.

4.  **Generate the Report:**
    Finally, generate a performance report, again using your `<run_id>`.
    ```bash
    python scripts/generate_report.py <run_id>
    ```
    This will print a summary to the console, including overall accuracy and per-category performance. It will also save a confusion matrix chart to the `reports/` directory (e.g., `reports/run_<run_id>_accuracy_report.png`).

## Troubleshooting

*   **`relation "..." does not exist` Error:** If you see this error in the FastAPI server logs, it means you forgot to run `python scripts/setup_db.py` before starting the server and seeding the data.
*   **`ConnectionError` in Seeding Script:** If the seeding script can't connect to the API, make sure the `uvicorn` server is running in a separate terminal.
*   **TensorFlow Errors / Segmentation Faults:** The project relies on a specific set of dependencies. If you encounter low-level TensorFlow errors, the conda environment may be in an inconsistent state. The most reliable solution is to remove and recreate the environment:
    ```bash
    conda deactivate
    conda env remove -n ml-eval-framework
    ./setup_env.sh
    conda activate ml-eval-framework
    ```
    Then, restart the demo from Step 2.
