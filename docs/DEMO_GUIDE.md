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


# Create ModelRun for baseline_model
curl -X 'POST' \
   'http://localhost:8000/api/v1/runs/' \
   -H 'Content-Type: application/json' \
   -d '{
     "model_name": "baseline_model",
    "model_version": "1.0",
    "model_type": "time_series_keras"
   }'


# Baseline Time series instructions


Only the baseline is implemented to run w/ adaptor/model/seed scripts properly.


## Step 4: Run the Baseline Time Series Demo

This demonstrates evaluating a baseline temperature prediction model on weather data using the same framework.

### 4.1 Train the Baseline Model and Create Golden Dataset

The baseline model predicts temperature one hour into the future by assuming "no change" from the current temperature. This simple model serves as a performance benchmark.
```bash
python ml_eval/core/implementations/train_baseline_time_series.py
```

**What this does:**
- Trains the baseline model on weather data (70% train, 20% validation, 10% test)
- Evaluates on validation and test sets (expect MAE ~0.078-0.085)
- Saves the model to `models/baseline_model.keras`
- Creates 50 golden test cases from the test set
- Saves golden dataset to `data/baseline_golden_dataset.json`
- Generates tutorial-style prediction charts

**Expected output:**
```
Validation - Loss: 0.0128, MAE: 0.0785
Test - Loss: 0.0142, MAE: 0.0852
✅ Baseline model saved!
✅ Created 50 golden test cases
```

### 4.2 Seed the Database with Baseline Test Cases

Load the golden dataset into the evaluation framework:
```bash
python scripts/seed_baseline_test_cases.py
```

**Expected output:** [currently, mayy change with a future seed script]
```
Loaded 50 test cases from golden dataset
...created 10 test cases so far...
...created 20 test cases so far...
...created 30 test cases so far...
...created 40 test cases so far...
...created 50 test cases so far...
✅ Successfully created: 50/50 test cases
```

### 4.3 Create a Model Run

Register a new evaluation run for the baseline model:
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/runs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "baseline_model",
    "model_version": "1.0",
    "model_type": "baseline_time_series"
  }'
```

Note the **`"id"`** value from the response (e.g., `"id":2`). This is your **`<run_id>`** for the baseline model.

### 4.4 Run the Evaluation

Execute the evaluation using your `<run_id>`:
```bash
python scripts/run_evaluation.py 
```

**What this does:**
- Loads the saved baseline model from `models/baseline_model.keras`
- Runs predictions on all 50 golden test cases
- Calculates MSE, MAE, and RMSE for each prediction
- Stores results in the database

**Expected output:**
```
--- Loading baseline time series model from models/baseline_model.keras... ---
✅ Baseline model loaded successfully
Starting evaluation for ModelRun <run_id> (baseline_model 1.0)...
Evaluation for ModelRun <run_id> completed.

🎉 Evaluation complete for ModelRun ID: <run_id>
   - Total Cases: 50
   - Completed: 50
   - Failed: 0
```

### 4.5 Generate the Performance Report

Create comprehensive visualizations and statistics:
```bash
python scripts/generate_report_time_series.py 
```

**What this generates:**

**Console Output:**
- Overall statistics (pass/fail rates)
- Error metrics (MSE, MAE, RMSE) with mean, median, std, min, max
- Normalized scores
- Best and worst predictions

**Chart Files** (saved to `reports/`):
1. **`baseline_model_v<run_id>_time_series_report.png`**
   - MSE, MAE, RMSE distributions
   - Score distribution
   - Error metrics box plots

2. **`baseline_model_v<run_id>_prediction_samples.png`**
   - Random sample predictions vs ground truth
   - Shows prediction quality visually

3. **`baseline_model_v<run_id>_tutorial_style.png`**
   - TensorFlow tutorial-style visualization
   - Blue dots: input values
   - Green circles: actual labels
   - Red X markers: predictions

4. **`baseline_model_v<run_id>_summary.csv`**
   - Detailed metrics for every test case

**Expected Performance:**
- Mean MSE: ~0.0142 (matches training test set performance)
- Mean MAE: ~0.0852
- For baseline: predictions should equal input values (demonstrating "no change" strategy)

### Understanding the Results

The baseline model's performance establishes the minimum bar that more sophisticated models (linear, dense, CNN, RNN) must beat. If a complex neural network can't outperform "just repeat the last value," it indicates the model isn't learning useful patterns.

**Key observations from charts:**
- The tutorial-style chart shows predictions (red X) matching inputs (blue dots) closely
- Ground truth (green circles) shows the actual temperature changes
- When input and ground truth are close, the baseline performs well
- Larger differences indicate periods of temperature change where advanced models could improve

---

## Next Steps: Evaluating More Complex Models

After establishing the baseline, you can train and evaluate more sophisticated models from the TensorFlow tutorial:
- **Linear Model**: Single dense layer
- **Dense Model**: Multi-layer neural network
- **CNN Model**: Convolutional neural network
- **RNN Model**: Recurrent neural network (LSTM)
- These will have essentially the same methods, with some tweaks to make them work. GOAL: Have an abadstracted function(s) that can run any time-series, or just leave them as examples for how to create the proper adaptor/model/seed scripts.


Each model should outperform the baseline's MAE of ~0.085 to demonstrate meaningful learning.

