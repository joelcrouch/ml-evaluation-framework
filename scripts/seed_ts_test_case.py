
import os
import sys
import pandas as pd
import requests
import json
import numpy as np

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_eval.schemas import TestPromptBase, ModelRunCreate

# --- Configuration ---
API_URL = "http://localhost:8000/api/v1/prompts/"
TEST_DATA_PATH = "data/weather_test_data.csv"

# These should match the values used in your training script from the TensorFlow tutorial
# For example, if your model takes 24 hours of data to predict the next 1 hour.
SEQUENCE_LENGTH = 24  # Number of past time steps to use as input
FORECAST_HORIZON = 1  # Number of future time steps to predict

MODEL_FILENAMES = [
    "baseline_model.keras",
    "cnn.keras",
    "dense_model.keras",
    "linear_model.keras",
    "multi_step_dense.keras",
]

def create_sequences(df, sequence_length, forecast_horizon):
    """
    Creates input sequences and corresponding ground truth targets from a DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        sequence_length (int): Number of past time steps for input.
        forecast_horizon (int): Number of future time steps for ground truth.
        
    Returns:
        list: A list of tuples, where each tuple contains (input_sequence, ground_truth).
    """
    sequences = []
    num_features = df.shape[1]
    
    for i in range(len(df) - sequence_length - forecast_horizon + 1):
        input_sequence_df = df.iloc[i : i + sequence_length]
        ground_truth_df = df.iloc[i + sequence_length : i + sequence_length + forecast_horizon]
        
        # Convert DataFrames to list of dicts for JSON serialization
        input_sequence = input_sequence_df.to_dict(orient='records')
        ground_truth = ground_truth_df.to_dict(orient='records')
        
        sequences.append((input_sequence, ground_truth))
        
    return sequences


def main():
    """
    Seeds the database with time series TestCases and ModelRuns for Keras models.
    """
    print(f"--- Seeding Time Series TestCases from {TEST_DATA_PATH} ---")

    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ Error: Test data not found at {TEST_DATA_PATH}.")
        print("Please ensure your data processing script has generated this file.")
        return

    # 1. Load the processed test data
    df_test = pd.read_csv(TEST_DATA_PATH)
    
    # Generate sequences and ground truths
    time_series_examples = create_sequences(df_test, SEQUENCE_LENGTH, FORECAST_HORIZON)
    print(f"Generated {len(time_series_examples)} time series examples for seeding.")

    # 2. Seed TestCases
    created_test_case_ids = []
    total_seeded_cases = 0

    print("Seeding TestCases via FastAPI...")
    for i, (input_seq, gt_seq) in enumerate(time_series_examples):
        payload = TestPromptBase(
            test_case_name=f"Weather Forecast {i+1} (SeqLen={SEQUENCE_LENGTH}, Horiz={FORECAST_HORIZON})",
            model_type="time_series_keras", # A new model type for Keras TS models
            input_type="sequence_of_features",
            output_type="sequence_of_features",
            input_data=input_seq,
            ground_truth=gt_seq,
            category="weather_forecasting",
            tags=["keras", "time_series", "weather"],
            origin="human",
            is_verified=True,
            test_case_metadata={"sequence_length": SEQUENCE_LENGTH, "forecast_horizon": FORECAST_HORIZON}
        ).model_dump_json() # Use model_dump_json for FastAPI

        try:
            response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=payload)
            if response.status_code == 200:
                test_case_id = response.json().get("id")
                if test_case_id:
                    created_test_case_ids.append(test_case_id)
                    total_seeded_cases += 1
                if total_seeded_cases % 100 == 0:
                    print(f"  ...created {total_seeded_cases} TestCases so far...")
            else:
                print(f"  ❌ Failed to create TestCase {i+1}. Status: {response.status_code}, Response: {response.text}")
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Error: Could not connect to the API at {API_URL}.")
            print("Please ensure your FastAPI application (uvicorn) is running.")
            return

    print(f"\n🎉 Seeding complete. Created {total_seeded_cases} new TestCases.")

    if not created_test_case_ids:
        print("No TestCases were created. Aborting ModelRun creation.")
        return
        
    # 3. Create ModelRuns for each Keras model
    print("\nCreating ModelRuns for each Keras time series model...")
    created_model_run_ids = []
    for model_filename in MODEL_FILENAMES:
        model_path = os.path.join("models", model_filename)
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: Model file not found at {model_path}. Skipping ModelRun for this model.")
            continue

        model_run_name = os.path.splitext(model_filename)[0] # e.g., "baseline_model"
        
        model_run_create = ModelRunCreate(
            model_name=model_run_name,
            model_version="1.0", # Assuming default version
            model_type="time_series_keras",
            model_endpoint=model_path, # Store the path to the .keras file
            config={"sequence_length": SEQUENCE_LENGTH, "forecast_horizon": FORECAST_HORIZON},
            total_cases=len(created_test_case_ids) # Total cases for this run
        ).model_dump_json()

        try:
            response = requests.post(API_URL.replace("prompts", "runs"), headers={"Content-Type": "application/json"}, data=model_run_create)
            if response.status_code == 200:
                model_run_id = response.json().get("id")
                if model_run_id:
                    created_model_run_ids.append(model_run_id)
                    print(f"✅ Created ModelRun for '{model_run_name}' with ID: {model_run_id}")
            else:
                print(f"  ❌ Failed to create ModelRun for '{model_run_name}'. Status: {response.status_code}, Response: {response.text}")
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Error: Could not connect to the API at {API_URL.replace('prompts', 'runs')}.")
            print("Please ensure your FastAPI application (uvicorn) is running.")
            return

    if not created_model_run_ids:
        print("No ModelRuns were created.")
    else:
        print(f"\n🎉 Database seeding for time series models complete. Created {len(created_model_run_ids)} ModelRuns.")
        print("You can now run evaluations using 'python scripts/run_evaluation.py <ModelRun ID>' for each created run.")


if __name__ == "__main__":
    main()
