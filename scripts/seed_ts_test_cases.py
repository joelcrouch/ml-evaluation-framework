# import os
# import requests
# import json
# import pandas as pd

# # --- Configuration ---
# API_URL = "http://localhost:8000/api/v1/prompts/"
# DATA_FILE = "/home/dell-linux-dev3/Projects/ml-evaluation-framework/data/weather_test_data.csv"
# NUM_TEST_CASES_TO_CREATE = 50 # Let's create a reasonable number of test cases
# INPUT_WINDOW_SIZE = 3 # From conv_window in tutorial
# LABEL_WINDOW_SIZE = 1 # From conv_window in tutorial
# SHIFT = 1 # From conv_window in tutorial

# def main():
#     """
#     Loads the processed time series test data, creates TestCases, and seeds the database.
#     """
#     print(f"--- Seeding Time Series Test Cases ---")

#     # 1. Load the dataset
#     if not os.path.exists(DATA_FILE):
#         print(f"  ❌ Error: Test data file not found at {DATA_FILE}")
#         print("  Please ensure ml_eval/core/implementations/tensorFlowtutorial.py has been run at least once to generate it.")
#         return

#     df_test = pd.read_csv(DATA_FILE)
#     print(f"Test dataset loaded. Shape: {df_test.shape}")

#     # 2. Create and seed TestCases
#     total_created = 0
#     # Iterate to create test cases from the test dataframe
#     # The last possible start index for a full window + label
#     max_start_index = len(df_test) - (INPUT_WINDOW_SIZE + SHIFT + LABEL_WINDOW_SIZE - 1)

#     for i in range(min(NUM_TEST_CASES_TO_CREATE, max_start_index + 1)):
#         input_start = i
#         input_end = input_start + INPUT_WINDOW_SIZE
#         label_start = input_end + SHIFT -1
#         label_end = label_start + LABEL_WINDOW_SIZE

#         if label_end > len(df_test):
#             print(f"  Warning: Reached end of data at index {i}. Stopping test case creation.")
#             break

#         input_data_window = df_test.iloc[input_start:input_end].values.tolist()
#         ground_truth_window = df_test.iloc[label_start:label_end].values.tolist()

#         payload = {
#             "test_case_name": f"Time Series Prediction for Window {i+1}",
#             "model_type": "time_series_keras", # This will be the model_type for our new models
#             "input_type": "time_series_window",
#             "output_type": "time_series_prediction",
#             "input_data": {"window": input_data_window},
#             "ground_truth": {"prediction": ground_truth_window},
#             "category": "weather_forecasting",
#             "tags": ["keras", "time_series", "weather"],
#             "origin": "human",
#             "is_verified": True,
#             "test_case_metadata": {"input_window_size": INPUT_WINDOW_SIZE, "prediction_horizon": LABEL_WINDOW_SIZE}
#         }

#         try:
#             response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
#             if response.status_code == 200:
#                 total_created += 1
#                 if total_created % 10 == 0:
#                     print(f"  ...created {total_created} TestCases so far...")
#             else:
#                 print(f"  ❌ Failed to create TestCase {i+1}. Status: {response.status_code}, Response: {response.text}")
#         except requests.exceptions.ConnectionError:
#             print(f"\n❌ Error: Could not connect to the API at {API_URL}.")
#             print("  Please ensure your FastAPI application (uvicorn) is running.")
#             return
#         except Exception as e:
#             print(f"  ❌ An unexpected error occurred for TestCase {i+1}: {e}")
#             return


#     print(f"\n🎉 Seeding complete. Created {total_created} new Time Series TestCases.")

# if __name__ == "__main__":
#     main()
import os
import requests
import json
import pandas as pd

# --- Configuration ---
API_URL = "http://localhost:8000/api/v1/prompts/"
DATA_FILE = "/home/dell-linux-dev3/Projects/ml-evaluation-framework/data/weather_test_data.csv"
NUM_TEST_CASES_TO_CREATE = 50
INPUT_WINDOW_SIZE = 3
LABEL_WINDOW_SIZE = 1
SHIFT = 1
TEMPERATURE_COLUMN = 'T (degC)'  # Column to predict

def main():
    """
    Loads the processed time series test data, creates TestCases, and seeds the database.
    """
    print(f"--- Seeding Time Series Test Cases ---")
    
    # 1. Load the dataset
    if not os.path.exists(DATA_FILE):
        print(f"  ❌ Error: Test data file not found at {DATA_FILE}")
        print("  Please ensure ml_eval/core/implementations/tensorFlowtutorial.py has been run at least once to generate it.")
        return
    
    df_test = pd.read_csv(DATA_FILE)
    print(f"Test dataset loaded. Shape: {df_test.shape}")
    print(f"Predicting column: {TEMPERATURE_COLUMN}")
    
    # 2. Create and seed TestCases
    total_created = 0
    max_start_index = len(df_test) - (INPUT_WINDOW_SIZE + SHIFT + LABEL_WINDOW_SIZE - 1)
    
    for i in range(min(NUM_TEST_CASES_TO_CREATE, max_start_index + 1)):
        input_start = i
        input_end = input_start + INPUT_WINDOW_SIZE
        label_start = input_end + SHIFT - 1
        label_end = label_start + LABEL_WINDOW_SIZE
        
        if label_end > len(df_test):
            print(f"  Warning: Reached end of data at index {i}. Stopping test case creation.")
            break
        
        # Input: ALL features for the window
        input_data_window = df_test.iloc[input_start:input_end].values.tolist()
        
        # Ground truth: ONLY temperature column for prediction
        ground_truth_window = df_test.iloc[label_start:label_end][[TEMPERATURE_COLUMN]].values.tolist()
        
        payload = {
            "test_case_name": f"Time Series Prediction for Window {i+1}",
            "model_type": "time_series_keras",
            "input_type": "time_series_window",
            "output_type": "time_series_prediction",
            "input_data": {"window": input_data_window},
            "ground_truth": {"prediction": ground_truth_window},
            "category": "weather_forecasting",
            "tags": ["keras", "time_series", "weather"],
            "origin": "human",
            "is_verified": True,
            "test_case_metadata": {
                "input_window_size": INPUT_WINDOW_SIZE,
                "prediction_horizon": LABEL_WINDOW_SIZE,
                "target_column": TEMPERATURE_COLUMN
            }
        }
        
        try:
            response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
            if response.status_code == 200:
                total_created += 1
                if total_created % 10 == 0:
                    print(f"  ...created {total_created} TestCases so far...")
            else:
                print(f"  ❌ Failed to create TestCase {i+1}. Status: {response.status_code}, Response: {response.text}")
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Error: Could not connect to the API at {API_URL}.")
            print("  Please ensure your FastAPI application (uvicorn) is running.")
            return
        except Exception as e:
            print(f"  ❌ An unexpected error occurred for TestCase {i+1}: {e}")
            return
    
    print(f"\n🎉 Successfully created {total_created} TestCases with correct ground truth!")
   
if __name__ == "__main__":
    main()

