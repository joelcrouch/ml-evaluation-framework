# scripts/seed_multistep_dense.py
import os
import json
import requests
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
API_URL = "http://localhost:8000/api/v1/prompts/"
GOLDEN_DATASET_PATH = "data/multistep_dense_golden_dataset.json"

def main():
    """
    Seeds the database with multi-step dense time series test cases from the golden dataset.
    """
    print(f"--- Seeding Multi-step Dense Test Cases from Golden Dataset ---")
    
    # Load golden dataset
    if not os.path.exists(GOLDEN_DATASET_PATH):
        print(f"❌ Error: Golden dataset not found at {GOLDEN_DATASET_PATH}")
        print("Please run the training script first to generate it:")
        print("  python scripts/train_multistep_dense.py")
        return
    
    with open(GOLDEN_DATASET_PATH, 'r') as f:
        golden_cases = json.load(f)
    
    print(f"Loaded {len(golden_cases)} test cases from golden dataset")
    
    # Seed test cases
    total_created = 0
    failed = 0
    
    for case in golden_cases:
        case_id = case['case_id']
        metadata = case['metadata']
        
        # Create payload for API
        payload = {
            "test_case_name": f"Multi-step Dense TS Prediction - Case {case_id}",
            "model_type": "time_series_multistep_dense",
            "input_type": "time_series_window",
            "output_type": "temperature_prediction",
            "input_data": case['input_data'],
            "ground_truth": case['ground_truth'],
            "category": "weather_forecasting",
            "tags": ["dense", "multistep", "flatten", "relu", "time_series"],
            "origin": "golden_dataset",
            "is_verified": True,
            "test_case_metadata": metadata
        }
        
        try:
            response = requests.post(
                API_URL, 
                headers={"Content-Type": "application/json"}, 
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                total_created += 1
                if total_created % 50 == 0 and len(golden_cases) > 100:
                    print(f"  ...created {total_created} test cases so far...")
            elif response.status_code == 409:
                print(f"  ⚠️  Test case {case_id} may already exist, skipping...")
                total_created +=1
            else:
                failed += 1
                print(f"  ❌ Failed to create case {case_id}. Status: {response.status_code}")
                if failed <= 3:
                    print(f"     Response: {response.text}")
                    
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Error: Could not connect to API at {API_URL}")
            print("Please ensure your FastAPI application is running:")
            print("  uvicorn ml_eval.main:app --reload")
            return
        except Exception as e:
            failed += 1
            print(f"  ❌ Unexpected error for case {case_id}: {e}")
    
    print(f"\n={'='*60}")
    print(f"Seeding Complete!")
    print(f"  ✅ Handled: {total_created}/{len(golden_cases)} test cases")
    if failed > 0:
        print(f"  ❌ Failed: {failed}/{len(golden_cases)} test cases")
    print(f"{'='*60}")
    
    # curl -X 'POST' 'http://localhost:8000/api/v1/runs/' -H 'Content-Type: application/json' -d '{"model_name": "multistep_dense_model", "model_version": "1.0", "model_type": "time_series_multistep_dense"}'


    if total_created > 0:
        print("\nNext steps:")
        # Fix for the SyntaxError: unterminated string literal
        print("  1. Create a model run:\n     curl -X 'POST' 'http://localhost:8000/api/v1/runs/' \
       -H 'Content-Type: application/json' \
       -d '{\"model_name\": \"multistep_dense_model\", \"model_version\": \"1.0\", \"model_type\": \"time_series_multistep_dense\"}'")
        print("\n  2. Run evaluation:")
        print("     python scripts/run_evaluation.py <RUN_ID>")
        print("\n  3. Generate report:")
        print("     python scripts/gemerate_report_time_series_v3.py <RUN_ID>")

if __name__ == "__main__":
    main()
