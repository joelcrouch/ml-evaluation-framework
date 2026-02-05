import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Configuration ---
# Assuming the FastAPI service is running locally
API_URL = "http://localhost:8000/api/v1/prompts/"

def load_and_parse_test_suite(file_path: str):
    """
    Loads and parses a test suite from a given JSON file.
    Performs initial validation based on Story 1 Go/No-Go criteria.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test suite file not found at: {file_path}")

    # For now, we only support JSON as per our initial format definition.
    # Future enhancement will add YAML.
    try:
        with open(file_path, 'r') as f:
            test_cases = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file {file_path}: {e}")
    except Exception as e:
        raise Exception(f"Error reading or parsing file {file_path}: {e}")

    if not isinstance(test_cases, list):
        raise ValueError(f"Test suite file {file_path} must contain a top-level JSON array (list of test cases).")

    return test_cases

def main():
    parser = argparse.ArgumentParser(description="Load a test suite into the evaluation platform.")
    parser.add_argument("file_path", type=str, help="Path to the test suite JSON file.")
    args = parser.parse_args()

    try:
        test_cases = load_and_parse_test_suite(args.file_path)
        
        if not test_cases:
            print(f"⚠️  Warning: Test suite file '{args.file_path}' is empty. No test cases to load.")
            sys.exit(0)

        print(f"✅ Successfully parsed {len(test_cases)} test cases from '{args.file_path}'.")
        print("--- (Next step: Implement database interaction to add these test cases) ---")
        # In future steps, this is where we'd add database interaction.
        # For now, just confirming parsing success.

    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"❌ Error loading test suite: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()