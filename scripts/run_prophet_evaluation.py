
import argparse
import sys
import os
import pandas as pd

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_eval.core.implementations.prophet_model import ProphetModel
from ml_eval.core.implementations.prophet_adapter import ProphetAdapter

def main():
    """
    Main function to run the prophet evaluation from the command line.
    """
    parser = argparse.ArgumentParser(description="Run a Prophet evaluation.")
    parser.add_argument("periods", type=int, help="The number of periods to forecast.")
    args = parser.parse_args()

    periods = args.periods

    print(f"--- Setting up evaluation for Prophet model with {periods} periods ---")

    # 1. Instantiate the model and adapter
    prophet_model = ProphetModel()
    model_adapter = ProphetAdapter(model=prophet_model)

    # 2. Run the prediction
    forecast = model_adapter.run(input_data={"periods": periods})
    
    # 3. Print the forecast
    df = pd.DataFrame(forecast)
    print("--- Forecast ---")
    print(df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

if __name__ == "__main__":
    main()
