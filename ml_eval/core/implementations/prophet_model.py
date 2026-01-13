
from prophet import Prophet
from prophet.serialize import model_from_json
import pandas as pd
from typing import Dict, Any, List
import os

class ProphetModel:
    """
    A time-series forecasting model using Facebook's Prophet library.
    Can either train a new model or load a pre-trained one.
    """

    def __init__(self, dataset_path: str = None, train_size: int = None, model_file_path: str = None):
        """
        Initializes the Prophet model.
        
        Args:
            dataset_path: Path to the time series dataset for training (if model_file_path is None).
            train_size: The number of data points to use for training. If None, use all data.
                        The remaining data points will be stored as ground truth.
            model_file_path: Path to a pre-trained Prophet model JSON file. If provided,
                             the model will be loaded from this file and dataset_path/train_size are ignored.
        """
        self.ground_truth_df = pd.DataFrame() # Initialize empty

        if model_file_path:
            print(f"--- Loading pre-trained Prophet model from {model_file_path} ---")
            if not os.path.exists(model_file_path):
                raise FileNotFoundError(f"Model file not found at: {model_file_path}")
            with open(model_file_path, 'r') as f:
                self.model = model_from_json(f.read())
            print(f"✅ Model loaded from {model_file_path}")
        else:
            if not dataset_path:
                raise ValueError("dataset_path must be provided if not loading a pre-trained model.")
            print("--- Initializing and training Prophet model... ---")
            self.model = Prophet()
            
            # Load and prepare data
            df = pd.read_csv(dataset_path)
            df['ds'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'value': 'y'})
            
            if train_size is None:
                train_df = df
            else:
                if train_size >= len(df):
                    raise ValueError("train_size cannot be greater than or equal to the total dataset length.")
                train_df = df.iloc[:train_size]
                self.ground_truth_df = df.iloc[train_size:] # Store ground truth if split

            # Train the model
            self.model.fit(train_df)
            print(f"✅ Model trained on {len(train_df)} data points from {dataset_path}")

    def get_ground_truth(self) -> List[Dict[str, Any]]:
        """
        Returns the ground truth data (test set) for evaluation.
        This is only populated if the model was trained with a train_size split
        and not loaded from a file.
        """
        if not self.ground_truth_df.empty:
            # Ensure 'ds' column is converted to string for JSON serialization
            ground_truth_for_json = self.ground_truth_df.copy()
            ground_truth_for_json['ds'] = ground_truth_for_json['ds'].apply(lambda x: x.isoformat())
            return ground_truth_for_json[['ds', 'y']].to_dict(orient='records')
        return []

    def predict(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Predicts future values.
        Expects 'periods' in input_data.
        """
        periods = input_data.get("periods")
        if not periods:
            raise ValueError("'periods' not found in input_data")

        # Make future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='MS')
        
        # Get prediction
        forecast = self.model.predict(future)
        
        # Convert 'ds' column to string for JSON serialization
        forecast['ds'] = forecast['ds'].apply(lambda x: x.isoformat())

        # Return only the future forecasted values
        return forecast.iloc[-periods:].to_dict(orient='records')

