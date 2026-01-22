# import tensorflow as tf
# import numpy as np
# from typing import Dict, Any, List

# class KerasTimeSeriesModel:
#     """
#     A generic Keras model loader for time series prediction.
#     """

#     def __init__(self, model_path: str):
#         """
#         Initializes the model by loading a pre-trained Keras model from the given path.
#         """
#         print(f"--- Loading trained Keras time series model from: {model_path} ---")
#         self.model = tf.keras.models.load_model(model_path)
#         print(f"✅ Model loaded successfully.")

#     def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Predicts the next value(s) in a time series from a window of previous values.
#         Expects 'window' in input_data, which is a List of Lists (the input window).
#         """
#         window = input_data.get("window")
#         if not window:
#             raise ValueError("'window' not found in input_data")

#         # Convert the input window (list of lists) to a NumPy array
#         # and add a batch dimension.
#         input_array = np.array(window)
#         input_array = np.expand_dims(input_array, axis=0) # Shape: (1, window_size, num_features)

#         # Get model prediction
#         predictions = self.model.predict(input_array)
        
#         # The output shape will be something like (1, horizon, num_features)
#         # We'll return the raw prediction array, converted to a list.
#         return {"prediction": predictions.tolist()}

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union

class KerasTimeSeriesModel:
    """
    A generic Keras model loader for time series prediction.
    Supports multiple input formats and works with all Keras time series models.
    """
    def __init__(self, model_path: str, window_config: Dict[str, Any] = None):
        """
        Initializes the model by loading a pre-trained Keras model from the given path.
        
        Args:
            model_path: Path to the .keras model file
            window_config: Optional configuration for input validation
                          (input_width, label_width, num_features, etc.)
        """
        print(f"--- Loading trained Keras time series model from: {model_path} ---")
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully.")
        
        self.window_config = window_config or {}
        self.input_shape = self.model.input_shape  # e.g., (None, timesteps, features)
        self.output_shape = self.model.output_shape
        
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts the next value(s) in a time series from a window of previous values.
        
        Supports multiple input formats:
        - 'window': List of lists [[val1, val2, ...], [val1, val2, ...], ...]
        - 'features': numpy array or pandas DataFrame
        - 'sequence': raw time series data
        
        Args:
            input_data: Dictionary containing input data in one of the supported formats
            
        Returns:
            Dictionary with 'prediction' (list) and optional 'metadata'
        """
        try:
            # Extract input in various formats (flexible)
            input_array = self._prepare_input(input_data)
            
            # Validate shape if window_config is provided
            if self.window_config:
                self._validate_input_shape(input_array)
            
            # Get model prediction
            predictions = self.model.predict(input_array, verbose=0)
            
            # Return results with metadata
            return {
                "prediction": predictions.tolist(),
                "metadata": {
                    "input_shape": input_array.shape,
                    "output_shape": predictions.shape,
                    "model_input_shape": self.input_shape,
                    "model_output_shape": self.output_shape
                }
            }
            
        except Exception as e:
            return {
                "prediction": None,
                "error": str(e),
                "metadata": {
                    "status": "failed",
                    "error_type": type(e).__name__
                }
            }
    
    def _prepare_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Prepares input data into the correct format for model prediction.
        
        Handles multiple input formats for flexibility.
        """
        # Try different input keys
        if "window" in input_data:
            data = input_data["window"]
        elif "features" in input_data:
            data = input_data["features"]
        elif "sequence" in input_data:
            data = input_data["sequence"]
        else:
            raise ValueError(
                "Input data must contain one of: 'window', 'features', or 'sequence'"
            )
        
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            input_array = data.values
        elif isinstance(data, list):
            input_array = np.array(data)
        elif isinstance(data, np.ndarray):
            input_array = data
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")
        
        # Ensure correct dimensionality
        # Expected: (batch, timesteps, features)
        if len(input_array.shape) == 1:
            # Shape: (timesteps,) -> (1, timesteps, 1)
            input_array = input_array.reshape(1, -1, 1)
        elif len(input_array.shape) == 2:
            # Shape: (timesteps, features) -> (1, timesteps, features)
            input_array = np.expand_dims(input_array, axis=0)
        elif len(input_array.shape) == 3:
            # Already correct shape: (batch, timesteps, features)
            pass
        else:
            raise ValueError(
                f"Input array has invalid shape: {input_array.shape}. "
                f"Expected 1D, 2D, or 3D array."
            )
        
        return input_array.astype(np.float32)
    
    def _validate_input_shape(self, input_array: np.ndarray):
        """
        Validates input shape against expected dimensions from window_config.
        """
        expected_timesteps = self.window_config.get('input_width')
        expected_features = self.window_config.get('num_features')
        
        actual_timesteps = input_array.shape[1]
        actual_features = input_array.shape[2]
        
        if expected_timesteps and actual_timesteps != expected_timesteps:
            raise ValueError(
                f"Input timesteps mismatch. Expected {expected_timesteps}, "
                f"got {actual_timesteps}"
            )
        
        if expected_features and actual_features != expected_features:
            raise ValueError(
                f"Input features mismatch. Expected {expected_features}, "
                f"got {actual_features}"
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the loaded model.
        """
        return {
            "model_type": "keras_time_series",
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "total_params": self.model.count_params(),
            "trainable_params": sum([np.prod(v.shape) for v in self.model.trainable_weights]),
            "window_config": self.window_config
        }
    
    def batch_predict(self, input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Performs batch prediction on multiple input samples.
        
        Args:
            input_data_list: List of input data dictionaries
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        for input_data in input_data_list:
            results.append(self.predict(input_data))
        return results
