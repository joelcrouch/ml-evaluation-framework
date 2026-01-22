# ml_eval/core/implementations/baseline_time_series_model.py
import tensorflow as tf
import numpy as np
from typing import Dict, Any

@tf.keras.utils.register_keras_serializable(package="TimeSeriesModels")
class Baseline(tf.keras.Model):
    """Custom Baseline model class for serialization."""
    def __init__(self, label_index=None, **kwargs):
        super().__init__(**kwargs)
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
    
    def get_config(self):
        config = super().get_config()
        config.update({'label_index': self.label_index})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BaselineTimeSeriesModel:
    """
    A baseline time series model that predicts temperature one hour into the future.
    Uses the simple "no change" baseline - predicts current temperature as next temperature.
    """
    
    def __init__(self, model_path: str = 'models/baseline_model.keras'):
        """
        Initializes the model by loading the trained Keras baseline model.
        
        Args:
            model_path: Path to the saved .keras model file
        """
        print(f"--- Loading baseline time series model from {model_path}... ---")
        
        # Load with custom objects so Keras knows about our Baseline class
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'Baseline': Baseline}
        )
        
        print("✅ Baseline model loaded successfully")
        
    def _preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocesses input data into the format expected by the model.
        
        Expected input_data format:
        {
            "window": [[feat1, feat2, ..., feat19], [feat1, feat2, ..., feat19], ...]
        }
        
        Returns:
            Numpy array of shape (batch_size, timesteps, features)
        """
        window = input_data.get("window")
        if window is None:
            raise ValueError("'window' not found in input_data")
        
        # Convert to numpy array
        input_array = np.array(window, dtype=np.float32)
        
        # Ensure proper shape: (batch, timesteps, features)
        if len(input_array.shape) == 2:
            # Shape is (timesteps, features) - add batch dimension
            input_array = np.expand_dims(input_array, axis=0)
        elif len(input_array.shape) != 3:
            raise ValueError(f"Invalid input shape: {input_array.shape}. Expected (timesteps, features) or (batch, timesteps, features)")
        
        return input_array
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts temperature one timestep into the future.
        
        Args:
            input_data: Dictionary containing:
                - "window": List of lists representing timesteps × features
        
        Returns:
            Dictionary containing:
                - "prediction": The predicted temperature value(s)
        """
        # Preprocess input
        preprocessed_input = self._preprocess_input(input_data)
        
        # Make prediction
        # Model expects shape: (batch, timesteps, features)
        # Model returns shape: (batch, timesteps, 1) for temperature
        predictions = self.model.predict(preprocessed_input, verbose=0)
        
        # Convert to list for JSON serialization
        prediction_list = predictions.tolist()
        
        return {
            "prediction": prediction_list,
            "metadata": {
                "input_shape": preprocessed_input.shape,
                "output_shape": predictions.shape,
                "model_type": "baseline_time_series"
            }
        }