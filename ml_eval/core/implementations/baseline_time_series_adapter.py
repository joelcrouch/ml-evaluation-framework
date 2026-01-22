# ml_eval/core/adapters/baseline_time_series_adapter.py
from typing import Dict, Any
from ml_eval.core.interfaces.imodel import IModelAdapter
from ml_eval.core.implementations.baseline_time_series_model import BaselineTimeSeriesModel

class BaselineTimeSeriesAdapter(IModelAdapter):
    """
    Adapter for the BaselineTimeSeriesModel, implementing IModelAdapter.
    """
    
    def __init__(self, model: BaselineTimeSeriesModel):
        self.model = model
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the BaselineTimeSeriesModel's predict method.
        
        Args:
            input_data: Dictionary containing the input window
        
        Returns:
            Dictionary containing the prediction
        """
        return self.model.predict(input_data)