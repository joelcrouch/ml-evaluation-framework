from typing import Dict, Any
from ml_eval.core.interfaces.imodel import IModelAdapter
from ml_eval.core.implementations.keras_time_series_model import KerasTimeSeriesModel

class KerasTimeSeriesAdapter(IModelAdapter):
    """
    Adapter for the KerasTimeSeriesModel, implementing IModelAdapter.
    """
    def __init__(self, model: KerasTimeSeriesModel):
        self.model = model

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the KerasTimeSeriesModel's predict method.
        """
        return self.model.predict(input_data)
