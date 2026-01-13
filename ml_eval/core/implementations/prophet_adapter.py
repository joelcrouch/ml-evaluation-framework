
from typing import Dict, Any
from ml_eval.core.interfaces.imodel import IModelAdapter
from ml_eval.core.implementations.prophet_model import ProphetModel

class ProphetAdapter(IModelAdapter):
    """
    Adapter for the ProphetModel, implementing IModelAdapter.
    """
    def __init__(self, model: ProphetModel):
        self.model = model

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the ProphetModel's predict method.
        """
        return self.model.predict(input_data)
