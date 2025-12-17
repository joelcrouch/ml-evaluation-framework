from typing import Dict, Any
from ml_eval.core.interfaces.imodel import IModelAdapter
from ml_eval.core.implementations.image_classifier_model import ImageClassifierModel

class ImageClassifierAdapter(IModelAdapter):
    """
    Adapter for the ImageClassifierModel, implementing IModelAdapter.
    """
    def __init__(self, model: ImageClassifierModel):
        self.model = model

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the ImageClassifierModel's predict method.
        """
        return self.model.predict(input_data)
