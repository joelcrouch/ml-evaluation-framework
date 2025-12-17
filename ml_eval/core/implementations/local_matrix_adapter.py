
from typing import Dict, Any
from ml_eval.core.interfaces.imodel import IModelAdapter
from ml_eval.core.implementations.matrix_model import MatrixMultiplicationModel

class LocalMatrixAdapter(IModelAdapter):
    """
    A concrete implementation of IModelAdapter for a local matrix multiplication model.
    """
    def __init__(self, model: MatrixMultiplicationModel):
        self.model = model

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the local matrix multiplication model with the given input.
        """
        return self.model.run(input)

