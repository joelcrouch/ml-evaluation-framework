
from abc import ABC, abstractmethod
from typing import Dict, Any

class IModelAdapter(ABC):
    """
    Abstract base class for a model adapter.
    A model adapter is responsible for interacting with a specific model.
    """

    @abstractmethod
    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the model with the given input.

        Args:
            input: A dictionary containing the input data for the model.

        Returns:
            A dictionary containing the model's output.
        """
        pass
