
from typing import Dict, Any
from ml_eval.core.interfaces.imodel import IModelAdapter

class SimpleModelAdapter(IModelAdapter):
    """
    A simple dummy implementation of the IModelAdapter.
    It returns a mocked output.
    """

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a mocked output that includes the original input
        and adds a 'processed' flag.

        Args:
            input: The input data.

        Returns:
            A dictionary with the mocked output.
        """
        output = input.copy()
        output["processed"] = True
        return output
