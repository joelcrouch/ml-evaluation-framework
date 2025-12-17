
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    """
    Represents the result of an evaluation.
    """
    score: float
    passed: bool
    metrics: Dict[str, Any]
    feedback: Optional[str] = None

class IEvaluator(ABC):
    """
    Abstract base class for an evaluator.
    An evaluator is responsible for assessing a model's output against a ground truth.
    """

    @abstractmethod
    def evaluate(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluates the model's output.

        Args:
            expected: The expected output (ground truth).
            actual: The actual output from the model.

        Returns:
            An EvaluationResult object.
        """
        pass
