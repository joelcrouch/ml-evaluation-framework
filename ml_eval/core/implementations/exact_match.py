
from typing import Dict, Any
from ml_eval.core.interfaces.ievaluator import IEvaluator, EvaluationResult

class ExactMatchEvaluator(IEvaluator):
    """
    A simple evaluator that checks for exact match between expected and actual outputs.
    """

    def evaluate(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> EvaluationResult:
        """
        Compares the expected and actual outputs for exact equality.

        Args:
            expected: The expected output.
            actual: The actual output from the model.

        Returns:
            An EvaluationResult object.
        """
        is_match = expected == actual
        score = 1.0 if is_match else 0.0
        
        return EvaluationResult(
            score=score,
            passed=is_match,
            metrics={"match": is_match}
        )
