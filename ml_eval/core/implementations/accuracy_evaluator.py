from typing import Dict, Any
from ml_eval.core.interfaces.ievaluator import IEvaluator, EvaluationResult

class AccuracyEvaluator(IEvaluator):
    """
    A simple evaluator that calculates accuracy for classification tasks.
    Compares expected["label"] with actual["predicted_label"].
    """

    def evaluate(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> EvaluationResult:
        """
        Calculates accuracy based on exact match of the predicted label.

        Args:
            expected: The expected output, e.g., {"label": "cat"}.
            actual: The actual output from the model, e.g., {"predicted_label": "cat"}.

        Returns:
            An EvaluationResult object with score, passed status, and metrics.
        """
        expected_label = expected.get("label")
        predicted_label = actual.get("predicted_label")

        is_correct = (expected_label == predicted_label)
        score = 1.0 if is_correct else 0.0

        return EvaluationResult(
            score=score,
            passed=is_correct,
            metrics={"is_correct": is_correct, "expected": expected_label, "actual": predicted_label}
        )
