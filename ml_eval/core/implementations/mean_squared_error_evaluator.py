
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from ml_eval.core.interfaces.ievaluator import IEvaluator, EvaluationResult

class MeanSquaredErrorEvaluator(IEvaluator):
    """
    An evaluator that calculates the Mean Squared Error (MSE) between
    the model's output (forecast) and the ground truth.
    
    The output is expected to be a list of dictionaries, each with a 'yhat' key.
    The ground truth is expected to be a list of dictionaries, each with a 'y' key.
    """

    def evaluate(self, output: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Calculates the MSE between the 'yhat' of the output and the 'y' of the ground truth.

        Args:
            output: A list of dictionaries, each containing the model's predicted 'yhat' value.
            ground_truth: A list of dictionaries, each containing the actual 'y' value.

        Returns:
            An EvaluationResult containing the score, pass/fail status, and metrics.
        """
        # Extract predicted and true values
        predicted_values = [item.get('yhat') for item in output]
        true_values = [item.get('y') for item in ground_truth]

        # Filter out None values and ensure all are numeric
        predicted_values = np.array([val for val in predicted_values if val is not None], dtype=float)
        true_values = np.array([val for val in true_values if val is not None], dtype=float)

        # Check for empty arrays after filtering
        if len(predicted_values) == 0 or len(true_values) == 0:
            return EvaluationResult(
                score=0.0,
                passed=False,
                metrics={},
                feedback="No valid predicted or true values found for evaluation."
            )

        # Ensure the lengths of the arrays are the same
        if len(predicted_values) != len(true_values):
            return EvaluationResult(
                score=0.0,
                passed=False,
                metrics={},
                feedback=f"Prediction and ground truth have different lengths: {len(predicted_values)} vs {len(true_values)}."
            )
            
        mse = np.mean((predicted_values - true_values) ** 2)

        # To get a score between 0 and 1, we can use a transformation.
        # A common approach for errors where lower is better is 1 / (1 + error).
        # We need to consider the scale of typical MSE values for this dataset.
        # For simplicity, let's assume a 'good' MSE is low, so higher score is better.
        # Let's cap the max MSE we expect for a "0 score" to avoid division by very small numbers,
        # or use a more dataset-specific normalization. For now, a generic inverse.
        # A better approach would be to define a "tolerance" or scale based on the data.
        
        # A simple score mapping: higher MSE means lower score
        # Let's set a maximum "acceptable" MSE, beyond which score is 0.
        # For AirPassengers, values are in hundreds, so MSE can be large.
        # A simple inverse might lead to very small scores for non-zero MSE.
        # For now, let's define a threshold for passing and return 1 - (mse / max_expected_mse)
        
        # For AirPassengers dataset, typical MSE values can be in the thousands or tens of thousands.
        # Let's consider a simple inverse relationship, or define a max error for scaling
        max_possible_error = np.max(true_values) ** 2 # max possible error if all predictions were 0
        if max_possible_error == 0: # Avoid division by zero if true_values are all zero
            score = 1.0 if mse == 0 else 0.0
        else:
            score = 1.0 - min(1.0, mse / max_possible_error) # Score decreases as MSE increases

        # Example threshold for passing: if score is above 0.7 (i.e., MSE is relatively low)
        # This threshold should ideally be determined by domain experts or historical performance.
        passed = score >= 0.7

        return EvaluationResult(
            score=float(score),
            passed=bool(passed),
            metrics={"mse": float(mse)},
            feedback=f"MSE: {mse:.2f}, Normalized Score: {score:.2f}"
        )
