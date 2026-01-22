
# import numpy as np
# import pandas as pd
# from typing import Dict, Any, List
# from ml_eval.core.interfaces.ievaluator import IEvaluator, EvaluationResult

# class MeanSquaredErrorEvaluator(IEvaluator):
#     """
#     An evaluator that calculates the Mean Squared Error (MSE) between
#     the model's output (forecast) and the ground truth.
    
#     The output is expected to be a list of dictionaries, each with a 'yhat' key.
#     The ground truth is expected to be a list of dictionaries, each with a 'y' key.
#     """

#     def evaluate(self, output: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> EvaluationResult:
#         """
#         Calculates the MSE between the 'yhat' of the output and the 'y' of the ground truth.

#         Args:
#             output: A list of dictionaries, each containing the model's predicted 'yhat' value.
#             ground_truth: A list of dictionaries, each containing the actual 'y' value.

#         Returns:
#             An EvaluationResult containing the score, pass/fail status, and metrics.
#         """
#         # Extract predicted and true values
#         predicted_values = [item.get('yhat') for item in output]
#         true_values = [item.get('y') for item in ground_truth]

#         # Filter out None values and ensure all are numeric
#         predicted_values = np.array([val for val in predicted_values if val is not None], dtype=float)
#         true_values = np.array([val for val in true_values if val is not None], dtype=float)

#         # Check for empty arrays after filtering
#         if len(predicted_values) == 0 or len(true_values) == 0:
#             return EvaluationResult(
#                 score=0.0,
#                 passed=False,
#                 metrics={},
#                 feedback="No valid predicted or true values found for evaluation."
#             )

#         # Ensure the lengths of the arrays are the same
#         if len(predicted_values) != len(true_values):
#             return EvaluationResult(
#                 score=0.0,
#                 passed=False,
#                 metrics={},
#                 feedback=f"Prediction and ground truth have different lengths: {len(predicted_values)} vs {len(true_values)}."
#             )
            
#         mse = np.mean((predicted_values - true_values) ** 2)

#         # To get a score between 0 and 1, we can use a transformation.
#         # A common approach for errors where lower is better is 1 / (1 + error).
#         # We need to consider the scale of typical MSE values for this dataset.
#         # For simplicity, let's assume a 'good' MSE is low, so higher score is better.
#         # Let's cap the max MSE we expect for a "0 score" to avoid division by very small numbers,
#         # or use a more dataset-specific normalization. For now, a generic inverse.
#         # A better approach would be to define a "tolerance" or scale based on the data.
        
#         # A simple score mapping: higher MSE means lower score
#         # Let's set a maximum "acceptable" MSE, beyond which score is 0.
#         # For AirPassengers, values are in hundreds, so MSE can be large.
#         # A simple inverse might lead to very small scores for non-zero MSE.
#         # For now, let's define a threshold for passing and return 1 - (mse / max_expected_mse)
        
#         # For AirPassengers dataset, typical MSE values can be in the thousands or tens of thousands.
#         # Let's consider a simple inverse relationship, or define a max error for scaling
#         max_possible_error = np.max(true_values) ** 2 # max possible error if all predictions were 0
#         if max_possible_error == 0: # Avoid division by zero if true_values are all zero
#             score = 1.0 if mse == 0 else 0.0
#         else:
#             score = 1.0 - min(1.0, mse / max_possible_error) # Score decreases as MSE increases

#         # Example threshold for passing: if score is above 0.7 (i.e., MSE is relatively low)
#         # This threshold should ideally be determined by domain experts or historical performance.
#         passed = score >= 0.7

#         return EvaluationResult(
#             score=float(score),
#             passed=bool(passed),
#             metrics={"mse": float(mse)},
#             feedback=f"MSE: {mse:.2f}, Normalized Score: {score:.2f}"
#         )

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from ml_eval.core.interfaces.ievaluator import IEvaluator, EvaluationResult

class MeanSquaredErrorEvaluator(IEvaluator):
    """
    An evaluator that calculates the Mean Squared Error (MSE) between
    the model's output (forecast) and the ground truth.
    
    Supports multiple input formats:
    - Legacy format: List of dicts with 'yhat' and 'y' keys
    - New format: Dicts with 'prediction'/'predictions' and 'expected_output' keys
    - Raw numpy arrays
    """

    def evaluate(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> EvaluationResult:
        """
        Calculates the MSE between actual predictions and expected values.

        Args:
            actual: Dictionary from model adapter containing predictions
                   Can have keys: 'prediction', 'predictions', or be a list with 'yhat'
            expected: Dictionary from test case containing expected values
                     Can have keys: 'expected_output', 'expected', 'target', or be a list with 'y'

        Returns:
            An EvaluationResult containing the score, pass/fail status, and metrics.
        """
        try:
            # Extract predicted values from actual output
            predicted_values = self._extract_predictions(actual)
            
            # Extract true values from expected output
            true_values = self._extract_ground_truth(expected)
            
            # Validate extracted values
            if predicted_values is None or true_values is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    metrics={},
                    feedback="Could not extract predictions or ground truth from input data."
                )
            
            # Convert to numpy arrays
            predicted_values = np.array(predicted_values, dtype=float).flatten()
            true_values = np.array(true_values, dtype=float).flatten()

            # Check for empty arrays
            if len(predicted_values) == 0 or len(true_values) == 0:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    metrics={},
                    feedback="No valid predicted or true values found for evaluation."
                )

            # Ensure the lengths match
            if len(predicted_values) != len(true_values):
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    metrics={},
                    feedback=f"Prediction and ground truth have different lengths: {len(predicted_values)} vs {len(true_values)}."
                )
                
            # Calculate MSE
            mse = np.mean((predicted_values - true_values) ** 2)
            mae = np.mean(np.abs(predicted_values - true_values))
            rmse = np.sqrt(mse)

            # Score calculation
            max_possible_error = np.max(true_values) ** 2
            if max_possible_error == 0:
                score = 1.0 if mse == 0 else 0.0
            else:
                score = 1.0 - min(1.0, mse / max_possible_error)

            # Get threshold from expected data or use default
            threshold = expected.get('score_threshold', 0.7)
            passed = score >= threshold

            return EvaluationResult(
                score=float(score),
                passed=bool(passed),
                metrics={
                    "mse": float(mse),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "threshold": float(threshold),
                    "num_predictions": len(predicted_values)
                },
                feedback=f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, Score: {score:.4f}"
            )
            
        except Exception as e:
            return EvaluationResult(
                score=0.0,
                passed=False,
                metrics={"error": str(e)},
                feedback=f"Error during evaluation: {str(e)}"
            )
    
    def _extract_predictions(self, actual: Dict[str, Any]) -> np.ndarray:
        """Extract predictions from various formats."""
        # Format 1: Direct prediction key
        if 'prediction' in actual:
            return actual['prediction']
        
        # Format 2: Predictions (plural)
        if 'predictions' in actual:
            return actual['predictions']
        
        # Format 3: List of dicts with 'yhat' key (legacy format)
        if isinstance(actual, list) and len(actual) > 0 and isinstance(actual[0], dict):
            if 'yhat' in actual[0]:
                return [item.get('yhat') for item in actual if item.get('yhat') is not None]
        
        # Format 4: Direct numpy array or list
        if isinstance(actual, (list, np.ndarray)):
            return actual
        
        return None
    
    def _extract_ground_truth(self, expected: Dict[str, Any]) -> np.ndarray:
        """Extract ground truth from various formats."""
        # Format 1: expected_output key
        if 'expected_output' in expected:
            return expected['expected_output']
        
        # Format 2: expected key
        if 'expected' in expected:
            return expected['expected']
        
        # Format 3: target key
        if 'target' in expected:
            return expected['target']
        
        # Format 4: ground_truth key
        if 'ground_truth' in expected:
            return expected['ground_truth']
        
        # Format 5: prediction key (for backwards compatibility)
        if 'prediction' in expected:
            return expected['prediction']
        
        # Format 5: List of dicts with 'y' key (legacy format)
        if isinstance(expected, list) and len(expected) > 0 and isinstance(expected[0], dict):
            if 'y' in expected[0]:
                return [item.get('y') for item in expected if item.get('y') is not None]
        
        # Format 6: Direct numpy array or list
        if isinstance(expected, (list, np.ndarray)):
            return expected
        
        return None
