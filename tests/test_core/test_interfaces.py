
import pytest
from ml_eval.core.interfaces.imodel import IModelAdapter
from ml_eval.core.interfaces.ievaluator import IEvaluator, EvaluationResult
from ml_eval.core.implementations.simple_model import SimpleModelAdapter
from ml_eval.core.implementations.exact_match import ExactMatchEvaluator

def test_simple_model_adapter_adheres_to_interface():
    """Tests that SimpleModelAdapter correctly implements IModelAdapter."""
    assert issubclass(SimpleModelAdapter, IModelAdapter)
    adapter = SimpleModelAdapter()
    assert isinstance(adapter, IModelAdapter)

def test_simple_model_adapter_run():
    """Tests the run method of the SimpleModelAdapter."""
    adapter = SimpleModelAdapter()
    input_data = {"text": "hello"}
    output = adapter.run(input_data)
    expected_output = {"text": "hello", "processed": True}
    assert output == expected_output

def test_exact_match_evaluator_adheres_to_interface():
    """Tests that ExactMatchEvaluator correctly implements IEvaluator."""
    assert issubclass(ExactMatchEvaluator, IEvaluator)
    evaluator = ExactMatchEvaluator()
    assert isinstance(evaluator, IEvaluator)

def test_exact_match_evaluator_evaluate_match():
    """Tests the evaluate method of ExactMatchEvaluator for a match."""
    evaluator = ExactMatchEvaluator()
    expected = {"text": "world"}
    actual = {"text": "world"}
    result = evaluator.evaluate(expected, actual)
    assert isinstance(result, EvaluationResult)
    assert result.passed is True
    assert result.score == 1.0

def test_exact_match_evaluator_evaluate_no_match():
    """Tests the evaluate method of ExactMatchEvaluator for a mismatch."""
    evaluator = ExactMatchEvaluator()
    expected = {"text": "hello"}
    actual = {"text": "world"}
    result = evaluator.evaluate(expected, actual)
    assert isinstance(result, EvaluationResult)
    assert result.passed is False
    assert result.score == 0.0
