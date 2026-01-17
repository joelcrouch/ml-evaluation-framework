
import pytest
from sqlalchemy.orm import Session
from ml_eval.database import crud
from ml_eval.database.models import Response # ADDED THIS LINE
from ml_eval.query_engine.engine import EvaluationEngine
from ml_eval.core.implementations.local_matrix_adapter import LocalMatrixAdapter
from ml_eval.core.implementations.matrix_model import MatrixMultiplicationModel
from ml_eval.core.implementations.exact_match import ExactMatchEvaluator

def test_run_evaluation_success(db_session: Session):
    """
    Test a successful end-to-end evaluation run.
    """
    # 1. Create a ModelRun
    model_run = crud.create_model_run(
        db=db_session,
        model_name="MatrixModel",
        model_version="1.0",
        model_type="matrix_multiplication",
        config={"dimensions": "2x2"},
    )

    # 2. Create some TestPrompts for matrix multiplication
    prompt_1_data = {
        "matrix_a": [[1, 2], [3, 4]],
        "matrix_b": [[5, 6], [7, 8]]
    }
    prompt_1_ground_truth = {"result_matrix": [[19, 22], [43, 50]]}
    prompt_1 = crud.create_prompt(
        db=db_session,
        test_case_name="Matrix Prompt 1",
        model_type="matrix_multiplication",
        input_type="json",
        output_type="json",
        input_data=prompt_1_data,
        ground_truth=prompt_1_ground_truth,
    )

    prompt_2_data = {
        "matrix_a": [[1, 0], [0, 1]],
        "matrix_b": [[10, 20], [30, 40]]
    }
    prompt_2_ground_truth = {"result_matrix": [[10, 20], [30, 40]]}
    prompt_2 = crud.create_prompt(
        db=db_session,
        test_case_name="Matrix Prompt 2",
        model_type="matrix_multiplication",
        input_type="json",
        output_type="json",
        input_data=prompt_2_data,
        ground_truth=prompt_2_ground_truth,
    )

    # 3. Instantiate model adapter and evaluator
    matrix_model = MatrixMultiplicationModel()
    model_adapter = LocalMatrixAdapter(model=matrix_model)
    evaluator = ExactMatchEvaluator()

    # 4. Instantiate and run EvaluationEngine
    engine = EvaluationEngine(db=db_session, model_adapter=model_adapter, evaluator=evaluator)
    completed_run = engine.run_evaluation(model_run.id)

    # 5. Assertions
    assert completed_run is not None
    assert completed_run.id == model_run.id
    assert completed_run.status == "completed"
    assert completed_run.completed_at is not None
    assert completed_run.total_cases == 2
    assert completed_run.completed_cases == 2
    assert completed_run.failed_cases == 0

    # Verify responses and evaluations
    responses = db_session.query(Response).filter(Response.run_id == model_run.id).all()
    assert len(responses) == 2

    for response in responses:
        assert response.output_data is not None
        evaluations = crud.get_evaluations_for_response(db_session, response.id)
        assert len(evaluations) == 1
        assert evaluations[0].evaluator_type == ExactMatchEvaluator.__name__
        assert evaluations[0].score == 1.0
        assert evaluations[0].passed is True

def test_run_evaluation_with_failure(db_session: Session):
    """
    Test an evaluation run where one prompt causes an error.
    """
    # 1. Create a ModelRun
    model_run = crud.create_model_run(
        db=db_session,
        model_name="MatrixModelWithFailure",
        model_version="1.1",
        model_type="matrix_multiplication",
        config={"dimensions": "2x2"},
    )

    # 2. Create one good prompt and one bad prompt (e.g., non-multipliable matrices)
    prompt_good_data = {
        "matrix_a": [[1, 2]],
        "matrix_b": [[3], [4]]
    }
    prompt_good_ground_truth = {"result_matrix": [[11]]}
    prompt_good = crud.create_prompt(
        db=db_session,
        test_case_name="Good Matrix Prompt",
        model_type="matrix_multiplication",
        input_type="json",
        output_type="json",
        input_data=prompt_good_data,
        ground_truth=prompt_good_ground_truth,
    )

    prompt_bad_data = { # Invalid dimensions
        "matrix_a": [[1, 2]],
        "matrix_b": [[3, 4]]
    }
    prompt_bad_ground_truth = {"result_matrix": []}
    prompt_bad = crud.create_prompt(
        db=db_session,
        test_case_name="Bad Matrix Prompt",
        model_type="matrix_multiplication",
        input_type="json",
        output_type="json",
        input_data=prompt_bad_data,
        ground_truth=prompt_bad_ground_truth,
    )

    # 3. Instantiate model adapter and evaluator
    matrix_model = MatrixMultiplicationModel()
    model_adapter = LocalMatrixAdapter(model=matrix_model)
    evaluator = ExactMatchEvaluator()

    # 4. Instantiate and run EvaluationEngine
    engine = EvaluationEngine(db=db_session, model_adapter=model_adapter, evaluator=evaluator)
    completed_run = engine.run_evaluation(model_run.id)

    # 5. Assertions
    assert completed_run is not None
    assert completed_run.id == model_run.id
    assert completed_run.status == "completed"
    assert completed_run.completed_at is not None
    assert completed_run.total_cases == 2
    assert completed_run.completed_cases == 1 # One good, one failed
    assert completed_run.failed_cases == 1

    # Verify responses and evaluations for the good prompt
    responses_good = db_session.query(Response).filter(Response.run_id == model_run.id, Response.test_case_id == prompt_good.id).all()
    assert len(responses_good) == 1
    assert responses_good[0].output_data == prompt_good_ground_truth # Should match
    evaluations_good = crud.get_evaluations_for_response(db_session, responses_good[0].id)
    assert len(evaluations_good) == 1
    assert evaluations_good[0].score == 1.0

    # Verify response for the bad prompt (should have error message)
    responses_bad = db_session.query(Response).filter(Response.run_id == model_run.id, Response.test_case_id == prompt_bad.id).all()
    assert len(responses_bad) == 1
    assert responses_bad[0].error_message is not None
    assert "Number of columns in Matrix A must be equal to number of rows in Matrix B" in responses_bad[0].error_message
    evaluations_bad = crud.get_evaluations_for_response(db_session, responses_bad[0].id)
    assert len(evaluations_bad) == 0 # No evaluation created for failed response
