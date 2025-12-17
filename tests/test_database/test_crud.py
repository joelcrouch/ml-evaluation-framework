import pytest
from datetime import datetime
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# Assuming a fixture setup for the database session
# You would typically define 'test_db' fixture in conftest.py
# For this example, we mock a database dependency:
from ml_eval.database.crud import (
    create_prompt, get_prompt, get_prompts_by_model_type, get_model_run,
    create_model_run, complete_model_run,
    create_response,
    create_evaluation, get_evaluations_for_response
)
from ml_eval.database.connection import SessionLocal
from ml_eval.database.models import Base

# --- Pytest Fixtures Setup ---
# You need a way to connect to your test database.
# In a real project, this would use a dedicated test database URL.

@pytest.fixture(scope="session")
def db_engine():
    """Engine used across all tests in the session."""
    # Use the existing engine, but we'll manage the lifecycle
    return SessionLocal().bind

@pytest.fixture(scope="function")
def test_db(db_engine):
    """Provides a fresh database session for each test."""
    # 1. Create all tables before the test runs
    Base.metadata.create_all(bind=db_engine)
    
    # 2. Start the session
    connection = db_engine.connect()
    db = Session(bind=connection)

    yield db # Give the session to the test

    # 3. Teardown: Close session and drop tables after the test
    db.close()
    Base.metadata.drop_all(bind=db_engine) # Clean slate for next test

# --- Test Functions ---

def test_create_prompt_with_defaults(test_db: Session):
    """Test creating a prompt and verifies user-first defaults and JSONB storage."""
    input_data = {"text": "Hello"}
    ground_truth = {"text": "World"}
    
    prompt = create_prompt(
        test_db, 
        test_case_name="Test Prompt 1", 
        model_type="NLP", 
        input_type="text",
        output_type="classification",
        input_data=input_data, 
        ground_truth=ground_truth,
        test_case_metadata={"author": "test"}
    )
    
    assert prompt.id is not None
    assert prompt.model_type == "NLP"
    assert prompt.origin == "human"  # Check User-First Default
    assert prompt.is_verified is True # Check User-First Default
    assert prompt.input_data == input_data
    assert prompt.test_case_metadata == {"author": "test"}
    
    # Test read operation
    fetched_prompt = get_prompt(test_db, prompt.id)
    assert fetched_prompt.test_case_name == "Test Prompt 1"
    assert fetched_prompt.test_case_metadata == {"author": "test"}


def test_model_run_lifecycle(test_db: Session):
    """Test creation and completion of a ModelRun."""
    run = create_model_run(test_db, "ResNet50", "v1.2", "CV", model_endpoint="http://localhost:8001/vision", config={"param": 1})
    assert run.model_name == "ResNet50"
    assert run.model_type == "CV"
    assert run.status == "pending"
    assert run.completed_at is None
    
    completed_run = complete_model_run(test_db, run.id)
    assert completed_run.completed_at is not None
    assert completed_run.completed_at >= run.started_at
    assert completed_run.status == "completed"


def test_end_to_end_data_flow_and_foreign_keys(test_db: Session):
    """Tests the entire data pipeline and foreign key relationships."""
    
    # 1. Create foundational records
    prompt = create_prompt(
        test_db, 
        test_case_name="FK Test", 
        model_type="CV", 
        input_type="image_path", 
        output_type="classification", 
        input_data={"path": "img"}, 
        ground_truth={"label": "cat"},
        test_case_metadata={"source": "auto"}
    )
    run = create_model_run(test_db, "VisionModel", "v1.0", "CV")
    
    # 2. Create Response (Links Prompt and Run)
    output_data = {"predictions": [{"label": "cat", "conf": 0.9}]}
    response = create_response(
        test_db, 
        run_id=run.id, 
        test_case_id=prompt.id, 
        output_data=output_data,
        latency_ms=100
    )
    assert response.test_case_id == prompt.id
    assert response.run_id == run.id
    assert response.output_data == output_data
    assert response.latency_ms == 100
    
    # Test relationship access (SQLAlchemy ORM)
    assert response.prompt.test_case_name == "FK Test" 
    assert response.model_run.model_version == "v1.0" 

    # 3. Create Evaluation (Links to Response)
    evaluation = create_evaluation(
        test_db,
        response_id=response.id,
        evaluator_type="IoU",
        score=0.95,
        passed=True,
        metrics={"threshold": 0.9},
        feedback="Great prediction!"
    )
    assert evaluation.response_id == response.id
    assert evaluation.evaluator_type == "IoU"
    assert evaluation.score == 0.95
    assert evaluation.passed is True
    assert evaluation.feedback == "Great prediction!"

    # Test query for evaluation
    evals = get_evaluations_for_response(test_db, response.id)
    assert len(evals) == 1
    assert evals[0].metrics == {"threshold": 0.9}


def test_unique_constraint_on_response(test_db: Session):
    """Ensure a model version cannot respond to the same prompt twice."""
    prompt = create_prompt(
        test_db, 
        test_case_name="Unique Test", 
        model_type="NLP", 
        input_type="text", 
        output_type="classification", 
        input_data={"a": 1}, 
        ground_truth={"b": 2},
        test_case_metadata={"source": "auto"}
    )
    run = create_model_run(test_db, "Unique Model", "v1.0", "NLP")

    # First attempt (should succeed)
    create_response(test_db, prompt.id, run.id, {"r": 1})
    
    # Second attempt (should fail due to UNIQUE(test_case_id, run_id) constraint)
    with pytest.raises(IntegrityError):
        create_response(test_db, prompt.id, run.id, {"r": 2})
        test_db.commit() # The commit is what triggers the constraint check