import pytest
from datetime import datetime
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# Assuming a fixture setup for the database session
# You would typically define 'test_db' fixture in conftest.py
# For this example, we mock a database dependency:
from ml_eval.database.crud import (
    create_prompt, get_prompt, get_prompts_by_domain,
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
    expected_output = {"text": "World"}
    
    prompt = create_prompt(
        test_db, 
        name="Test Prompt 1", 
        domain="NLP", 
        input_data=input_data, 
        expected_output=expected_output
    )
    
    assert prompt.id is not None
    assert prompt.domain == "NLP"
    assert prompt.origin == "human"  # Check User-First Default
    assert prompt.is_verified is True # Check User-First Default
    assert prompt.input_data == input_data
    
    # Test read operation
    fetched_prompt = get_prompt(test_db, prompt.id)
    assert fetched_prompt.name == "Test Prompt 1"


def test_model_run_lifecycle(test_db: Session):
    """Test creation and completion of a ModelRun."""
    run = create_model_run(test_db, "ResNet50", "v1.2")
    assert run.model_name == "ResNet50"
    assert run.finished_at is None
    
    completed_run = complete_model_run(test_db, run.id)
    assert completed_run.finished_at is not None
    assert completed_run.finished_at >= run.started_at


def test_end_to_end_data_flow_and_foreign_keys(test_db: Session):
    """Tests the entire data pipeline and foreign key relationships."""
    
    # 1. Create foundational records
    prompt = create_prompt(test_db, "FK Test", "CV", {"path": "img"}, {"label": "cat"})
    run = create_model_run(test_db, "VisionModel", "v1.0")
    
    # 2. Create Response (Links Prompt and Run)
    output_data = {"predictions": [{"label": "cat", "conf": 0.9}]}
    response = create_response(
        test_db, 
        prompt_id=prompt.id, 
        model_run_id=run.id, 
        output_data=output_data
    )
    assert response.prompt_id == prompt.id
    assert response.model_run_id == run.id
    assert response.output_data == output_data
    
    # Test relationship access (SQLAlchemy ORM)
    assert response.prompt.name == "FK Test" 
    assert response.model_run.model_version == "v1.0" 

    # 3. Create Evaluation (Links to Response)
    evaluation = create_evaluation(
        test_db,
        response_id=response.id,
        evaluator_name="IoU",
        score=0.95,
        is_pass=True,
        details={"threshold": 0.9}
    )
    assert evaluation.response_id == response.id
    assert evaluation.evaluator_name == "IoU"
    assert evaluation.score == 0.95
    assert evaluation.is_pass is True

    # Test query for evaluation
    evals = get_evaluations_for_response(test_db, response.id)
    assert len(evals) == 1
    assert evals[0].details == {"threshold": 0.9}


def test_unique_constraint_on_response(test_db: Session):
    """Ensure a model version cannot respond to the same prompt twice."""
    prompt = create_prompt(test_db, "Unique Test", "NLP", {"a": 1}, {"b": 2})
    run = create_model_run(test_db, "Unique Model", "v1.0")

    # First attempt (should succeed)
    create_response(test_db, prompt.id, run.id, {"r": 1})
    
    # Second attempt (should fail due to UNIQUE(prompt_id, model_run_id) constraint)
    with pytest.raises(IntegrityError):
        create_response(test_db, prompt.id, run.id, {"r": 2})
        test_db.commit() # The commit is what triggers the constraint check