from datetime import datetime
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional

from ml_eval import schemas
# Import all models (assuming the .models file is in the same directory)
from .models import TestPrompt, ModelRun, Response, Evaluation

# ====================================================================
# 1. TestPrompt CRUD (Golden Set Management)
# ====================================================================

def create_prompt(
    db: Session,
    test_case_name: str,
    model_type: str,
    input_type: str,
    output_type: str,
    input_data: Dict[str, Any],
    ground_truth: Dict[str, Any],
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    difficulty: Optional[str] = None,
    origin: str = "human",
    is_verified: bool = True,
    test_case_metadata: Optional[Dict[str, Any]] = None,
    created_by: Optional[str] = None,
) -> TestPrompt:
    """Creates a new TestPrompt (Golden Set) with user-first defaults."""
    db_prompt = TestPrompt(
        test_case_name=test_case_name,
        model_type=model_type,
        input_type=input_type,
        output_type=output_type,
        input_data=input_data,
        ground_truth=ground_truth,
        category=category,
        tags=tags,
        difficulty=difficulty,
        origin=origin,
        is_verified=is_verified,
        test_case_metadata=test_case_metadata,
        created_by=created_by,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    return db_prompt

def get_prompt(db: Session, prompt_id: int) -> Optional[TestPrompt]:
    """Reads a single prompt by its ID."""
    return db.query(TestPrompt).filter(TestPrompt.id == prompt_id).first()

def get_prompts_by_model_type(db: Session, model_type: str, limit: Optional[int] = None) -> List[TestPrompt]:
    """Reads a list of prompts filtered by model_type. Fetches all if limit is None."""
    query = db.query(TestPrompt).filter(TestPrompt.model_type == model_type)
    if limit is not None:
        query = query.limit(limit)
    return query.all()

def update_prompt(db: Session, prompt_id: int, prompt_update: schemas.TestPromptUpdate) -> Optional[TestPrompt]:
    """Updates a TestPrompt with new data."""
    db_prompt = get_prompt(db, prompt_id)
    if db_prompt:
        update_data = prompt_update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_prompt, key, value)
        db_prompt.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_prompt)
    return db_prompt

def delete_prompt(db: Session, prompt_id: int) -> bool:
    """Deletes a TestPrompt."""
    db_prompt = get_prompt(db, prompt_id)
    if db_prompt:
        db.delete(db_prompt)
        db.commit()
        return True
    return False

# ====================================================================
# 2. ModelRun CRUD (Version Tracking)
# ====================================================================

def create_model_run(
    db: Session,
    model_name: str,
    model_version: str,
    model_type: str,
    model_endpoint: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    status: str = 'pending',
    total_cases: int = 0,
    completed_cases: int = 0,
    failed_cases: int = 0,
) -> ModelRun:
    """Starts a new ModelRun tracking a specific model version."""
    db_run = ModelRun(
        model_name=model_name,
        model_version=model_version,
        model_type=model_type,
        model_endpoint=model_endpoint,
        config=config,
        status=status,
        total_cases=total_cases,
        completed_cases=completed_cases,
        failed_cases=failed_cases,
        started_at=datetime.utcnow()
    )
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    return db_run

def complete_model_run(db: Session, model_run_id: int) -> Optional[ModelRun]:
    """Marks a ModelRun as complete by setting the completed_at timestamp."""
    db_run = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
    if db_run:
        db_run.completed_at = datetime.utcnow()
        db_run.status = "completed"
        db.commit()
        db.refresh(db_run)
    return db_run

def get_model_run(db: Session, run_id: int) -> Optional[ModelRun]:
    """Retrieves a single ModelRun by its ID."""
    return db.query(ModelRun).filter(ModelRun.id == run_id).first()

def update_model_run(db: Session, run_id: int, run_update: schemas.ModelRunUpdate) -> Optional[ModelRun]:
    """Updates a ModelRun with new data."""
    db_run = get_model_run(db, run_id)
    if db_run:
        update_data = run_update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_run, key, value)
        db.commit()
        db.refresh(db_run)
    return db_run

def delete_model_run(db: Session, run_id: int) -> bool:
    """Deletes a ModelRun."""
    db_run = get_model_run(db, run_id)
    if db_run:
        db.delete(db_run)
        db.commit()
        return True
    return False

# ====================================================================
# 3. Response CRUD (Model Output)
# ====================================================================

def create_response(
    db: Session,
    run_id: int,
    test_case_id: int,
    output_data: Dict[str, Any],
    latency_ms: Optional[int] = None,
    memory_mb: Optional[float] = None,
    tokens_used: Optional[int] = None,
    error_message: Optional[str] = None,
) -> Response:
    """Stores the output from a model for a specific prompt and run."""
    db_response = Response(
        run_id=run_id,
        test_case_id=test_case_id,
        output_data=output_data,
        latency_ms=latency_ms,
        memory_mb=memory_mb,
        tokens_used=tokens_used,
        error_message=error_message,
        created_at=datetime.utcnow()
    )
    db.add(db_response)
    db.commit()
    db.refresh(db_response)
    return db_response

# ====================================================================
# 4. Evaluation CRUD (Metrics/Scoring)
# ====================================================================

def create_evaluation(
    db: Session,
    response_id: int,
    evaluator_type: str,
    score: float,
    passed: bool,
    metrics: Optional[Dict[str, Any]] = None,
    feedback: Optional[str] = None,
) -> Evaluation:
    """Stores a single evaluation score for a model response."""
    db_evaluation = Evaluation(
        response_id=response_id,
        evaluator_type=evaluator_type,
        score=score,
        passed=passed,
        metrics=metrics,
        feedback=feedback,
        evaluated_at=datetime.utcnow()
    )
    db.add(db_evaluation)
    db.commit()
    db.refresh(db_evaluation)
    return db_evaluation


def get_evaluations_for_response(db: Session, response_id: int) -> List[Evaluation]:
    """Retrieves all evaluations for a specific model response."""
    return db.query(Evaluation).filter(Evaluation.response_id == response_id).all()