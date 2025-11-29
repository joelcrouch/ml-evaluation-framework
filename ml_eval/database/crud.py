from datetime import datetime
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional

# Import all models (assuming the .models file is in the same directory)
from .models import TestPrompt, ModelRun, Response, Evaluation

# ====================================================================
# 1. TestPrompt CRUD (Golden Set Management)
# ====================================================================

def create_prompt(
    db: Session,
    name: str,
    domain: str,
    input_data: Dict[str, Any],
    expected_output: Dict[str, Any],
    origin: str = "human",
    is_verified: bool = True # User-First Default
) -> TestPrompt:
    """Creates a new TestPrompt (Golden Set) with user-first defaults."""
    db_prompt = TestPrompt(
        name=name,
        domain=domain,
        input_data=input_data,
        expected_output=expected_output,
        origin=origin,
        is_verified=is_verified,
        created_at=datetime.utcnow()
    )
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    return db_prompt

def get_prompt(db: Session, prompt_id: int) -> Optional[TestPrompt]:
    """Reads a single prompt by its ID."""
    return db.query(TestPrompt).filter(TestPrompt.id == prompt_id).first()

def get_prompts_by_domain(db: Session, domain: str, limit: int = 100) -> List[TestPrompt]:
    """Reads a list of prompts filtered by domain."""
    return db.query(TestPrompt).filter(TestPrompt.domain == domain).limit(limit).all()

# ====================================================================
# 2. ModelRun CRUD (Version Tracking)
# ====================================================================

def create_model_run(
    db: Session,
    model_name: str,
    model_version: str
) -> ModelRun:
    """Starts a new ModelRun tracking a specific model version."""
    db_run = ModelRun(
        model_name=model_name,
        model_version=model_version,
        started_at=datetime.utcnow()
    )
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    return db_run

def complete_model_run(db: Session, model_run_id: int) -> Optional[ModelRun]:
    """Marks a ModelRun as complete by setting the finished_at timestamp."""
    db_run = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
    if db_run:
        db_run.finished_at = datetime.utcnow()
        db.commit()
        db.refresh(db_run)
    return db_run

# ====================================================================
# 3. Response CRUD (Model Output)
# ====================================================================

def create_response(
    db: Session,
    prompt_id: int,
    model_run_id: int,
    output_data: Dict[str, Any]
) -> Response:
    """Stores the output from a model for a specific prompt and run."""
    db_response = Response(
        prompt_id=prompt_id,
        model_run_id=model_run_id,
        output_data=output_data,
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
    evaluator_name: str,
    score: float,
    is_pass: bool,
    details: Optional[Dict[str, Any]] = None
) -> Evaluation:
    """Stores a single evaluation score for a model response."""
    db_evaluation = Evaluation(
        response_id=response_id,
        evaluator_name=evaluator_name,
        score=score,
        is_pass=is_pass,
        details=details,
        evaluated_at=datetime.utcnow()
    )
    db.add(db_evaluation)
    db.commit()
    db.refresh(db_evaluation)
    return db_evaluation


def get_evaluations_for_response(db: Session, response_id: int) -> List[Evaluation]:
    """Retrieves all evaluations for a specific model response."""
    return db.query(Evaluation).filter(Evaluation.response_id == response_id).all()