from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy import UniqueConstraint
from datetime import datetime
from .connection import Base

class TestPrompt(Base):
    """
    Represents a single verified test case (Golden Set).
    It contains the input data and the ground truth expected output.
    """
    __tablename__ = "test_cases"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Core Data Fields
    test_case_name = Column(String, index=True, nullable=False)
    model_type = Column(String, index=True, nullable=False) # e.g., 'NLP', 'CV', 'Recommender'
    input_type = Column(String, nullable=False)
    output_type = Column(String, nullable=False)

    # Universal Input/Output Storage (uses JSONB for flexibility across domains)
    input_data = Column(JSONB, nullable=False)        # e.g., {'text': 'What is the capital?', 'image_path': '...', 'user_id': 123}
    ground_truth = Column(JSONB, nullable=False)   # The ground truth

    # Organization
    category = Column(String, index=True)
    tags = Column(ARRAY(String))
    difficulty = Column(String)

    # User-First Philosophy Fields
    origin = Column(String, default="human", nullable=False)  # 'human', 'ai_generated', 'production_log'
    is_verified = Column(Boolean, default=True, nullable=False) # True for human-submitted Golden Sets

    # Metadata
    test_case_metadata = Column(JSONB)
    created_by = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    responses = relationship("Response", back_populates="prompt")


class ModelRun(Base):
    """
    Represents a specific execution of a model version across the test set.
    """
    __tablename__ = "model_runs"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Model Identification
    model_name = Column(String, index=True, nullable=False) # e.g., 'claude-3-sonnet', 'resnet50-v2'
    model_version = Column(String, nullable=False)       # Version tracking is a core feature
    model_type = Column(String, nullable=False) # e.g., 'computer_vision', 'nlp', etc.
    model_endpoint = Column(String) # API URL or local path
    config = Column(JSONB) # Model-specific configuration
    
    # Environment/Metadata
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String, default='pending') # 'pending', 'running', 'completed', 'failed'
    total_cases = Column(Integer, default=0)
    completed_cases = Column(Integer, default=0)
    failed_cases = Column(Integer, default=0)
    
    # Relationships
    responses = relationship("Response", back_populates="model_run")


class Response(Base):
    """
    Represents the output from a single model for a single test prompt.
    This links ModelRun and TestPrompt.
    """
    __tablename__ = "responses"

    # ----------------------------------------------------------------------
    # 2. ADD THE UNIQUE CONSTRAINT HERE
    # This prevents the same model version from running the same prompt twice.
    __table_args__ = (
        UniqueConstraint('test_case_id', 'run_id', name='uq_test_case_run'),
    )
    # ----------------------------------------------------------------------

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign Keys
    run_id = Column(Integer, ForeignKey("model_runs.id"), nullable=False, index=True)
    test_case_id = Column(Integer, ForeignKey("test_cases.id"), nullable=False, index=True)

    # Universal Output Storage (JSONB)
    output_data = Column(JSONB, nullable=False) # The actual model prediction

    # Performance metrics
    latency_ms = Column(Integer)
    memory_mb = Column(Float)
    tokens_used = Column(Integer)

    # Error handling
    error_message = Column(String)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    prompt = relationship("TestPrompt", back_populates="responses")
    model_run = relationship("ModelRun", back_populates="responses")
    evaluations = relationship("Evaluation", back_populates="response")


class Evaluation(Base):
    """
    Represents a quantitative assessment (score) of a Response against the ground truth.
    """
    __tablename__ = "evaluations"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign Key
    response_id = Column(Integer, ForeignKey("responses.id"), nullable=False, index=True)

    # Evaluation Data
    evaluator_type = Column(String, nullable=False) # e.g., 'BLEU', 'Accuracy', 'Custom_Metric_V1'
    score = Column(Float, nullable=False)           # The calculated metric score
    passed = Column(Boolean, nullable=False)       # Simple pass/fail against a threshold

    # Metadata
    metrics = Column(JSONB)                         # Optional: Store evaluation configuration or debug info
    feedback = Column(String)
    evaluated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    response = relationship("Response", back_populates="evaluations")