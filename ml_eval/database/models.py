from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Float
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import UniqueConstraint
from datetime import datetime

# Base class for declarative class definitions
Base = declarative_base()

class TestPrompt(Base):
    """
    Represents a single verified test case (Golden Set).
    It contains the input data and the ground truth expected output.
    """
    __tablename__ = "test_prompts"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Core Data Fields
    name = Column(String, index=True, nullable=False)
    domain = Column(String, index=True, nullable=False) # e.g., 'NLP', 'CV', 'Recommender'

    # Universal Input/Output Storage (uses JSONB for flexibility across domains)
    input_data = Column(JSONB, nullable=False)        # e.g., {'text': 'What is the capital?', 'image_path': '...', 'user_id': 123}
    expected_output = Column(JSONB, nullable=False)   # The ground truth

    # User-First Philosophy Fields
    origin = Column(String, default="human", nullable=False)  # 'human', 'ai_generated', 'production_log'
    is_verified = Column(Boolean, default=True, nullable=False) # True for human-submitted Golden Sets

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
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
    
    # Environment/Metadata
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime)
    
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
        UniqueConstraint('prompt_id', 'model_run_id', name='uq_prompt_model_run'),
    )
    # ----------------------------------------------------------------------

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign Keys
    prompt_id = Column(Integer, ForeignKey("test_prompts.id"), nullable=False, index=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"), nullable=False, index=True)

    # Universal Output Storage (JSONB)
    output_data = Column(JSONB, nullable=False) # The actual model prediction

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
    evaluator_name = Column(String, nullable=False) # e.g., 'BLEU', 'Accuracy', 'Custom_Metric_V1'
    score = Column(Float, nullable=False)           # The calculated metric score
    is_pass = Column(Boolean, nullable=False)       # Simple pass/fail against a threshold

    # Metadata
    evaluated_at = Column(DateTime, default=datetime.utcnow)
    details = Column(JSONB)                         # Optional: Store evaluation configuration or debug info

    # Relationships
    response = relationship("Response", back_populates="evaluations")