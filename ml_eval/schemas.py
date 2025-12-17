# ml_eval/schemas.py

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime

# --- Base Schemas for Data Exchange ---

# 1. TestPrompt (Input Test Case)
class TestPromptBase(BaseModel):
    """Base class for creating a new test prompt."""
    test_case_name: str = Field(..., description="Unique name for the test prompt.")
    model_type: str = Field(..., description="The ML domain/model type this test belongs to (e.g., 'nlp', 'computer_vision').")
    input_type: str = Field(..., description="The type of the input data (e.g., 'text', 'image_path').")
    output_type: str = Field(..., description="The type of the expected output (e.g., 'classification', 'bounding_boxes').")
    input_data: Dict[str, Any] = Field(..., description="The universal JSONB input data for the model.")
    ground_truth: Dict[str, Any] = Field(..., description="The expected ground truth output for evaluation.")
    
    # Organization
    category: Optional[str] = Field(None)
    tags: Optional[List[str]] = Field(None)
    difficulty: Optional[str] = Field(None)

    # User-First Philosophy Fields
    origin: str = Field("human", description="The source of the prompt ('human', 'ai-generated').")
    is_verified: bool = Field(True, description="True for human-submitted Golden Sets.")
    
    # Metadata
    test_case_metadata: Dict[str, Any] = Field(default={})
    created_by: Optional[str] = Field(None)
    
class TestPrompt(TestPromptBase):
    """Schema for reading/displaying an existing TestPrompt."""
    id: int
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True) # Allows mapping from SQLAlchemy ORM models

class TestPromptUpdate(BaseModel):
    """Schema for updating a TestPrompt, all fields are optional."""
    test_case_name: Optional[str] = None
    model_type: Optional[str] = None
    input_type: Optional[str] = None
    output_type: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    ground_truth: Optional[Dict[str, Any]] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    difficulty: Optional[str] = None
    origin: Optional[str] = None
    is_verified: Optional[bool] = None
    test_case_metadata: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None

# 2. ModelRun (A Single Model Version Execution)
class ModelRunCreate(BaseModel):
    """Schema for starting a new model execution run."""
    model_name: str = Field(..., description="The identifier for the model being tested.")
    model_version: str = Field(..., description="The version of the model being tested.")
    model_type: str = Field(..., description="The ML domain/model type (e.g., 'nlp', 'computer_vision').")
    model_endpoint: Optional[str] = Field(None, description="API URL or local path for the model.")
    config: Dict[str, Any] = Field(default={}, description="Model-specific configuration or hyperparameters.")
    
class ModelRun(ModelRunCreate):
    """Schema for reading/displaying an existing ModelRun."""
    id: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_cases: int
    completed_cases: int
    failed_cases: int
    
    model_config = ConfigDict(from_attributes=True)

class ModelRunUpdate(BaseModel):
    """Schema for updating a ModelRun, all fields are optional."""
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_type: Optional[str] = None
    model_endpoint: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None

# 3. Response (Model Output)
class ResponseCreate(BaseModel):
    """Schema for saving a model's response to a prompt."""
    run_id: int = Field(..., description="Foreign Key to the ModelRun this response belongs to.")
    test_case_id: int = Field(..., description="Foreign Key to the TestPrompt used.")
    output_data: Dict[str, Any] = Field(..., description="The raw JSONB output from the model.")
    latency_ms: Optional[int] = Field(None, description="Time taken for the model to generate the response.")
    memory_mb: Optional[float] = Field(None, description="Memory used by the model for this response in MB.")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used for LLM responses.")
    error_message: Optional[str] = Field(None, description="Error message if the model failed to respond.")

class Response(ResponseCreate):
    """Schema for reading/displaying an existing Response."""
    id: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)
        
# 4. Evaluation (The Score)
class EvaluationBase(BaseModel):
    """Base class for evaluation results."""
    response_id: int
    evaluator_type: str = Field(..., description="Name or type of the evaluator used.")
    score: float = Field(..., description="The calculated metric score (0.0 to 1.0).")
    passed: bool = Field(..., description="Whether the response passed the evaluation criteria.")
    metrics: Dict[str, Any] = Field(default={}, description="Task-specific metrics or additional details.")
    feedback: Optional[str] = Field(None, description="Human-readable feedback or explanation.")
    
class Evaluation(EvaluationBase):
    """Schema for reading/displaying an existing Evaluation."""
    id: int
    evaluated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)