# ML Evaluation Framework - Architecture Overview

**Last Updated:** 2026-02-07
**Sprint:** 4
**Status:** Production-Ready Infrastructure

---

## Table of Contents
1. [Quick Reference](#quick-reference)
2. [System Overview](#system-overview)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Database Schema](#database-schema)
6. [Design Patterns](#design-patterns)
7. [UML Diagrams](#uml-diagrams)

---

## Quick Reference

### Key Files by Function

**Core Interfaces:**
- `ml_eval/core/interfaces/imodel.py` - IModelAdapter interface
- `ml_eval/core/interfaces/ievaluator.py` - IEvaluator interface

**Database:**
- `ml_eval/database/models.py` - SQLAlchemy models (TestPrompt, ModelRun, Response, Evaluation)
- `ml_eval/database/crud.py` - All CRUD operations
- `ml_eval/database/connection.py` - Database connection & session management
- `migrations/versions/` - Alembic migration files

**Evaluation Engine:**
- `ml_eval/query_engine/engine.py` - EvaluationEngine (orchestrates evaluation)

**Model Implementations:**
- `ml_eval/core/implementations/baseline_time_series_model.py` - Baseline time series model
- `ml_eval/core/implementations/keras_time_series_model.py` - Generic Keras time series
- `ml_eval/core/implementations/image_classifier_model.py` - Image classification

**Adapters:**
- `ml_eval/core/implementations/baseline_time_series_adapter.py`
- `ml_eval/core/implementations/keras_time_series_adapter.py`
- `ml_eval/core/implementations/image_classifier_adapter.py`

**Evaluators:**
- `ml_eval/core/implementations/exact_match.py` - ExactMatchEvaluator
- `ml_eval/core/implementations/accuracy_evaluator.py` - AccuracyEvaluator
- `ml_eval/core/implementations/mean_squared_error_evaluator.py` - MeanSquaredErrorEvaluator

**API:**
- `ml_eval/main.py` - FastAPI application
- `ml_eval/routers/crud.py` - REST API endpoints

**Scripts:**
- `scripts/init_db.py` - Initialize database (THE ONLY WAY)
- `scripts/seed_cv_test_cases.py` - Seed image classification test cases
- `scripts/seed_baseline_test_cases.py` - Seed time series test cases
- `scripts/run_evaluation.py` - Execute model evaluation
- `scripts/generate_report.py` - Generate performance reports

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                   ML EVALUATION FRAMEWORK                           │
│                                                                     │
│  Universal platform for evaluating ML models across domains        │
│  (Time Series, Computer Vision, NLP, etc.)                        │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Golden Dataset  │  ← Training scripts create verified test cases
│   (JSON/DB)      │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│   Database       │  ← PostgreSQL stores test cases, runs, results
│  (PostgreSQL)    │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ Evaluation       │  ← EvaluationEngine orchestrates evaluation
│   Engine         │
└────────┬─────────┘
         │
         ├──→ Model Adapters (IModelAdapter implementations)
         ├──→ Evaluators (IEvaluator implementations)
         └──→ CRUD Operations

         ↓
┌──────────────────┐
│   Reporting      │  ← Generate metrics, charts, summaries
│  & Analytics     │
└──────────────────┘
```

---

## Component Details

### 1. Database Models (Golden Dataset Storage)

**TestPrompt** (`test_cases` table)
- Stores verified test cases ("golden dataset")
- Universal JSONB fields support any ML domain
- Example input_data:
  - Time Series: `{"window": [[feat1, feat2, ...]]}`
  - CV: `{"image_path": "/path/to/image.jpg"}`
  - NLP: `{"text": "What is the capital of France?"}`
- Example ground_truth:
  - Time Series: `{"prediction": [[temp_value]]}`
  - CV: `{"label": "tulips"}`
  - NLP: `{"answer": "Paris"}`

**ModelRun** (`model_runs` table)
- Tracks execution of a specific model version
- Status: `pending` → `running` → `completed`/`failed`
- Progress tracking: `total_cases`, `completed_cases`, `failed_cases`

**Response** (`responses` table)
- Stores model output for each test case
- Links ModelRun ↔ TestPrompt
- UNIQUE constraint prevents duplicate evaluations
- Performance metrics: `latency_ms`, `memory_mb`, `tokens_used`

**Evaluation** (`evaluations` table)
- Stores scoring/metrics for each Response
- Contains: `score`, `passed`, `metrics` (JSONB), `feedback`
- Multiple evaluations per response (different metrics)

### 2. Core Interfaces (Strategy Pattern)

**IModelAdapter**
```python
class IModelAdapter(ABC):
    @abstractmethod
    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model and return predictions"""
        pass
```

**Implementations:**
- `BaselineTimeSeriesAdapter` - Baseline time series models
- `KerasTimeSeriesAdapter` - Generic Keras time series
- `ImageClassifierAdapter` - Image classification
- `LocalMatrixAdapter` - Matrix operations
- `SimpleModelAdapter` - Simple string matching

**Benefits:** EvaluationEngine is model-agnostic. Adding new model types only requires implementing this interface.

---

**IEvaluator**
```python
class IEvaluator(ABC):
    @abstractmethod
    def evaluate(self, expected: Dict, actual: Dict) -> EvaluationResult:
        """Compare predictions against ground truth"""
        pass
```

**Implementations:**
- `MeanSquaredErrorEvaluator` - For regression (time series, etc.)
- `AccuracyEvaluator` - For classification
- `ExactMatchEvaluator` - For exact string/value matching

**Benefits:** Easy to add new evaluation metrics without changing the engine.

---

### 3. Evaluation Engine (Orchestration)

**EvaluationEngine** (`ml_eval/query_engine/engine.py`)

```python
class EvaluationEngine:
    def __init__(self, db: Session, model_adapter: IModelAdapter, evaluator: IEvaluator):
        self.db = db
        self.model_adapter = model_adapter
        self.evaluator = evaluator

    def run_evaluation(self, run_id: int) -> Optional[ModelRun]:
        # 1. Fetch ModelRun
        # 2. Update status to 'running'
        # 3. Get all test prompts for this model type
        # 4. For each prompt:
        #    a. Run model prediction via adapter
        #    b. Store response in DB
        #    c. Evaluate prediction vs ground truth
        #    d. Store evaluation in DB
        #    e. Update progress
        # 5. Mark ModelRun as complete
```

**Key Features:**
- Model-agnostic (uses IModelAdapter)
- Metric-agnostic (uses IEvaluator)
- Transactional (commits after each test case)
- Progress tracking (updates ModelRun after each case)
- Error handling (stores errors in Response.error_message)

---

### 4. CRUD Operations (Repository Pattern)

**Location:** `ml_eval/database/crud.py`

All database operations go through these functions:

**TestPrompt CRUD:**
- `create_prompt()` - Insert new test case
- `get_prompt()` - Fetch by ID
- `get_prompts_by_model_type()` - Filter by model type
- `update_prompt()` - Modify test case
- `delete_prompt()` - Remove test case

**ModelRun CRUD:**
- `create_model_run()` - Start new evaluation run
- `get_model_run()` - Fetch by ID
- `complete_model_run()` - Mark as completed
- `update_model_run()` - Update fields

**Response CRUD:**
- `create_response()` - Store model output

**Evaluation CRUD:**
- `create_evaluation()` - Store evaluation score
- `get_evaluations_for_response()` - Fetch all evaluations for a response

**Benefits:** Single source of truth for database queries, easier testing and maintenance.

---

### 5. FastAPI Router (REST API)

**Location:** `ml_eval/routers/crud.py`

**TestPrompt Endpoints:**
```
POST   /api/v1/prompts/                    Create test case
GET    /api/v1/prompts/{prompt_id}         Get test case
GET    /api/v1/prompts/domain/{model_type} List by model type
PUT    /api/v1/prompts/{prompt_id}         Update test case
PATCH  /api/v1/prompts/{prompt_id}         Partial update
DELETE /api/v1/prompts/{prompt_id}         Delete test case
```

**ModelRun Endpoints:**
```
POST   /api/v1/runs/                       Create model run
GET    /api/v1/runs/{run_id}               Get model run
POST   /api/v1/runs/{run_id}/complete      Mark as complete
PUT    /api/v1/runs/{run_id}               Update model run
PATCH  /api/v1/runs/{run_id}               Partial update
DELETE /api/v1/runs/{run_id}               Delete model run
```

**Usage:**
- Seeding scripts call these endpoints to populate database
- External tools can integrate via REST API
- FastAPI auto-generates OpenAPI docs at `/docs`

---

## Data Flow

### Complete Evaluation Pipeline

```
1. TRAINING & GOLDEN DATASET CREATION
   ↓
   train_baseline_time_series.py
   ├─ Load & split data (70/20/10)
   ├─ Train model
   ├─ Save models/baseline_model.keras
   └─ create_golden_dataset() → data/baseline_golden_dataset.json

2. DATABASE SEEDING
   ↓
   seed_baseline_test_cases.py
   ├─ Load golden_dataset.json
   └─ For each case:
      POST /api/v1/prompts/ → INSERT INTO test_cases

3. MODEL RUN CREATION
   ↓
   curl POST /api/v1/runs/
   {
     "model_name": "baseline_model",
     "model_version": "1.0",
     "model_type": "baseline_time_series"
   }
   → INSERT INTO model_runs (status='pending')

4. EVALUATION EXECUTION
   ↓
   python run_evaluation.py <run_id>
   ├─ Load model & create adapter
   ├─ Create evaluator
   ├─ Instantiate EvaluationEngine
   └─ run_evaluation():
      ├─ UPDATE model_runs SET status='running'
      ├─ SELECT test_cases WHERE model_type='...'
      └─ For each test_case:
         ├─ adapter.run() → model.predict()
         ├─ INSERT INTO responses (output_data)
         ├─ evaluator.evaluate(ground_truth, output_data)
         ├─ INSERT INTO evaluations (score, metrics)
         └─ UPDATE model_runs (progress)
      → UPDATE model_runs SET status='completed'

5. REPORT GENERATION
   ↓
   python generate_report.py <run_id>
   ├─ SELECT evaluations JOIN responses JOIN test_cases
   ├─ Calculate aggregate metrics (mean, median, std)
   ├─ Generate charts (distributions, trends)
   └─ Save reports/run_<run_id>_report.png
```

---

## Database Schema

### Entity Relationships

```
test_cases (TestPrompt)
    ↓ 1:N
responses (Response)
    ↓ 1:N
evaluations (Evaluation)

model_runs (ModelRun)
    ↓ 1:N
responses (Response)
```

### Key Constraints

**UNIQUE Constraint on responses:**
```sql
UNIQUE(test_case_id, run_id)
```
Prevents a model version from running the same test case twice.

**Foreign Keys:**
- `responses.run_id` → `model_runs.id`
- `responses.test_case_id` → `test_cases.id`
- `evaluations.response_id` → `responses.id`

### JSONB Fields (Flexibility)

**test_cases.input_data:**
- Time Series: `{"window": [[0.1, 0.2, ...], ...]}`
- CV: `{"image_path": "/path/to/image.jpg"}`
- NLP: `{"text": "What is AI?"}`

**test_cases.ground_truth:**
- Time Series: `{"prediction": [[0.234]]}`
- CV: `{"label": "tulips"}`
- NLP: `{"answer": "Artificial Intelligence"}`

**responses.output_data:**
- Time Series: `{"prediction": [[0.235]], "metadata": {...}}`
- CV: `{"predicted_label": "sunflowers"}`
- NLP: `{"answer": "AI is..."}`

**evaluations.metrics:**
- Regression: `{"mse": 0.0142, "mae": 0.0852, "rmse": 0.119}`
- Classification: `{"is_correct": true, "expected": "tulips", "actual": "tulips"}`

---

## Design Patterns

### 1. Strategy Pattern (Adapters)

**Problem:** Different model types require different prediction logic.

**Solution:** Define `IModelAdapter` interface, implement for each model type.

**Benefit:** EvaluationEngine doesn't need to know about specific model implementations.

```python
# Adding a new model type is simple:
class NewModelAdapter(IModelAdapter):
    def run(self, input_data: Dict) -> Dict:
        # Custom prediction logic
        return predictions

# Use in run_evaluation.py:
if model_type == "new_model":
    adapter = NewModelAdapter(model)
```

---

### 2. Strategy Pattern (Evaluators)

**Problem:** Different domains need different evaluation metrics.

**Solution:** Define `IEvaluator` interface, implement for each metric type.

**Benefit:** Easy to add new metrics without touching the engine.

```python
# Adding a new metric is simple:
class BLEUScoreEvaluator(IEvaluator):
    def evaluate(self, expected: Dict, actual: Dict) -> EvaluationResult:
        # Calculate BLEU score
        return EvaluationResult(score, passed, metrics)
```

---

### 3. Repository Pattern (CRUD)

**Problem:** Database queries scattered throughout codebase.

**Solution:** Centralize all database operations in `crud.py`.

**Benefit:** Single source of truth, easier testing, consistent transactions.

---

### 4. Entity-Relationship (Database)

**Problem:** Complex relationships between test cases, runs, responses, evaluations.

**Solution:** Well-defined foreign keys and constraints.

**Benefit:** Data integrity, efficient queries, prevents duplicates.

---

## UML Diagrams

Comprehensive UML diagrams are available in `docs/uml/`:

1. **Sequence Diagram** (`sequence_diagram_complete_flow.puml`)
   - Shows end-to-end flow from training to reporting
   - All method calls and database operations
   - Actor interactions

2. **Class Diagram** (`class_diagram_architecture.puml`)
   - All classes, interfaces, and relationships
   - Method signatures
   - Design patterns illustrated

3. **Database ER Diagram** (`database_er_diagram.puml`)
   - Complete schema with types, indexes, constraints
   - Relationships and cardinality
   - JSONB field examples

4. **Data Flow Diagram** (`data_flow_diagram.puml`)
   - High-level architecture
   - Component interactions
   - File artifacts

**View online:** http://www.plantuml.com/plantuml/uml/

See `docs/uml/README.md` for viewing instructions.

---

## Adding New Model Types

To add a new model type to the framework:

### Step 1: Implement the Model Class
```python
# ml_eval/core/implementations/my_model.py
class MyModel:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self, input_data: Dict) -> Dict:
        # Prediction logic
        return {"output": result}
```

### Step 2: Create an Adapter
```python
# ml_eval/core/implementations/my_adapter.py
from ml_eval.core.interfaces.imodel import IModelAdapter

class MyAdapter(IModelAdapter):
    def __init__(self, model: MyModel):
        self.model = model

    def run(self, input_data: Dict) -> Dict:
        return self.model.predict(input_data)
```

### Step 3: Choose/Create an Evaluator
```python
# Use existing evaluator or create new one
from ml_eval.core.implementations.accuracy_evaluator import AccuracyEvaluator
# OR
class MyEvaluator(IEvaluator):
    def evaluate(self, expected: Dict, actual: Dict) -> EvaluationResult:
        # Custom evaluation logic
```

### Step 4: Update run_evaluation.py
```python
elif model_run.model_type == "my_model_type":
    model = MyModel(model_path)
    adapter = MyAdapter(model)
    evaluator = MyEvaluator()  # or use existing
```

### Step 5: Create Training & Seeding Scripts
- `ml_eval/core/implementations/train_my_model.py` - Train & create golden dataset
- `scripts/seed_my_test_cases.py` - Seed database

### Step 6: Test End-to-End
```bash
# 1. Train & create golden dataset
python ml_eval/core/implementations/train_my_model.py

# 2. Seed database
python scripts/seed_my_test_cases.py

# 3. Create model run
curl -X POST http://localhost:8000/api/v1/runs/ \
  -H 'Content-Type: application/json' \
  -d '{"model_name": "my_model", "model_version": "1.0", "model_type": "my_model_type"}'

# 4. Run evaluation
python scripts/run_evaluation.py <run_id>

# 5. Generate report
python scripts/generate_report.py <run_id>
```

---

## Key Principles

### 1. Alembic is the Single Source of Truth
- **NEVER** use `Base.metadata.create_all()`
- **ALWAYS** use `python scripts/init_db.py` to initialize database
- **ALWAYS** use Alembic migrations for schema changes

See `docs/alembic_migration_refactor.md` for details.

### 2. Universal JSONB Fields
- `input_data` and `ground_truth` use JSONB for flexibility
- Supports any ML domain without schema changes
- Each model type defines its own JSON structure

### 3. Model-Agnostic Design
- EvaluationEngine works with any IModelAdapter implementation
- Adding new model types requires no changes to the engine
- Consistent evaluation flow across all domains

### 4. Explicit Database Initialization
- Database setup is never implicit
- `init_db.py` detects and fixes schema drift automatically
- Repeatable and reliable across environments

---

## Performance Considerations

### Database Indexes
- `test_cases`: Indexed on `id`, `test_case_name`, `model_type`, `category`
- `model_runs`: Indexed on `id`, `model_name`
- `responses`: Indexed on `id`, `run_id`, `test_case_id`
- `evaluations`: Indexed on `id`, `response_id`

### JSONB Performance
- PostgreSQL efficiently stores and queries JSONB
- Use GIN indexes for complex JSONB queries (future optimization)

### Batch Processing
- EvaluationEngine processes one test case at a time
- Future: Add batch prediction support for models that support it

---

## Testing Strategy

### Unit Tests
- Test individual models, adapters, evaluators in isolation
- Mock database dependencies

### Integration Tests
- Test CRUD operations with real database (test DB)
- Test EvaluationEngine with mocked adapters/evaluators

### End-to-End Tests
- Full pipeline from seeding to reporting
- Uses separate test database (`ml_eval_db_test`)
- Test fixtures run Alembic migrations fresh for each test

See `tests/conftest.py` for fixture details.

---

## Troubleshooting

### Schema Drift Issues
**Symptom:** Alembic says migrations are applied, but tables don't exist.

**Solution:** Run `python scripts/init_db.py` - it auto-detects and fixes this.

### Duplicate Evaluation Errors
**Symptom:** `UNIQUE constraint failed: test_case_id, run_id`

**Reason:** Trying to evaluate the same test case with the same ModelRun twice.

**Solution:** Create a new ModelRun or delete old responses.

### Model Loading Errors
**Symptom:** `Model file not found` or `Failed to load model`

**Solution:** Ensure model file exists at expected path (e.g., `models/baseline_model.keras`).

---

## Related Documentation

- **Demo Guide:** `docs/DEMO_GUIDE.md` - Complete usage walkthrough
- **Alembic Refactor:** `docs/alembic_migration_refactor.md` - Database migration approach
- **Troubleshooting:** `docs/troubleshooting_db_errors.md` - Common issues & solutions
- **UML Diagrams:** `docs/uml/` - Visual architecture documentation

---

## Future Enhancements

### Planned Features
- [ ] Batch prediction support for faster evaluation
- [ ] Async evaluation with Celery/Redis
- [ ] Web dashboard for viewing results
- [ ] Model comparison reports (compare multiple runs side-by-side)
- [ ] Support for streaming models (LLMs)
- [ ] A/B testing framework
- [ ] Statistical significance testing

### Architecture Improvements
- [ ] Plugin system for dynamically loading model types
- [ ] GIN indexes on JSONB fields for complex queries
- [ ] Caching layer for frequently accessed test cases
- [ ] Distributed evaluation across multiple workers

---

**Maintained by:** Joel Crouch
**Project:** ML Evaluation Framework
**Sprint:** 4 (Production-Ready Infrastructure)
**Last Updated:** 2026-02-07
