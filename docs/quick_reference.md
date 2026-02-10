# ML Evaluation Framework - Quick Reference

 **One-page cheat sheet for daily development**

 ---

 ## Database Tables

 ```
 test_cases      → Golden dataset (TestPrompt model)
 model_runs      → Execution tracking (ModelRun model)
 responses       → Model outputs (Response model)
 evaluations     → Scores & metrics (Evaluation model)
 ```

 **Relationships:**
 ```
 test_cases ─1:N→ responses ─1:N→ evaluations
 model_runs ─1:N→ responses
 ```

 ---

 ## Key Commands

 ### Database Setup
 ```bash
 # Initialize database (ONLY way to create schema)
 python scripts/init_db.py

 # Check Alembic status
 alembic current

 # View migration history
 alembic history
 ```

 ### Running Evaluations
 ```bash
 # 1. Seed database with test cases
 python scripts/seed_baseline_test_cases.py
 python scripts/seed_cv_test_cases.py

 # 2. Create model run
 curl -X POST http://localhost:8000/api/v1/runs/ \
   -H 'Content-Type: application/json' \
   -d '{"model_name": "baseline_model", "model_version": "1.0", "model_type": "baseline_time_series"}'
 # Returns: {"id": 1, ...}

 # 3. Run evaluation
 python scripts/run_evaluation.py 1

 # 4. Generate report
 python scripts/generate_report.py 1
 ```

 ### Testing
 ```bash
 # Run all tests
 pytest

 # Run with coverage
 pytest --cov=ml_eval tests/

 # Run specific test file
 pytest tests/test_database/test_crud.py -v
 ```

 ### Development Server
 ```bash
 # Start FastAPI server
 uvicorn ml_eval.main:app --reload --host 0.0.0.0 --port 8000

 # View API docs
 http://localhost:8000/docs
 ```

 ---

 ## Core Interfaces

 ### IModelAdapter
 ```python
 from ml_eval.core.interfaces.imodel import IModelAdapter

 class MyAdapter(IModelAdapter):
     def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
         """Run model and return predictions"""
         return {"prediction": [...]}
 ```

 ### IEvaluator
 ```python
 from ml_eval.core.interfaces.ievaluator import IEvaluator, EvaluationResult

 class MyEvaluator(IEvaluator):
     def evaluate(self, expected: Dict, actual: Dict) -> EvaluationResult:
         """Compare predictions against ground truth"""
         score = calculate_score(expected, actual)
         return EvaluationResult(
             score=score,
             passed=(score >= threshold),
             metrics={"metric1": value1, ...},
             feedback="..."
         )
 ```

 ---

 ## CRUD Operations

 ```python
 from ml_eval.database import crud
 from ml_eval.database.connection import SessionLocal

 db = SessionLocal()

 # TestPrompt (test_cases)
 prompt = crud.create_prompt(db, test_case_name="...", model_type="...", ...)
 prompt = crud.get_prompt(db, prompt_id=1)
 prompts = crud.get_prompts_by_model_type(db, "baseline_time_series", limit=50)

 # ModelRun
 run = crud.create_model_run(db, model_name="...", model_version="...", ...)
 run = crud.get_model_run(db, run_id=1)
 run = crud.complete_model_run(db, model_run_id=1)

 # Response
 response = crud.create_response(db, run_id=1, test_case_id=1, output_data={...})

 # Evaluation
 evaluation = crud.create_evaluation(
     db, response_id=1, evaluator_type="MeanSquaredErrorEvaluator",
     score=0.95, passed=True, metrics={...}
 )
 ```

 ---

 ## API Endpoints

 ### TestPrompt (Test Cases)
 ```
 POST   /api/v1/prompts/                    # Create
 GET    /api/v1/prompts/{id}                # Read
 GET    /api/v1/prompts/domain/{model_type} # List by type
 PUT    /api/v1/prompts/{id}                # Update (full)
 PATCH  /api/v1/prompts/{id}                # Update (partial)
 DELETE /api/v1/prompts/{id}                # Delete
 ```

 ### ModelRun
 ```
 POST   /api/v1/runs/                       # Create
 GET    /api/v1/runs/{id}                   # Read
 POST   /api/v1/runs/{id}/complete          # Mark complete
 PUT    /api/v1/runs/{id}                   # Update (full)
 PATCH  /api/v1/runs/{id}                   # Update (partial)
 DELETE /api/v1/runs/{id}                   # Delete
 ```

 ---

 ## File Locations

 ### Core Code
 ```
 ml_eval/
 ├── core/
 │   ├── interfaces/          # IModelAdapter, IEvaluator
 │   └── implementations/     # Model classes, adapters, evaluators
 ├── database/
 │   ├── models.py           # SQLAlchemy models
 │   ├── crud.py             # CRUD operations
 │   └── connection.py       # Database connection
 ├── query_engine/
 │   └── engine.py           # EvaluationEngine
 ├── routers/
 │   └── crud.py             # FastAPI endpoints
 └── main.py                 # FastAPI app
 ```

 ### Scripts
 ```
 scripts/
 ├── init_db.py                      # Initialize database
 ├── seed_cv_test_cases.py           # Seed image classification
 ├── seed_baseline_test_cases.py     # Seed time series
 ├── run_evaluation.py               # Execute evaluation
 ├── generate_report.py              # Generate reports
 └── generate_report_time_series.py  # Time series reports
 ```

 ### Data & Models
 ```
 data/
 ├── processed_data.csv              # Training data
 ├── baseline_golden_dataset.json    # Golden test cases
 └── seeded_test_images/             # Image test cases

 models/
 ├── baseline_model.keras            # Trained models
 ├── dense_model.keras
 └── cv_flower_classifier.keras
 ```

 ### Reports
 ```
 reports/
 ├── run_1_time_series_report.png
 ├── run_1_prediction_samples.png
 └── run_1_summary.csv
 ```

 ---

 ## Common Patterns

 ### Adding a New Model Type

 1. **Create Model Class**
    ```python
    # ml_eval/core/implementations/my_model.py
    class MyModel:
        def __init__(self, model_path: str):
            self.model = load_model(model_path)

        def predict(self, input_data: Dict) -> Dict:
            return {"output": result}
    ```

 2. **Create Adapter**
    ```python
    # ml_eval/core/implementations/my_adapter.py
    class MyAdapter(IModelAdapter):
        def __init__(self, model: MyModel):
            self.model = model

        def run(self, input_data: Dict) -> Dict:
            return self.model.predict(input_data)
    ```

 3. **Update run_evaluation.py**
    ```python
    elif model_run.model_type == "my_model_type":
        model = MyModel(model_path)
        adapter = MyAdapter(model)
        evaluator = MyEvaluator()
    ```

 ---

 ## Database Schema Changes

 **NEVER use `Base.metadata.create_all()`**

 ### Proper Way to Change Schema

 1. **Modify models**
    ```bash
    vim ml_eval/database/models.py
    ```

 2. **Generate migration**
    ```bash
    alembic revision --autogenerate -m "Add new field"
    ```

 3. **Review migration**
    ```bash
    vim migrations/versions/<hash>_add_new_field.py
    ```

 4. **Apply migration**
    ```bash
    alembic upgrade head
    ```

 ---

 ## JSONB Field Formats

 ### Time Series
 ```python
 # input_data
 {
     "window": [
         [feature1, feature2, feature3, ...],  # timestep 1
         [feature1, feature2, feature3, ...],  # timestep 2
         ...
     ]
 }

 # ground_truth
 {
     "prediction": [[temperature_value]]
 }
 ```

 ### Computer Vision
 ```python
 # input_data
 {
     "image_path": "/absolute/path/to/image.jpg"
 }

 # ground_truth
 {
     "label": "tulips"
 }
 ```

 ### NLP (Example)
 ```python
 # input_data
 {
     "text": "What is the capital of France?"
 }

 # ground_truth
 {
     "answer": "Paris"
 }
 ```

 ---

 ## Evaluation Flow

 ```
 1. Create ModelRun → INSERT INTO model_runs (status='pending')

 2. EvaluationEngine.run_evaluation(run_id):
    ├─ UPDATE model_runs SET status='running'
    ├─ SELECT test_cases WHERE model_type='...'
    └─ FOR EACH test_case:
       ├─ adapter.run() → predictions
       ├─ INSERT INTO responses (output_data)
       ├─ evaluator.evaluate() → EvaluationResult
       ├─ INSERT INTO evaluations (score, metrics)
       └─ UPDATE model_runs (progress)
    → UPDATE model_runs SET status='completed'
 ```

 ---

 ## Troubleshooting

 ### Tables don't exist
 ```bash
 python scripts/init_db.py
 ```

 ### "UNIQUE constraint failed"
 - You're trying to evaluate the same test case twice
 - Create a new ModelRun or delete old responses

 ### "Model file not found"
 - Check that model file exists: `ls -la models/`
 - Verify model_name in ModelRun matches filename

 ### Tests failing
 ```bash
 # Drop and recreate test database
 psql -h localhost -p 5433 -U ml_user -c "DROP DATABASE IF EXISTS ml_eval_db_test;"
 psql -h localhost -p 5433 -U ml_user -c "CREATE DATABASE ml_eval_db_test;"
 pytest
 ```

 ---

 ## Environment Variables

 ```bash
 # .env file
 POSTGRES_USER=ml_user
 POSTGRES_PASSWORD=ml_password
 POSTGRES_DB=ml_eval_db
 POSTGRES_HOST=localhost
 POSTGRES_PORT=5433

 SQLALCHEMY_DATABASE_URL=postgresql+psycopg2://ml_user:ml_password@localhost:5433/ml_eval_db
 ```

 ---

 ## Useful Queries

 ### View all test cases for a model type
 ```sql
 SELECT * FROM test_cases WHERE model_type = 'baseline_time_series';
 ```

 ### View results for a run
 ```sql
 SELECT
     e.score, e.passed, e.metrics,
     r.output_data,
     t.test_case_name, t.ground_truth
 FROM evaluations e
 JOIN responses r ON e.response_id = r.id
 JOIN test_cases t ON r.test_case_id = t.id
 WHERE r.run_id = 1;
 ```

 ### Calculate average score for a run
 ```sql
 SELECT AVG(score) as avg_score, COUNT(*) as total
 FROM evaluations e
 JOIN responses r ON e.response_id = r.id
 WHERE r.run_id = 1;
 ```

 ### Find failed evaluations
 ```sql
 SELECT * FROM evaluations WHERE passed = false;
 ```

 ---

 ## Key Principles

 1. ✅ **Always use Alembic** for schema changes
 2. ✅ **Run `init_db.py`** to initialize databases
 3. ✅ **Use CRUD operations** for database access
 4. ✅ **Implement IModelAdapter** for new model types
 5. ✅ **Implement IEvaluator** for new metrics
 6. ❌ **Never use `Base.metadata.create_all()`**
 7. ❌ **Never skip migrations**

 ---

 ## Documentation Links

 - **Architecture:** `docs/ARCHITECTURE.md`
 - **Demo Guide:** `docs/DEMO_GUIDE.md`
 - **UML Diagrams:** `docs/uml/`
 - **Alembic Refactor:** `docs/alembic_migration_refactor.md`
 - **Troubleshooting:** `docs/troubleshooting_db_errors.md`

 ---

 **Last Updated:** 2026-02-07
