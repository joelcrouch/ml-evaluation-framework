
import argparse
import sys
import os
from sqlalchemy.orm import Session

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_eval.database.connection import get_db
from ml_eval.query_engine.engine import EvaluationEngine
from ml_eval.core.implementations.local_matrix_adapter import LocalMatrixAdapter
from ml_eval.core.implementations.matrix_model import MatrixMultiplicationModel
from ml_eval.core.implementations.simple_model import SimpleModelAdapter
from ml_eval.core.implementations.image_classifier_model import ImageClassifierModel
from ml_eval.core.implementations.image_classifier_adapter import ImageClassifierAdapter
from ml_eval.core.implementations.accuracy_evaluator import AccuracyEvaluator
from ml_eval.core.implementations.exact_match import ExactMatchEvaluator
from ml_eval.database import crud

def main():
    """
    Main function to run the evaluation from the command line.
    """
    parser = argparse.ArgumentParser(description="Run an evaluation for a given ModelRun ID.")
    parser.add_argument("run_id", type=int, help="The ID of the ModelRun to evaluate.")
    args = parser.parse_args()

    run_id = args.run_id

    print(f"--- Setting up evaluation for ModelRun ID: {run_id} ---")

    # Get a database session
    db: Session = next(get_db())

    # 1. Check if the ModelRun exists
    model_run = crud.get_model_run(db, run_id=run_id)
    if not model_run:
        print(f"❌ Error: ModelRun with ID {run_id} not found.")
        return

    print("--- Initializing components ---")
    model_adapter = None
    evaluator = None

    if model_run.model_type == "matrix_multiplication":
        matrix_model = MatrixMultiplicationModel()
        model_adapter = LocalMatrixAdapter(model=matrix_model)
        evaluator = ExactMatchEvaluator()
    elif model_run.model_type == "simple_match":
        model_adapter = SimpleModelAdapter()
        evaluator = ExactMatchEvaluator() # Simple model can use exact match
    elif model_run.model_type == "image_classification":
        image_classifier_model = ImageClassifierModel()
        model_adapter = ImageClassifierAdapter(model=image_classifier_model)
        evaluator = AccuracyEvaluator()
    else:
        print(f"❌ Error: Unsupported model_type '{model_run.model_type}' for evaluation.")
        return
        
    print(f"✅ Components initialized for model_type: {model_run.model_type}.")

    # 2. Instantiate and run EvaluationEngine
    print("--- Instantiating Evaluation Engine ---")
    engine = EvaluationEngine(db=db, model_adapter=model_adapter, evaluator=evaluator)
    print("✅ Engine instantiated.")

    print(f"--- Running evaluation for ModelRun ID: {run_id} ---")
    completed_run = engine.run_evaluation(run_id)

    if completed_run and completed_run.status == "completed":
        print(f"\n🎉 Evaluation complete for ModelRun ID: {completed_run.id}")
        print(f"   - Total Cases: {completed_run.total_cases}")
        print(f"   - Completed: {completed_run.completed_cases}")
        print(f"   - Failed: {completed_run.failed_cases}")
    else:
        print(f"\n❌ Evaluation failed or did not complete for ModelRun ID: {run_id}")

if __name__ == "__main__":
    main()
