
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from ml_eval.database import crud
from ml_eval.database.models import ModelRun, TestPrompt, Response, Evaluation
from ml_eval.core.interfaces.imodel import IModelAdapter
from ml_eval.core.interfaces.ievaluator import IEvaluator, EvaluationResult

class EvaluationEngine:
    def __init__(self, db: Session, model_adapter: IModelAdapter, evaluator: IEvaluator):
        self.db = db
        self.model_adapter = model_adapter
        self.evaluator = evaluator

    def run_evaluation(self, run_id: int) -> Optional[ModelRun]:
        """
        Orchestrates the evaluation process for a given ModelRun.
        """
        model_run = crud.get_model_run(self.db, run_id=run_id)
        if not model_run:
            print(f"ModelRun with ID {run_id} not found.")
            return None

        print(f"Starting evaluation for ModelRun {run_id} ({model_run.model_name} {model_run.model_version})...")

        # Update ModelRun status to 'running'
        # Note: A dedicated update_model_run_status function in crud.py would be better
        model_run.status = "running"
        self.db.add(model_run)
        self.db.commit()
        self.db.refresh(model_run)

        test_prompts = crud.get_prompts_by_model_type(self.db, model_run.model_type)
        if not test_prompts:
            print(f"No test prompts found for model type '{model_run.model_type}'.")
            # Mark ModelRun as failed or completed with no tests
            model_run.status = "completed" # or 'failed' if no tests means failure
            self.db.add(model_run)
            self.db.commit()
            self.db.refresh(model_run)
            return model_run

        total_cases = len(test_prompts)
        completed_cases = 0
        failed_cases = 0

        for prompt in test_prompts:
            try:
                # 1. Get model output using the adapter
                model_output = self.model_adapter.run(prompt.input_data)

                # 2. Store response
                response = crud.create_response(
                    self.db,
                    run_id=model_run.id,
                    test_case_id=prompt.id,
                    output_data=model_output,
                    # Add latency, memory, tokens, error_message if available from adapter
                )

                # 3. Evaluate response
                eval_result: EvaluationResult = self.evaluator.evaluate(
                    expected=prompt.ground_truth,
                    actual=response.output_data
                )

                # 4. Store evaluation
                crud.create_evaluation(
                    self.db,
                    response_id=response.id,
                    evaluator_type=self.evaluator.__class__.__name__, # Use class name as type
                    score=eval_result.score,
                    passed=eval_result.passed,
                    metrics=eval_result.metrics,
                    feedback=eval_result.feedback
                )
                completed_cases += 1

            except Exception as e:
                print(f"Error processing prompt {prompt.id} for ModelRun {run_id}: {e}")
                # Optionally, store error message in Response
                crud.create_response(
                    self.db,
                    run_id=model_run.id,
                    test_case_id=prompt.id,
                    output_data={}, # Empty or error output
                    error_message=str(e)
                )
                failed_cases += 1
            
            finally:
                # Update model_run progress
                model_run.total_cases = total_cases
                model_run.completed_cases = completed_cases
                model_run.failed_cases = failed_cases
                self.db.add(model_run)
                self.db.commit()
                self.db.refresh(model_run)


        # 5. Mark ModelRun as complete
        # This will also set completed_at and update status to 'completed'
        crud.complete_model_run(self.db, model_run.id)
        print(f"Evaluation for ModelRun {run_id} completed.")

        return crud.get_model_run(self.db, run_id=run_id)

