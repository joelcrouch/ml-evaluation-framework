import os
import sys
from datetime import datetime
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from typing import Dict, Any

# Adjust Python Path to find your 'ml_eval' package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules
try:
    from ml_eval.database.connection import SessionLocal
    from ml_eval.database.crud import (
        create_prompt, create_model_run, create_response, create_evaluation, complete_model_run
    )
    from ml_eval.database.models import Base, TestPrompt
except ImportError as e:
    print(f"FATAL ERROR: Failed to import database modules. Error: {e}")
    sys.exit(1)

def seed_data(db: Session):
    """Inserts multi-domain test data to demonstrate the universal schema."""
    print("--- Starting database seeding with multi-domain data ---")

    # --- 1. Create Multi-Domain TestPrompts (Golden Sets) ---

    # 1A. NLP Prompt (Question Answering)
    nlp_input: Dict[str, Any] = {"text": "What is the capital of France?"}
    nlp_expected: Dict[str, Any] = {"answer": "Paris"}
    
    prompt_nlp = create_prompt(
        db,
        name="QA-France-Capital",
        domain="NLP",
        input_data=nlp_input,
        expected_output=nlp_expected,
        origin="human",
        is_verified=True
    )
    print(f"✅ Created NLP Prompt (ID: {prompt_nlp.id}, Domain: {prompt_nlp.domain})")

    # 1B. Computer Vision Prompt (Object Detection)
    cv_input: Dict[str, Any] = {"image_path": "s3://golden-sets/cv/cat_dog.jpg", "classes": ["cat", "dog"]}
    cv_expected: Dict[str, Any] = {
        "annotations": [
            {"box": [10, 20, 100, 120], "label": "cat"},
            {"box": [150, 160, 250, 260], "label": "dog"}
        ]
    }
    prompt_cv = create_prompt(
        db,
        name="CV-Object-Detection-Sample",
        domain="CV",
        input_data=cv_input,
        expected_output=cv_expected,
        origin="ai_generated", # Example of non-human origin
        is_verified=False
    )
    print(f"✅ Created CV Prompt (ID: {prompt_cv.id}, Domain: {prompt_cv.domain})")

    # --- 2. Create Model Runs (Version Tracking) ---
    
    # Run 1: NLP Model V1.0.0
    run_nlp_v1 = create_model_run(db, model_name="LLM-A-2024", model_version="1.0.0")
    print(f"✅ Created NLP Model Run (ID: {run_nlp_v1.id}, Version: {run_nlp_v1.model_version})")

    # Run 2: CV Model V2.1
    run_cv_v2 = create_model_run(db, model_name="YOLOv8-Small", model_version="2.1")
    print(f"✅ Created CV Model Run (ID: {run_cv_v2.id}, Version: {run_cv_v2.model_version})")

    # --- 3. Create Responses (Linking Prompt and Run) ---
    
    # Response for NLP Prompt (Correct answer)
    response_nlp = create_response(
        db, 
        prompt_id=prompt_nlp.id, 
        model_run_id=run_nlp_v1.id, 
        output_data={"text": "Paris, the capital city."}
    )
    print(f"✅ Created Response for NLP Run (ID: {response_nlp.id})")

    # Response for CV Prompt (Slight error: missed the dog)
    response_cv = create_response(
        db, 
        prompt_id=prompt_cv.id, 
        model_run_id=run_cv_v2.id, 
        output_data={
            "predictions": [
                {"box": [12, 22, 102, 122], "label": "cat", "confidence": 0.98}
            ]
        }
    )
    print(f"✅ Created Response for CV Run (ID: {response_cv.id})")


    # --- 4. Create Evaluations (Metrics/Scoring) ---
    
    # Evaluation 1: Simple Pass/Fail for NLP
    eval_nlp = create_evaluation(
        db,
        response_id=response_nlp.id,
        evaluator_name="ExactMatch",
        score=1.0, # Perfect score
        is_pass=True,
        details={"case_sensitivity": False}
    )
    print(f"✅ Created Evaluation for NLP Response (Score: {eval_nlp.score})")
    
    # Evaluation 2: IoU Metric for CV (Partial Score)
    eval_cv = create_evaluation(
        db,
        response_id=response_cv.id,
        evaluator_name="IoU_0.5",
        score=0.45, # Failed the threshold
        is_pass=False,
        details={"missing_objects": ["dog"]}
    )
    print(f"✅ Created Evaluation for CV Response (Score: {eval_cv.score})")

    # --- 5. Complete Model Runs ---
    complete_model_run(db, run_nlp_v1.id)
    complete_model_run(db, run_cv_v2.id)
    print("✅ Completed both Model Runs.")
    
    print("\n🎉 Database seeding complete. Ready for full testing!")


if __name__ == "__main__":
    # Ensure tables are created first (optional if Alembic was run)
    Base.metadata.create_all(bind=SessionLocal().bind)
    
    try:
        db = SessionLocal()
        seed_data(db)
        db.close()
    except IntegrityError as e:
        print(f"\n🛑 Error: Integrity Error during seeding. Data may already exist.")
        print("Please ensure your database is empty or handle data conflicts.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n🛑 An unexpected error occurred during seeding: {e}")
        sys.exit(1)