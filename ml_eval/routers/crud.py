
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ml_eval.database import crud
from ml_eval.database.connection import get_db
from ml_eval import schemas

router = APIRouter()

@router.post("/prompts/", response_model=schemas.TestPrompt)
def create_prompt(prompt: schemas.TestPromptBase, db: Session = Depends(get_db)):
    return crud.create_prompt(
        db=db,
        test_case_name=prompt.test_case_name,
        model_type=prompt.model_type,
        input_type=prompt.input_type,
        output_type=prompt.output_type,
        input_data=prompt.input_data,
        ground_truth=prompt.ground_truth,
        category=prompt.category,
        tags=prompt.tags,
        difficulty=prompt.difficulty,
        origin=prompt.origin,
        is_verified=prompt.is_verified,
        test_case_metadata=prompt.test_case_metadata,
        created_by=prompt.created_by,
    )

@router.get("/prompts/{prompt_id}", response_model=schemas.TestPrompt)
def read_prompt(prompt_id: int, db: Session = Depends(get_db)):
    db_prompt = crud.get_prompt(db, prompt_id=prompt_id)
    if db_prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return db_prompt

@router.get("/prompts/domain/{model_type}", response_model=List[schemas.TestPrompt])
def read_prompts_by_domain(model_type: str, db: Session = Depends(get_db)):
    return crud.get_prompts_by_model_type(db, model_type=model_type)

@router.put("/prompts/{prompt_id}", response_model=schemas.TestPrompt)
def update_prompt(prompt_id: int, prompt: schemas.TestPromptBase, db: Session = Depends(get_db)):
    update_data = schemas.TestPromptUpdate(**prompt.model_dump())
    db_prompt = crud.update_prompt(db, prompt_id=prompt_id, prompt_update=update_data)
    if db_prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return db_prompt

@router.patch("/prompts/{prompt_id}", response_model=schemas.TestPrompt)
def patch_prompt(prompt_id: int, prompt: schemas.TestPromptUpdate, db: Session = Depends(get_db)):
    db_prompt = crud.update_prompt(db, prompt_id=prompt_id, prompt_update=prompt)
    if db_prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return db_prompt

@router.delete("/prompts/{prompt_id}", status_code=204)
def delete_prompt(prompt_id: int, db: Session = Depends(get_db)):
    if not crud.delete_prompt(db, prompt_id=prompt_id):
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"ok": True}

@router.post("/runs/", response_model=schemas.ModelRun)
def create_model_run(model_run: schemas.ModelRunCreate, db: Session = Depends(get_db)):
    return crud.create_model_run(
        db=db,
        model_name=model_run.model_name,
        model_version=model_run.model_version,
        model_type=model_run.model_type,
        model_endpoint=model_run.model_endpoint,
        config=model_run.config,
        # status=model_run.status, # Status has a default value, can be overridden later
        # total_cases=model_run.total_cases, # Default value
        # completed_cases=model_run.completed_cases, # Default value
        # failed_cases=model_run.failed_cases, # Default value
    )

@router.get("/runs/{run_id}", response_model=schemas.ModelRun)
def read_model_run(run_id: int, db: Session = Depends(get_db)):
    db_run = crud.get_model_run(db, run_id=run_id)
    if db_run is None:
        raise HTTPException(status_code=404, detail="ModelRun not found")
    return db_run


@router.post("/runs/{run_id}/complete", response_model=schemas.ModelRun)
def complete_model_run(run_id: int, db: Session = Depends(get_db)):
    db_run = crud.complete_model_run(db, model_run_id=run_id)
    if db_run is None:
        raise HTTPException(status_code=404, detail="ModelRun not found")
    return db_run

@router.put("/runs/{run_id}", response_model=schemas.ModelRun)
def update_model_run(run_id: int, run: schemas.ModelRunCreate, db: Session = Depends(get_db)):
    update_data = schemas.ModelRunUpdate(**run.model_dump())
    db_run = crud.update_model_run(db, run_id=run_id, run_update=update_data)
    if db_run is None:
        raise HTTPException(status_code=404, detail="ModelRun not found")
    return db_run

@router.patch("/runs/{run_id}", response_model=schemas.ModelRun)
def patch_model_run(run_id: int, run: schemas.ModelRunUpdate, db: Session = Depends(get_db)):
    db_run = crud.update_model_run(db, run_id=run_id, run_update=run)
    if db_run is None:
        raise HTTPException(status_code=404, detail="ModelRun not found")
    return db_run

@router.delete("/runs/{run_id}", status_code=204)
def delete_model_run(run_id: int, db: Session = Depends(get_db)):
    if not crud.delete_model_run(db, run_id=run_id):
        raise HTTPException(status_code=404, detail="ModelRun not found")
    return {"ok": True}
