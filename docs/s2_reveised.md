### Day 1 FastAPI Setup & Connection

    - [] T2.1.1	Install fastapi, uvicorn, pydantic.
    
    Add those two the environment.yml and run conda env update --file environment.yml, followed by conda activate <nameofcondvenv>,  These actions will install, and give you access to the three dependencies immediately. 

    - [] T2.1.2	Create the core FastAPI application file (main.py).

    Checked this with  ' uvicorn ml_eval.main:app --reload --port 8000' and it ran, evaentually.  Went back and forth devising a passive/active check of if the applciation could chat with the database.  It does. Will add some more stuff in test for this.
    
    - [] T2.1.3	Implement the get_db Dependency function to manage SQLAlchemy sessions (using try/finally).
    - [] T2.1.4	Define the base Pydantic Schemas (schemas.py) for all four entities (e.g., PromptCreate, PromptResponse, RunStart).

### Day 2-3: CRUD API Endpoints
    wrap the crud.py functions with FastAPI routes.
    - [] T2.2.1	POST /prompts/	crud.create_prompt
    - [] T2.2.2	GET /prompts/{id}	crud.get_prompt
    - [] T2.2.3	POST /runs/	crud.create_model_run
    - [] T2.2.4	POST /runs/{id}/complete	crud.complete_model_run (Update)
    - [] T2.2.5	GET /prompts/domain/{domain}	crud.get_prompts_by_domain

### Day 4: Core Pluggable Architecture
    defining the interfaces that ensure universality.
    - [] T2.3.1	core/interfaces/imodel.py	Define the IModelAdapter abstract base class (ABC) with a single method: run(input: dict) -> dict.
    - [] T2.3.2	core/interfaces/ievaluator.py	Define the IEvaluator abstract base class (ABC) with a single method: evaluate(expected: dict, actual: dict) -> EvaluationResult.
    - [] T2.3.3	core/implementations/simple_model.py	Create a dummy implementation of IModelAdapter that just returns a mocked output (e.g., adds "processed: True" to the input).
    - [] T2.3.4	core/implementations/exact_match.py	Create a dummy implementation of IEvaluator that checks if the actual output exactly matches the expected output.

#### Day 5-7: Testing & Documentation
    service layer is robust and documenting the new API.
    - [] T2.4.1	Set up pytest-asyncio for testing asynchronous FastAPI endpoints.
    - [] T2.4.2	Write unit tests for the CRUD API endpoints (e.g., ensure POST and GET work correctly).
    - [] T2.4.3	Write unit tests for the Adapter/Evaluator Interfaces, ensuring the dummy implementations adhere to the ABCs.
    - [] T2.4.4	Use pytest --cov to verify API test coverage is ≥90%.
    - [] T2.4.5	Update the main README with the new API routes and instructions on how to start the FastAPI server (uvicorn).


##  Deliverables Checklist

### Code & Config Deliverables

    - [ ] main.py (FastAPI instance)
    - [ ] schemas.py (Pydantic models)
    - [ ] routers/crud.py (All CRUD API endpoints)
    - [ ] core/interfaces/imodel.py (IModelAdapter ABC)
    - [ ] core/interfaces/ievaluator.py (IEvaluator ABC)
    - [ ] core/implementations/simple_model.py (Dummy Adapter)
    - [ ] core/implementations/exact_match.py (Dummy Evaluator)

### Test Deliverables

    - [ ] tests/test_api/test_crud_endpoints.py (API tests)
    - [ ] tests/test_core/test_interfaces.py (Interface/ABC adherence tests)
    - [ ] Test coverage ≥90% for ml_eval source code


## 🎯 Definition of Done

Sprint is "done" when:

    - [ ] The FastAPI server starts successfully and connects to the PostgreSQL database.
    - [ ] All Pydantic schemas correctly handle and validate JSONB data types.
    - [ ] All CRUD API endpoints are functional and return correct HTTP status codes.
    - [ ] All unit tests pass, achieving ≥90% test coverage for the new service layer.
    - [ ] The IModelAdapter and IEvaluator interfaces are defined and tested, proving the pluggable architecture is ready for domain-specific implementation in Sprint 3.