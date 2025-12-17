## RECAP  
### What the?
So i jumped back into sprint 3 after a week or so, and I fully remmember that that i had done implemented all the CRUD ops.  

Welp. I was wrong.  Guess i just did PUT.
Sprint 2.5 was focused on getting the foundational FastApi architecture, and pluggable interface for models/evaluators, and basic CRUD API endpoints.  

So i did some junk on S3-frogged around with making the schema 'correct' for all the E2E eval. workflow.  I got a bunch wrong/halfdone. (ie garbage, why stop working when you know whats going on?)

I didnt actually check to see if s2/s3 actually worked...jsut thought my tests were good enough. They weren't. They test the logic of the functions, not the integration. I had no integration tests. So it all fell apart. 

I should have been able to start the application
```
conda activate ml-eval-framework
docker-compose up -d  db (or in my case ./start_db.sh)
uvicorn ml_eval.main:app --reload --port 8000 (in a different terminal)
```
, and then goto localhost:8000/docs and explore the api. 
or test it like this:
```

curl -X 'POST'   'http://localhost:8000/api/v1/prompts/'   -H 'Content-Type: application/json'   -d '{
    "test_case_name": "original_name",
    "model_type": "nlp",
    "input_type": "text",
    "output_type": "text",
    "input_data": {"prompt": "original_prompt"},
    "ground_truth": {"answer": "original_answer"}
  }'
{"test_case_name":"original_name","model_type":"nlp","input_type":"text","output_type":"text","input_data":{"prompt":"original_prompt"},"ground_truth":{"answer":"original_answer"},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":1,"created_at":"2025-12-17T04:29:03.619825","updated_at":"2025-12-17T04:29:03.619831"}
```

Thats what should habve been the result, instead only GET was available so i had to make a PUT, PATCH and DELETE. (what did i actually do last time?Good lord.)  So i added those endpoints in for both TestCase (prompts) and ModelRun resources.  I had to update the pydantic shcemas, frog around with the database logic and define the new routes.  Cool...stay on target, luke.

So update ml_eval/schemas.py with TestPromptUpdate and ModelRunUpdate, implement update_prompt, delet)prompt, update)model)run, and delete_model_run in database/crud.py. Next, add PUT, PATCH, and DELETE to /prompts/{prompt_id}. PUT will take a TestPromptBase, convert it to TestPromptUpdate, and call crud.update_prompt. PATCH will use TestPromptUpdate to call crud.update_prompt. DELETE will call crud.delete_prompt. Also add the PUT, PATCH, and DELETE endpoints for /runs/{run_id}. PUT will use ModelRunCreate and call update_model_run. PATCH  will use ModelRunUpdate and call update_model_run. DELETE will call delete_model_run.

Okay, I've added CRUD endpoints for prompts and runs in routers/crud.py. Now I'll fix two issues: first, I'll remove the string-based  type hints in crud.py and use direct imports instead. Second, I'll correct the TestPromptBase type hints in routers/crud.py.  I've fixed the type hints in ml_eval/database/crud.py, but now I see a problem in ml_eval/routers/crud.py: the update functions have the  wrong type hints. I'll correct them to use schemas.TestPromptUpdate and schemas. ModelRunUpdate for both PUT and PATCH to be pragmatic,  ensuring the endpoints function as requested even if it's not strictly RESTful.   OK try to stay in your lane, Luke. You are not even close to the death star anymor. GOlly. I just threw some junk at the wall and really hoped it would stick. 

OK i added some tests. They passed. Hooray?

I did find some wierdo typos. I named some functions wrong, fixed some imports, and did some other sundry error chase downs and my tests worked. Ok, i think. Lets see if we can make the actual app do something.

So get the fastapi and database running in the virtual environment and run some comammands of the curl variety that show your app is running and connected to the db.  I got a bunch of 404 errors and attempted a bunch of stuff, including checking is the port correct, is the app running, am I in an activated venv, and finally discoverd my database doesn't have the right tables. 
so
```
almembic upgrade head
```
from root of my project should do the trick....doddeeedodo. No dice. Close the app, shut down container, rerun the above command, open up the app again, bring the container back up and check into the db:
```
ocker exec -it ml_eval_postgres psql -U ml_user -d ml_eval_db
psql (15.15)
Type "help" for help.

ml_eval_db=# \dt
             List of relations
 Schema |      Name       | Type  |  Owner  
--------+-----------------+-------+---------
 public | alembic_version | table | ml_user
(1 row)

ml_eval_db=# \q

```

Thats not what I'm looking for.  SO I know alembic works, I have seen it do its thing in this app before.  How does almebic knwo when/where to look? Good question.Alembic works in two stages:
   1. Generation: It compares the models in ml_eval/database/models.py to its last known migration and generates a new migration script
      containing the differences.
   2. Application: It runs the new script (using alembic upgrade head) to apply those changes to the database.

I completed Sprint 3 and did not perform step 1. No migration script was ever created for the new schema. Therefore, when  you run alembic upgrade head, Alembic sees no new scripts to apply and does nothing.   The solution is to generate the missing migration script first.  You gotta run this command and you should see output similar to below.

```
alembic revision --autogenerate -m "Add Sprint 3 schema changes"                                                                                                                                         
│ Generating /home/dell-linux-dev3/Projects/ml-evaluation-framework/migrations/versions/d7c66d5e9ce2_add_sprint_3_schema_cha             │
│ nges.py ...  done                                                                                                                      │
│                                                                                                                                        │
│ INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.                                                                         │
│ INFO  [alembic.runtime.migration] Will assume transactional DDL.                                                                       │
│ INFO  [alembic.autogenerate.compare] Detected added table 'model_runs'                                                                 │
│ INFO  [alembic.autogenerate.compare] Detected added index 'ix_model_runs_id' on '('id',)'                                              │
│ INFO  [alembic.autogenerate.compare] Detected added index 'ix_model_runs_model_name' on '('model_name',)'                              │
│ INFO  [alembic.autogenerate.compare] Detected added table 'test_cases'                                                                 │
│ INFO  [alembic.autogenerate.compare] Detected added index 'ix_test_cases_category' on '('category',)'                                  │
│ INFO  [alembic.autogenerate.compare] Detected added index 'ix_test_cases_id' on '('id',)'                                              │
│ INFO  [alembic.autogenerate.compare] Detected added index 'ix_test_cases_model_type' on '('model_type',)'                              │
│ INFO  [alembic.autogenerate.compare] Detected added index 'ix_test_cases_test_case_name' on '('test_case_name',)'                      │
│ INFO  [alembic.autogenerate.compare] Detected added table 'responses'                                                                  │
│ INFO  [alembic.autogenerate.compare] Detected added index 'ix_responses_id' on '('id',)'                                               │
│ INFO  [alembic.autogenerate.compare] Detected added index 'ix_responses_run_id' on '('run_id',)'                                       │
│ INFO  [alembic.autogenerate.compare] Detected added index 'ix_responses_test_case_id' on '('test_case_id',)'                           │
│ INFO  [alembic.autogenerate.compare] Detected added table 'evaluations'                                                                │
│ INFO  [alembic.autogenerate.compare] Detected added index 'ix_evaluations_id' on '('id',)'                                             │
│ INFO  [alembic.autogenerate.compare] Detected added index 'ix_evaluations_response_id' on '('response_id',)'   
```

Now the previous command worked perfectly.  As you can see from the output, Alembic detected all the new tables (test_cases, model_runs, etc.) and created a new migration script named d7c66d5e9ce2_add_sprint_3_schema_changes.py to represent these changes.  Now that the missing script exists, you can finally run the upgrade command to apply it to your database. This will create the tables.

Below you can see succesfull curl commands exectued :

```
A)UDATE with alembic
/Projects/ml-evaluation-framework$ conda run -n ml-eval-framework alembic upgrade head
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.

B)  STEP INTO THE DB AND VERIFY IT WORKED.
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker exec -it ml_eval_postgres psql -U ml_user -d ml_eval_db
psql (15.15)
Type "help" for help.

ml_eval_db=# \dt
             List of relations
 Schema |      Name       | Type  |  Owner  
--------+-----------------+-------+---------
 public | alembic_version | table | ml_user
 public | evaluations     | table | ml_user
 public | model_runs      | table | ml_user
 public | responses       | table | ml_user
 public | test_cases      | table | ml_user
(5 rows)

ml_eval_db=# \q

C) DO A POST NOTE THE ID (1)
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'POST'   'http://localhost:8000/api/v1/prompts/'   -H 'Content-Type: application/json'   -d '{
    "test_case_name": "original_name",
    "model_type": "nlp",
    "input_type": "text",
    "output_type": "text",
    "input_data": {"prompt": "original_prompt"},
    "ground_truth": {"answer": "original_answer"}
  }'
{"test_case_name":"original_name","model_type":"nlp","input_type":"text","output_type":"text","input_data":{"prompt":"original_prompt"},"ground_truth":{"answer":"original_answer"},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":1,"created_at":"2025-12-17T04:29:03.619825","updated_at":"2025-12-17T04:29:03.619831"}

D) PERFORM A PATCH
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluatcurl -X 'PATCH' \rl -X 'PATCH' \
  'http://localhost:8000/api/v1/prompts/1' \
  -H 'Content-Type: application/json' \
  -d '{
  "test_case_name": "patched_name_for_prompt"
}'
{"test_case_name":"patched_name_for_prompt","model_type":"nlp","input_type":"text","output_type":"text","input_data":{"prompt":"original_prompt"},"ground_truth":{"answer":"original_answer"},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":1,"created_at":"2025-12-17T04:29:03.619825","updated_at":"2025-12-17T05:22:39.537641"}

E)  CHECK IF PUT WORKS
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/curl -X 'PUT' \ramework$ curl -X 'PUT' \
  'http://localhost:8000/api/v1/prompts/1' \
  -H 'Content-Type: application/json' \
  -d '{
  "test_case_name": "put_name_for_prompt",
  "model_type": "cv",
  "input_type": "image",
  "output_type": "classification",
  "input_data": {"path": "/new/path"},
  "ground_truth": {"label": "new_label"}
}'
{"test_case_name":"put_name_for_prompt","model_type":"cv","input_type":"image","output_type":"classification","input_data":{"path":"/new/path"},"ground_truth":{"label":"new_label"},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":1,"created_at":"2025-12-17T04:29:03.619825","updated_at":"2025-12-17T05:22:50.956222"}

F))CHECK IF THAT PUT WORKED WITH THIS GET 
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluacurl -X 'GET' 'http://localhost:8000/api/v1/prompts/1'pi/v1/prompts/1'
{"test_case_name":"put_name_for_prompt","model_type":"cv","input_type":"image","output_type":"classification","input_data":{"path":"/new/path"},"ground_truth":{"label":"new_label"},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":1,"created_at":"2025-12-17T04:29:03.619825","updated_at":"2025-12-17T05:22:50.956222"}

G)  DELETE THAT 
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluacurl -X 'DELETE' 'http://localhost:8000/api/v1/prompts/1'pi/v1/prompts/1'

H)  CHECK IF DELETE WORKED.
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'GET' 'http://localhost:8000/api/v1/prompts/1'
{"detail":"Prompt not found"}(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ 



```

In a nutshell:
Today's session focused on restoring the development environment after a system crash and evolving the ML Evaluation Framework from a basic prototype into a complete CRUD-capable testing platform.

#### 1. Environment Restoration & Recovery

    Context Re-alignment: The session began by reviewing the Sprint 2 and Sprint 3 documentation to understand the current state of the foundational FastAPI architecture and the newly implemented end-to-end evaluation workflow.

Infrastructure Check: Verified the readiness of the PostgreSQL database (via Docker) and the Conda environment (ml-eval-framework).

#### 2. Implementation of Full CRUD Capabilities

While the previous state supported creating and reading test cases, you identified the need for a full suite of management tools. The following was accomplished to meet this requirement:

    Schema Updates: Created new Pydantic schemas (TestPromptUpdate and ModelRunUpdate) with optional fields to support partial updates.

Database Logic: Added update_prompt, delete_prompt, update_model_run, and delete_model_run functions to the database layer. These use dynamic setattr calls to handle partial PATCH requests efficiently.

API Endpoints: Exposed new PUT, PATCH, and DELETE routes for both /prompts/ and /runs/.

    PUT: Performs a full resource replacement.

PATCH: Allows updating specific fields without overwriting the entire record.

DELETE: Removes records and verifies their absence in subsequent requests.

#### 3. Testing & Verification

    Automated Tests: Expanded the integration test suite (test_crud_endpoints.py) to include 11+ tests covering the new update and deletion lifecycles.

Manual Validation: You successfully executed a manual POST test which confirmed that the API is correctly creating test cases with assigned IDs (e.g., id: 1) and appropriate metadata timestamps.

#### 4. Technical Refinement

    Code Integrity: Corrected type hints and fixed a schema import error to ensure the application runs stably in a local environment.

Documentation: Provided a streamlined set of curl commands for manual testing of the new features in your local terminal.

TLDR:
After discovering that the "completed" Sprint 3 was actually missing core integration and database tables, the session shifted to building out the full PUT, PATCH, and DELETE lifecycle for all resources. By synchronizing the Alembic migrations with the new Pydantic schemas and database logic, the framework was successfully transformed into a verified, CRUD-complete platform.



FINALLY:  
All the other goals for sprint3 were actually completed.  I had to amend how the project takes in the models under test, and how it will do it in the future is up for debate.  Anyways, I have two very simple tests that it tests and it works.  BOOOM!

```
:~/Projects/ml-evaluation-framework$ curl -X 'POST'   'http://localhost:8000/api/v1/runs/'   -H 'Content-Type: application/json'   -d '{
  "model_name": "SimpleModel-Test-2",
  "model_version": "1.0",
  "model_type": "simple_match"
}'
{"model_name":"SimpleModel-Test-2","model_version":"1.0","model_type":"simple_match","model_endpoint":null,"config":{},"id":2,"status":"pending","started_at":"2025-12-17T05:58:33.258277","completed_at":null,"total_cases":0,"completed_cases":0,"failed_cases":0}(ml-eval-framework) dell-linux-decurl -X 'POST'   'http://localhost:8000/api/v1/prompts/'   -H 'Content-Type: application/json'   -d '{
    "test_case_name": "Simple Match Test 2",
    "model_type": "simple_match",
    "input_type": "text",
    "output_type": "json",
    "input_data": {"text": "This is a simple input."},
    "ground_truth": {"text": "This is a simple input.", "processed": true}
  }'
{"test_case_name":"Simple Match Test 2","model_type":"simple_match","input_type":"text","output_type":"json","input_data":{"text":"This is a simple input."},"ground_truth":{"text":"This is a simple input.","processed":true},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":3,"created_at":"2025-12-17T05:59:15.787208","updated_at":"2025-12-17T05:59:15.787213"}(ml-eval-framework) dell-linux-dev3@dell-linpython scripts/run_evaluation.py 2
--- Setting up evaluation for ModelRun ID: 2 ---
--- Initializing components ---
✅ Components initialized for model_type: simple_match.
--- Instantiating Evaluation Engine ---
✅ Engine instantiated.
--- Running evaluation for ModelRun ID: 2 ---
Starting evaluation for ModelRun 2 (SimpleModel-Test-2 1.0)...
Evaluation for ModelRun 2 completed.

🎉 Evaluation complete for ModelRun ID: 2
   - Total Cases: 2
   - Completed: 2
   - Failed: 0
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'POST' \
  'http://localhost:8000/api/v1/runs/' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_name": "MatrixModel-Test-3x3",
  "model_version": "1.0",
  "model_type": "matrix_multiplication"
}'
{"model_name":"MatrixModel-Test-3x3","model_version":"1.0","model_type":"matrix_multiplication","model_endpoint":null,"config":{},"id":3,"status":"pending","started_at":"2025-12-17T06:03:38.563723","completed_at":null,"total_cases":0,"completed_cases":0,"failed_cases":0}(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'POST'curl -X 'POST' \
  'http://localhost:8000/api/v1/prompts/' \
  -H 'Content-Type: application/json' \
  -d '{
    "test_case_name": "3x3 Identity Matrix Test",
    "model_type": "matrix_multiplication",
    "input_type": "json",
    "output_type": "json",
    "input_data": {
        "matrix_a": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "matrix_b": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    },
    "ground_truth": {"result_matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
  }'
{"test_case_name":"3x3 Identity Matrix Test","model_type":"matrix_multiplication","input_type":"json","output_type":"json","input_data":{"matrix_a":[[1,0,0],[0,1,0],[0,0,1]],"matrix_b":[[1,2,3],[4,5,6],[7,8,9]]},"ground_truth":{"result_matrix":[[1,2,3],[4,5,6],[7,8,9]]},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":4,"created_at":"2025-12-17T06:04:03.222147","updated_at":"2025-12-17T06:04:03.222154"}(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-python scripts/run_evaluation.py 3
--- Setting up evaluation for ModelRun ID: 3 ---
--- Initializing components ---
✅ Components initialized for model_type: matrix_multiplication.
--- Instantiating Evaluation Engine ---
✅ Engine instantiated.
--- Running evaluation for ModelRun ID: 3 ---
Starting evaluation for ModelRun 3 (MatrixModel-Test-3x3 1.0)...
Evaluation for ModelRun 3 completed.

🎉 Evaluation complete for ModelRun ID: 3
   - Total Cases: 1
   - Completed: 1
   - Failed: 0
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ 



```

So the eval works on simple stuff.  Now lets see if we can get it to do similar evals on more complex types of calculations.  How much sentiment analysis will I have to use?  VADER, get over here.