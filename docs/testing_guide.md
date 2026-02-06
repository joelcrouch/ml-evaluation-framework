# ML Evaluation Framework - Testing Guide

**Last Updated**: 2026-02-05
**Purpose**: Comprehensive guide for testing each sprint's functionality

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Test Environment Setup](#test-environment-setup)
3. [Sprint 1 Testing](#sprint-1-testing)
4. [Sprint 2 Testing](#sprint-2-testing)
5. [Sprint 3 Testing](#sprint-3-testing)
6. [Integration Testing](#integration-testing)
7. [Manual Testing Procedures](#manual-testing-procedures)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This guide explains how to test each sprint's functionality, what is being tested, and why it matters. Tests are organized by sprint to match the development workflow.

### Testing Philosophy

Our testing strategy follows these principles:
1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test components working together
3. **Manual Tests**: Verify end-to-end workflows
4. **Database Tests**: Ensure data integrity and relationships

---

## Test Environment Setup

### Prerequisites

```bash
# 1. Ensure you're in the project root
cd /home/dell-linux-dev3/Projects/ml-evaluation-framework

# 2. Activate conda environment
conda activate ml-eval-framework

# 3. Start test database
docker-compose up -d db

# 4. Create test database (one-time setup)
PGPASSWORD=ml_password createdb -h localhost -p 5433 -U ml_user ml_eval_db_test
```

### Running All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=ml_eval --cov-report=html

# Run specific test file
pytest tests/test_database/test_crud.py

# Run specific test function
pytest tests/test_database/test_crud.py::test_create_prompt_with_defaults
```

### Test Structure

```
tests/
├── conftest.py              # Shared test fixtures
├── test_database/           # Sprint 1: Database tests
│   ├── test_connection.py   # Connection management
│   └── test_crud.py         # CRUD operations
├── test_api/                # Sprint 1: API tests
│   └── test_crud_endpoints.py
├── test_core/               # Sprint 1-2: Interface tests
│   └── test_interfaces.py
├── test_query_engine/       # Sprint 3: Evaluation engine tests
│   └── test_engine.py
└── test_suite/              # Sprint 2: Test suite manager (TBD)
    └── (to be implemented)
```

---

## Sprint 1 Testing

**Sprint Goal**: Verify universal database schema, CRUD operations, and API endpoints

### What We're Testing

Sprint 1 tests ensure:
1. ✅ Database connection and session management work
2. ✅ Tables are created with correct schema
3. ✅ JSONB columns store and retrieve complex data
4. ✅ Foreign key relationships work correctly
5. ✅ User-first defaults are applied (`origin='human'`, `is_verified=TRUE`)
6. ✅ CRUD operations work for all entities
7. ✅ API endpoints expose database operations correctly
8. ✅ Cascade deletes work properly

### Database Tests

Location: `tests/test_database/`

#### 1. Connection Tests (`test_connection.py`)

**Purpose**: Verify database connectivity and session management

```bash
pytest tests/test_database/test_connection.py -v
```

**What's Tested**:
- Database URL construction
- Connection pooling
- Session lifecycle management
- `get_db()` dependency injection

**Why It Matters**: These tests ensure the application can reliably connect to PostgreSQL and manage database sessions without leaks.

#### 2. CRUD Operation Tests (`test_crud.py`)

**Purpose**: Verify all database operations work correctly

```bash
pytest tests/test_database/test_crud.py -v
```

**Test Cases**:

##### a) `test_create_prompt_with_defaults`
- **What**: Creates a test case and verifies defaults
- **Why**: Ensures user-first philosophy (`origin='human'`, `is_verified=TRUE`)
- **Verifies**:
  - JSONB storage for `input_data` and `ground_truth`
  - Default values are applied
  - Metadata storage works
  - Read operations retrieve correct data

##### b) `test_model_run_lifecycle`
- **What**: Creates and completes a model run
- **Why**: Verifies status tracking and timestamp management
- **Verifies**:
  - Initial status is "pending"
  - `started_at` is set on creation
  - `completed_at` is set on completion
  - Status changes to "completed"

##### c) `test_end_to_end_data_flow_and_foreign_keys`
- **What**: Tests full pipeline with all 4 tables
- **Why**: Ensures foreign key relationships and cascades work
- **Verifies**:
  - TestCase → Response relationship
  - ModelRun → Response relationship
  - Response → Evaluation relationship
  - SQLAlchemy ORM navigation works
  - JSONB data flows through pipeline

##### d) `test_unique_constraint_on_response`
- **What**: Attempts duplicate response creation
- **Why**: Prevents duplicate evaluations for same test case + run
- **Verifies**:
  - UNIQUE constraint on (test_case_id, run_id) works
  - Database raises IntegrityError on violation

**Running Database Tests**:
```bash
# All database tests
pytest tests/test_database/ -v

# Single test with output
pytest tests/test_database/test_crud.py::test_end_to_end_data_flow_and_foreign_keys -v -s
```

### API Endpoint Tests

Location: `tests/test_api/test_crud_endpoints.py`

**Purpose**: Verify FastAPI endpoints expose database operations correctly

```bash
pytest tests/test_api/test_crud_endpoints.py -v
```

**Test Cases**:

##### a) `test_create_and_read_prompt`
- **What**: POST to create, GET to retrieve
- **Why**: Verifies basic CRUD via HTTP
- **Verifies**:
  - POST `/api/v1/prompts/` returns 200
  - Response includes generated ID
  - GET `/api/v1/prompts/{id}` retrieves correct data
  - JSONB fields serialize properly

##### b) `test_read_prompts_by_domain`
- **What**: Filters test cases by model_type
- **Why**: Essential for Sprint 3 evaluation workflow
- **Verifies**:
  - GET `/api/v1/prompts/domain/{model_type}` works
  - Filtering returns only matching records
  - Multiple test cases can be retrieved

##### c) `test_create_and_complete_model_run`
- **What**: Creates run, then marks complete
- **Why**: Tests state management via API
- **Verifies**:
  - POST `/api/v1/runs/` creates run with "pending" status
  - POST `/api/v1/runs/{id}/complete` updates status
  - Timestamps are set correctly

##### d) `test_update_prompt`
- **What**: Tests PUT (full replace) and PATCH (partial update)
- **Why**: Allows modifying test cases after creation
- **Verifies**:
  - PATCH updates only specified fields
  - PUT replaces entire resource
  - Timestamps update correctly

##### e) `test_delete_prompt` & `test_delete_model_run`
- **What**: Tests DELETE operations
- **Why**: Cleanup and test management
- **Verifies**:
  - DELETE returns 204 on success
  - GET returns 404 after deletion
  - Deleting non-existent resource returns 404

**Running API Tests**:
```bash
# All API tests
pytest tests/test_api/ -v

# Test with coverage
pytest tests/test_api/ --cov=ml_eval.routers
```

### Core Interface Tests

Location: `tests/test_core/test_interfaces.py`

**Purpose**: Verify model adapters and evaluators adhere to interfaces

```bash
pytest tests/test_core/test_interfaces.py -v
```

**What's Tested**:
- `SimpleModelAdapter` implements `IModelAdapter` correctly
- `ExactMatchEvaluator` implements `IEvaluator` correctly
- Interfaces enforce required methods

**Why It Matters**: These tests ensure pluggability - any new adapter/evaluator must follow the same contract.

### Manual Sprint 1 Testing

#### 1. Verify Database Tables

```bash
# Connect to database
docker exec -it ml_eval_postgres psql -U ml_user -d ml_eval_db

# List tables
\dt

# Expected output:
#  public | test_cases      | table | ml_user
#  public | model_runs      | table | ml_user
#  public | responses       | table | ml_user
#  public | evaluations     | table | ml_user
#  public | alembic_version | table | ml_user

# Describe test_cases table
\d test_cases

# Exit
\q
```
## HAD SOME issues
```
 curl -X 'POST' 'http://localhost:8000/api/v1/prompts/' \
  -H 'Content-Type: application/json' \
  -d '{
    "test_case_name": "Manual Test",
    "model_type": "nlp",
    "input_type": "text",
    "output_type": "classification",
    "input_data": {"text": "Hello"},
    "ground_truth": {"label": "greeting"}
  }'
Internal Server Error(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ a^C
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ alembic current
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
d7c66d5e9ce2 (head)
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker exec -it ml_eval_postgres psql -U ml_user -d ml_eval_db -c "\dt"
             List of relations
 Schema |      Name       | Type  |  Owner  
--------+-----------------+-------+---------
 public | alembic_version | table | ml_user
(1 row)

(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker exec -it ml_eval_postgres psql -U ml_user -d ml_eval_db -c "\dt"
           List of relations
 Schema |    Name     | Type  |  Owner  
--------+-------------+-------+---------
 public | evaluations | table | ml_user
 public | model_runs  | table | ml_user
 public | responses   | table | ml_user
 public | test_cases  | table | ml_user
             List of relations
 Schema |      Name       | Type  |  Owner  
--------+-----------------+-------+---------
 public | alembic_version | table | ml_user
(1 row)

Command 'List' not found, did you mean:
  command 'mist' from snap mist (master)
  command 'dist' from deb mmh (0.4-6)
  command 'dist' from deb nmh (1.8-1)
  command 'gist' from deb yorick (2.2.04+dfsg1-12)
  command 'hist' from deb loki (2.4.7.4-10)
See 'snap info <snapname>' for additional versions.
Schema: command not found
Type: command not found
Command 'Name' not found, did you mean:
  command 'mame' from snap mame (mame0285)
  command 'lame' from deb lame (3.100-6)
  command 'mame' from deb mame (0.261+dfsg.1-1)
See 'snap info <snapname>' for additional versions.
Owner: command not found
--------+-------------+-------+---------: command not found
evaluations: command not found
ml_user: command not found
Command 'table' not found, but can be installed with:
sudo snap install table
public: command not found
Command 'table' not found, but can be installed with:
sudo snap install table
ml_user: command not found
model_runs: command not found
public: command not found
public: command not found
ml_user: command not found
responses: command not found
Command 'table' not found, but can be installed with:
sudo snap install table
public: command not found
ml_user: command not found
Command 'table' not found, but can be installed with:
sudo snap install table
test_cases: command not found
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker exec -it ml_eval_postgres psql -U ml_user -d ml_eval_db -c "\dt"
             List of relations
 Schema |      Name       | Type  |  Owner  
--------+-----------------+-------+---------
 public | alembic_version | table | ml_user
(1 row)

(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ alembic downgrade base
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running downgrade d7c66d5e9ce2 -> 6b831d2c3e57, Add Sprint 3 schema changes
Traceback (most recent call last):
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1967, in _exec_single_context
    self.dialect.do_execute(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/default.py", line 952, in do_execute
    cursor.execute(statement, parameters)
psycopg2.errors.UndefinedObject: index "ix_evaluations_response_id" does not exist


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/bin/alembic", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/config.py", line 1047, in main
    CommandLine(prog=prog).main(argv=argv)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/config.py", line 1037, in main
    self.run_cmd(cfg, options)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/config.py", line 971, in run_cmd
    fn(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/command.py", line 530, in downgrade
    script.run_env()
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/script/base.py", line 545, in run_env
    util.load_python_file(self.dir, "env.py")
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/util/pyfiles.py", line 116, in load_python_file
    module = load_module_py(module_id, path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/util/pyfiles.py", line 136, in load_module_py
    spec.loader.exec_module(module)  # type: ignore
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/migrations/env.py", line 78, in <module>
    run_migrations_online()
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/migrations/env.py", line 73, in run_migrations_online
    context.run_migrations()
  File "<string>", line 8, in run_migrations
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/runtime/environment.py", line 969, in run_migrations
    self.get_context().run_migrations(**kw)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/runtime/migration.py", line 626, in run_migrations
    step.migration_fn(**kw)
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/migrations/versions/d7c66d5e9ce2_add_sprint_3_schema_changes.py", line 102, in downgrade
    op.drop_index(op.f('ix_evaluations_response_id'), table_name='evaluations')
  File "<string>", line 8, in drop_index
  File "<string>", line 3, in drop_index
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/operations/ops.py", line 1142, in drop_index
    return operations.invoke(op)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/operations/base.py", line 454, in invoke
    return fn(self, operation)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/operations/toimpl.py", line 121, in drop_index
    operations.impl.drop_index(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/ddl/impl.py", line 478, in drop_index
    self._exec(schema.DropIndex(index, **kw))
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/ddl/impl.py", line 256, in _exec
    return conn.execute(construct, params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1419, in execute
    return meth(
           ^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/sql/ddl.py", line 187, in _execute_on_connection
    return connection._execute_ddl(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1530, in _execute_ddl
    ret = self._execute_context(
          ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1846, in _execute_context
    return self._exec_single_context(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1986, in _exec_single_context
    self._handle_dbapi_exception(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 2363, in _handle_dbapi_exception
    raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1967, in _exec_single_context
    self.dialect.do_execute(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/default.py", line 952, in do_execute
    cursor.execute(statement, parameters)
sqlalchemy.exc.ProgrammingError: (psycopg2.errors.UndefinedObject) index "ix_evaluations_response_id" does not exist

[SQL: 
DROP INDEX ix_evaluations_response_id]
(Background on this error at: https://sqlalche.me/e/20/f405)
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ alembic upgrade head
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> 668bc8211f5e, Initial schema
INFO  [alembic.runtime.migration] Running upgrade 668bc8211f5e -> 6b831d2c3e57, Add unique constraint to responses table
Traceback (most recent call last):
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1967, in _exec_single_context
    self.dialect.do_execute(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/default.py", line 952, in do_execute
    cursor.execute(statement, parameters)
psycopg2.errors.DuplicateTable: relation "model_runs" already exists


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/bin/alembic", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/config.py", line 1047, in main
    CommandLine(prog=prog).main(argv=argv)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/config.py", line 1037, in main
    self.run_cmd(cfg, options)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/config.py", line 971, in run_cmd
    fn(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/command.py", line 483, in upgrade
    script.run_env()
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/script/base.py", line 545, in run_env
    util.load_python_file(self.dir, "env.py")
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/util/pyfiles.py", line 116, in load_python_file
    module = load_module_py(module_id, path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/util/pyfiles.py", line 136, in load_module_py
    spec.loader.exec_module(module)  # type: ignore
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/migrations/env.py", line 78, in <module>
    run_migrations_online()
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/migrations/env.py", line 73, in run_migrations_online
    context.run_migrations()
  File "<string>", line 8, in run_migrations
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/runtime/environment.py", line 969, in run_migrations
    self.get_context().run_migrations(**kw)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/runtime/migration.py", line 626, in run_migrations
    step.migration_fn(**kw)
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/migrations/versions/6b831d2c3e57_add_unique_constraint_to_responses_table.py", line 24, in upgrade
    op.create_table('model_runs',
  File "<string>", line 8, in create_table
  File "<string>", line 3, in create_table
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/operations/ops.py", line 1332, in create_table
    return operations.invoke(op)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/operations/base.py", line 454, in invoke
    return fn(self, operation)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/operations/toimpl.py", line 135, in create_table
    operations.impl.create_table(table, **kw)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/ddl/impl.py", line 437, in create_table
    self._exec(schema.CreateTable(table, **kw))
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/ddl/impl.py", line 256, in _exec
    return conn.execute(construct, params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1419, in execute
    return meth(
           ^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/sql/ddl.py", line 187, in _execute_on_connection
    return connection._execute_ddl(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1530, in _execute_ddl
    ret = self._execute_context(
          ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1846, in _execute_context
    return self._exec_single_context(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1986, in _exec_single_context
    self._handle_dbapi_exception(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 2363, in _handle_dbapi_exception
    raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1967, in _exec_single_context
    self.dialect.do_execute(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/default.py", line 952, in do_execute
    cursor.execute(statement, parameters)
sqlalchemy.exc.ProgrammingError: (psycopg2.errors.DuplicateTable) relation "model_runs" already exists

[SQL: 
CREATE TABLE model_runs (
	id SERIAL NOT NULL, 
	model_name VARCHAR NOT NULL, 
	model_version VARCHAR NOT NULL, 
	started_at TIMESTAMP WITHOUT TIME ZONE, 
	finished_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id)
)

]
(Background on this error at: https://sqlalche.me/e/20/f405)
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker exec ml_eval_postgres psql -U ml_user -d ml_eval_db -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public; GRANT 
  ALL ON SCHEMA public TO ml_user; GRANT ALL ON SCHEMA public TO public;"
NOTICE:  drop cascades to table alembic_version
DROP SCHEMA
CREATE SCHEMA
GRANT
GRANT
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ alembic upgrade head
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> 668bc8211f5e, Initial schema
INFO  [alembic.runtime.migration] Running upgrade 668bc8211f5e -> 6b831d2c3e57, Add unique constraint to responses table
Traceback (most recent call last):
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1967, in _exec_single_context
    self.dialect.do_execute(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/default.py", line 952, in do_execute
    cursor.execute(statement, parameters)
psycopg2.errors.DuplicateTable: relation "model_runs" already exists


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/bin/alembic", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/config.py", line 1047, in main
    CommandLine(prog=prog).main(argv=argv)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/config.py", line 1037, in main
    self.run_cmd(cfg, options)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/config.py", line 971, in run_cmd
    fn(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/command.py", line 483, in upgrade
    script.run_env()
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/script/base.py", line 545, in run_env
    util.load_python_file(self.dir, "env.py")
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/util/pyfiles.py", line 116, in load_python_file
    module = load_module_py(module_id, path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/util/pyfiles.py", line 136, in load_module_py
    spec.loader.exec_module(module)  # type: ignore
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/migrations/env.py", line 78, in <module>
    run_migrations_online()
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/migrations/env.py", line 73, in run_migrations_online
    context.run_migrations()
  File "<string>", line 8, in run_migrations
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/runtime/environment.py", line 969, in run_migrations
    self.get_context().run_migrations(**kw)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/runtime/migration.py", line 626, in run_migrations
    step.migration_fn(**kw)
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/migrations/versions/6b831d2c3e57_add_unique_constraint_to_responses_table.py", line 24, in upgrade
    op.create_table('model_runs',
  File "<string>", line 8, in create_table
  File "<string>", line 3, in create_table
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/operations/ops.py", line 1332, in create_table
    return operations.invoke(op)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/operations/base.py", line 454, in invoke
    return fn(self, operation)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/operations/toimpl.py", line 135, in create_table
    operations.impl.create_table(table, **kw)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/ddl/impl.py", line 437, in create_table
    self._exec(schema.CreateTable(table, **kw))
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/ddl/impl.py", line 256, in _exec
    return conn.execute(construct, params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1419, in execute
    return meth(
           ^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/sql/ddl.py", line 187, in _execute_on_connection
    return connection._execute_ddl(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1530, in _execute_ddl
    ret = self._execute_context(
          ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1846, in _execute_context
    return self._exec_single_context(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1986, in _exec_single_context
    self._handle_dbapi_exception(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 2363, in _handle_dbapi_exception
    raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1967, in _exec_single_context
    self.dialect.do_execute(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/default.py", line 952, in do_execute
    cursor.execute(statement, parameters)
sqlalchemy.exc.ProgrammingError: (psycopg2.errors.DuplicateTable) relation "model_runs" already exists

[SQL: 
CREATE TABLE model_runs (
	id SERIAL NOT NULL, 
	model_name VARCHAR NOT NULL, 
	model_version VARCHAR NOT NULL, 
	started_at TIMESTAMP WITHOUT TIME ZONE, 
	finished_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id)
)

]
(Background on this error at: https://sqlalche.me/e/20/f405)
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker ps
CONTAINER ID   IMAGE                COMMAND                  CREATED          STATUS          PORTS                                         NAMES
820590a54d06   postgres:15-alpine   "docker-entrypoint.s…"   26 minutes ago   Up 26 minutes   0.0.0.0:5433->5432/tcp, [::]:5433->5432/tcp   ml_eval_postgres
f0da256e8dc7   postgres:16          "docker-entrypoint.s…"   2 weeks ago      Up 27 hours     0.0.0.0:6432->5432/tcp, [::]:6432->5432/tcp   pythoncrud-db-1
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker compose down -v
WARN[0000] /home/dell-linux-dev3/Projects/ml-evaluation-framework/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
[+] down 3/3
 ✔ Container ml_eval_postgres                   Removed                                                                  0.3s
 ✔ Volume ml-evaluation-framework_postgres_data Removed                                                                  0.0s
 ✔ Network ml-evaluation-framework_default      Removed                                                                  0.2s
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker ps
CONTAINER ID   IMAGE         COMMAND                  CREATED       STATUS        PORTS                                         NAMES
f0da256e8dc7   postgres:16   "docker-entrypoint.s…"   2 weeks ago   Up 27 hours   0.0.0.0:6432->5432/tcp, [::]:6432->5432/tcp   pythoncrud-db-1
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker compose up -d
WARN[0000] /home/dell-linux-dev3/Projects/ml-evaluation-framework/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
[+] up 3/3
 ✔ Network ml-evaluation-framework_default      Created                                                                  0.0s
 ✔ Volume ml-evaluation-framework_postgres_data Created                                                                  0.0s
 ✔ Container ml_eval_postgres                   Created                                                                  0.1s
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker ps
CONTAINER ID   IMAGE                COMMAND                  CREATED          STATUS          PORTS                                         NAMES
1f27d5a7a0d2   postgres:15-alpine   "docker-entrypoint.s…"   13 seconds ago   Up 12 seconds   0.0.0.0:5433->5432/tcp, [::]:5433->5432/tcp   ml_eval_postgres
f0da256e8dc7   postgres:16          "docker-entrypoint.s…"   2 weeks ago      Up 27 hours     0.0.0.0:6432->5432/tcp, [::]:6432->5432/tcp   pythoncrud-db-1
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker ps | grep ml_eval_postgres
1f27d5a7a0d2   postgres:15-alpine   "docker-entrypoint.s…"   28 seconds ago   Up 28 seconds   0.0.0.0:5433->5432/tcp, [::]:5433->5432/tcp   ml_eval_postgres
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker ps
CONTAINER ID   IMAGE                COMMAND                  CREATED          STATUS          PORTS                                         NAMES
1f27d5a7a0d2   postgres:15-alpine   "docker-entrypoint.s…"   42 seconds ago   Up 42 seconds   0.0.0.0:5433->5432/tcp, [::]:5433->5432/tcp   ml_eval_postgres
f0da256e8dc7   postgres:16          "docker-entrypoint.s…"   2 weeks ago      Up 27 hours     0.0.0.0:6432->5432/tcp, [::]:6432->5432/tcp   pythoncrud-db-1
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ alembic upgrade head
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> 668bc8211f5e, Initial schema
INFO  [alembic.runtime.migration] Running upgrade 668bc8211f5e -> 6b831d2c3e57, Add unique constraint to responses table
Traceback (most recent call last):
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1967, in _exec_single_context
    self.dialect.do_execute(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/default.py", line 952, in do_execute
    cursor.execute(statement, parameters)
psycopg2.errors.DuplicateTable: relation "model_runs" already exists


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/bin/alembic", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/config.py", line 1047, in main
    CommandLine(prog=prog).main(argv=argv)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/config.py", line 1037, in main
    self.run_cmd(cfg, options)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/config.py", line 971, in run_cmd
    fn(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/command.py", line 483, in upgrade
    script.run_env()
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/script/base.py", line 545, in run_env
    util.load_python_file(self.dir, "env.py")
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/util/pyfiles.py", line 116, in load_python_file
    module = load_module_py(module_id, path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/util/pyfiles.py", line 136, in load_module_py
    spec.loader.exec_module(module)  # type: ignore
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/migrations/env.py", line 78, in <module>
    run_migrations_online()
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/migrations/env.py", line 73, in run_migrations_online
    context.run_migrations()
  File "<string>", line 8, in run_migrations
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/runtime/environment.py", line 969, in run_migrations
    self.get_context().run_migrations(**kw)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/runtime/migration.py", line 626, in run_migrations
    step.migration_fn(**kw)
  File "/home/dell-linux-dev3/Projects/ml-evaluation-framework/migrations/versions/6b831d2c3e57_add_unique_constraint_to_responses_table.py", line 24, in upgrade
    op.create_table('model_runs',
  File "<string>", line 8, in create_table
  File "<string>", line 3, in create_table
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/operations/ops.py", line 1332, in create_table
    return operations.invoke(op)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/operations/base.py", line 454, in invoke
    return fn(self, operation)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/operations/toimpl.py", line 135, in create_table
    operations.impl.create_table(table, **kw)
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/ddl/impl.py", line 437, in create_table
    self._exec(schema.CreateTable(table, **kw))
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/alembic/ddl/impl.py", line 256, in _exec
    return conn.execute(construct, params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1419, in execute
    return meth(
           ^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/sql/ddl.py", line 187, in _execute_on_connection
    return connection._execute_ddl(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1530, in _execute_ddl
    ret = self._execute_context(
          ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1846, in _execute_context
    return self._exec_single_context(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1986, in _exec_single_context
    self._handle_dbapi_exception(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 2363, in _handle_dbapi_exception
    raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 1967, in _exec_single_context
    self.dialect.do_execute(
  File "/home/dell-linux-dev3/anaconda3/envs/ml-eval-framework/lib/python3.11/site-packages/sqlalchemy/engine/default.py", line 952, in do_execute
    cursor.execute(statement, parameters)
sqlalchemy.exc.ProgrammingError: (psycopg2.errors.DuplicateTable) relation "model_runs" already exists

[SQL: 
CREATE TABLE model_runs (
	id SERIAL NOT NULL, 
	model_name VARCHAR NOT NULL, 
	model_version VARCHAR NOT NULL, 
	started_at TIMESTAMP WITHOUT TIME ZONE, 
	finished_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id)
)

]
(Background on this error at: https://sqlalche.me/e/20/f405)
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker ps
CONTAINER ID   IMAGE                COMMAND                  CREATED         STATUS         PORTS                                         NAMES
1f27d5a7a0d2   postgres:15-alpine   "docker-entrypoint.s…"   2 minutes ago   Up 2 minutes   0.0.0.0:5433->5432/tcp, [::]:5433->5432/tcp   ml_eval_postgres
f0da256e8dc7   postgres:16          "docker-entrypoint.s…"   2 weeks ago     Up 27 hours    0.0.0.0:6432->5432/tcp, [::]:6432->5432/tcp   pythoncrud-db-1
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$  docker exec ml_eval_postgres psql -U ml_user -d ml_eval_db -c "CREATE TABLE IF NOT EXISTS alembic_version (version_num 
  VARCHAR(32) NOT NULL, CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)); INSERT INTO alembic_version 
  (version_num) VALUES ('d7c66d5e9ce2');"
CREATE TABLE
INSERT 0 1
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker exec ml_eval_postgres psql -U ml_user -d ml_eval_db <<'EOF'
> ^C
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker exec ml_eval_postgres psql -U ml_user -d ml_eval_db <<'EOF'
  -- This is the content from d7c66d5e9ce2 migration
  CREATE TABLE model_runs (
      id SERIAL NOT NULL,
      model_name VARCHAR NOT NULL,
      model_version VARCHAR NOT NULL,
      model_type VARCHAR NOT NULL,
      model_endpoint VARCHAR,
      config JSONB,
      started_at TIMESTAMP,
      completed_at TIMESTAMP,
      status VARCHAR,
      total_cases INTEGER,
      completed_cases INTEGER,
      failed_cases INTEGER,
      PRIMARY KEY (id)
  );
  CREATE INDEX ix_model_runs_id ON model_runs (id);
  CREATE INDEX ix_model_runs_model_name ON model_runs (model_name);

  CREATE TABLE test_cases (
      id SERIAL NOT NULL,
      test_case_name VARCHAR NOT NULL,
  EOFATE INDEX ix_evaluations_response_id ON evaluations (response_id);me);
> ^C
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker exec ml_eval_postgres psql -U ml_user -d ml_eval_db <<'EOF'
  -- This is the content from d7c66d5e9ce2 migration
  CREATE TABLE model_runs (
      id SERIAL NOT NULL,
      model_name VARCHAR NOT NULL,
      model_version VARCHAR NOT NULL,
      model_type VARCHAR NOT NULL,
      model_endpoint VARCHAR,
      config JSONB,
      started_at TIMESTAMP,
      completed_at TIMESTAMP,
      status VARCHAR,
      total_cases INTEGER,
      completed_cases INTEGER,
      failed_cases INTEGER,
      PRIMARY KEY (id)
  );
  CREATE INDEX ix_model_runs_id ON model_runs (id);
  CREATE INDEX ix_model_runs_model_name ON model_runs (model_name);

  CREATE TABLE test_cases (
      id SERIAL NOT NULL,
      test_case_name VARCHAR NOT NULL,
  EOFATE INDEX ix_evaluations_response_id ON evaluations (response_id);me);
> 
> epf
> eof
> EOF
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker exec ml_eval_postgres psql -U ml_user -d ml_eval_db -c "\dt"
             List of relations
 Schema |      Name       | Type  |  Owner  
--------+-----------------+-------+---------
 public | alembic_version | table | ml_user
(1 row)

(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker exec ml_eval_postgres psql -U ml_user -d ml_eval_db -c "\dt"^C
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker exec -i ml_eval_postgres psql -U ml_user -d ml_eval_db < create_tables.sql
CREATE TABLE
CREATE INDEX
CREATE INDEX
CREATE TABLE
CREATE INDEX
CREATE INDEX
CREATE INDEX
CREATE INDEX
CREATE TABLE
CREATE INDEX
CREATE INDEX
CREATE INDEX
CREATE TABLE
CREATE INDEX
CREATE INDEX
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ docker exec ml_eval_postgres psql -U ml_user -d ml_eval_db -c "\dt"
             List of relations
 Schema |      Name       | Type  |  Owner  
--------+-----------------+-------+---------
 public | alembic_version | table | ml_user
 public | evaluations     | table | ml_user
 public | model_runs      | table | ml_user
 public | responses       | table | ml_user
 public | test_cases      | table | ml_user
(5 rows)

(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ alembic current
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
d7c66d5e9ce2 (head)
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'POST' 'http://localhost:8000/api/v1/prompts/' \
    -H 'Content-Type: application/json' \
    -d '{
      "test_case_name": "Manual Test",
      "model_type": "nlp",
      "input_type": "text",
      "output_type": "classification",
      "input_data": {"text": "Hello"},
      "ground_truth": {"label": "greeting"}
    }'
{"test_case_name":"Manual Test","model_type":"nlp","input_type":"text","output_type":"classification","input_data":{"text":"Hello"},"ground_truth":{"label":"greeting"},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":1,"created_at":"2026-02-05T20:48:17.414761","updated_at":"2026-02-05T20:48:17.(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ murl -X 'GET' 'http://localhost:8000/api/v1/prompts/1'ompts/1'
Command 'Curl' not found, did you mean:
  command 'yurl' from snap yurl (v0.6.3)
  command 'curl' from snap curl (8.18.0)
  command 'surl' from snap surl (0.8.0)
  command 'zurl' from deb zurl (1.12.0-1)
  command 'curl' from deb curl (8.5.0-2ubuntu10.6)
See 'snap info <snapname>' for additional versions.
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'GET' 'http://localhost:8000/api/v1/prompts/1'
{"test_case_name":"Manual Test","model_type":"nlp","input_type":"text","output_type":"classification","input_data":{"text":"Hello"},"ground_truth":{"label":"greeting"},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":1,"created_at":"2026-02-05T20:48:17.414761","updated_at":"2026-02-05T20:48:17.(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'PATCH' 'http://localhost:8000/api/v1/prompts/1' \pts/1' \
  -H 'Content-Type: application/json' \
  -d '{"test_case_name": "Updated Name"}'
{"test_case_name":"Updated Name","model_type":"nlp","input_type":"text","output_type":"classification","input_data":{"text":"Hello"},"ground_truth":{"label":"greeting"},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":1,"created_at":"2026-02-05T20:48:17.414761","updated_at":"2026-02-05T20:51:11.850406"}(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'PATCH' 'http://localhost:8000/api/v1/prompts/1'   -H 'Content-Type: application/json'   -d '{"test_case_name": "Updated Name"}'
{"test_case_name":"Updated Name","model_type":"nlp","input_type":"text","output_type":"classification","input_data":{"text":"Hello"},"ground_truth":{"label":"greeting"},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":1,"created_at":"2026-02-05T20:48:17.414761","updated_at":"2026-02-05T20:51:24(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'DELETE' 'http://localhost:8000/api/v1/prompts/1'rompts/1'
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'GET' 'http://localhost:8000/api/v1/prompts/1'
{"detail":"Prompt not found"}(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision
```

What Was Fixed

  1. Removed conflicting migrations: Deleted two old migrations that were creating duplicate tables with wrong names
  2. Updated the remaining migration: Changed d7c66d5e9ce2 to be the initial migration (down_revision = None)
  3. Database is clean: All tables created correctly with proper schema



#### 2. Test API Endpoints Manually

```bash
# Start API server (in separate terminal)
uvicorn ml_eval.main:app --reload --port 8000

# Create a test case
curl -X 'POST' 'http://localhost:8000/api/v1/prompts/' \
  -H 'Content-Type: application/json' \
  -d '{
    "test_case_name": "Manual Test",
    "model_type": "nlp",
    "input_type": "text",
    "output_type": "classification",
    "input_data": {"text": "Hello"},
    "ground_truth": {"label": "greeting"}
  }'

# Note the ID returned (e.g., 1)

# Retrieve the test case
curl -X 'GET' 'http://localhost:8000/api/v1/prompts/1'

# Update test case (PATCH)
curl -X 'PATCH' 'http://localhost:8000/api/v1/prompts/1' \
  -H 'Content-Type: application/json' \
  -d '{"test_case_name": "Updated Name"}'

# Delete test case
curl -X 'DELETE' 'http://localhost:8000/api/v1/prompts/1'

# Verify deletion (should return 404)
curl -X 'GET' 'http://localhost:8000/api/v1/prompts/1'
```
EXAMPLE OUTPUT:
```
 curl -X 'POST' 'http://localhost:8000/api/v1/prompts/' \
    -H 'Content-Type: application/json' \
    -d '{
      "test_case_name": "Manual Test",
      "model_type": "nlp",
      "input_type": "text",
      "output_type": "classification",
      "input_data": {"text": "Hello"},
      "ground_truth": {"label": "greeting"}
    }'
{"test_case_name":"Manual Test","model_type":"nlp","input_type":"text","output_type":"classification","input_data":{"text":"Hello"},"ground_truth":{"label":"greeting"},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":1,"created_at":"2026-02-05T20:48:17.414761","updated_at":"2026-02-05T20:48:17.(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ murl -X 'GET' 'http://localhost:8000/api/v1/prompts/1'ompts/1'
Command 'Curl' not found, did you mean:
  command 'yurl' from snap yurl (v0.6.3)
  command 'curl' from snap curl (8.18.0)
  command 'surl' from snap surl (0.8.0)
  command 'zurl' from deb zurl (1.12.0-1)
  command 'curl' from deb curl (8.5.0-2ubuntu10.6)
See 'snap info <snapname>' for additional versions.
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'GET' 'http://localhost:8000/api/v1/prompts/1'
{"test_case_name":"Manual Test","model_type":"nlp","input_type":"text","output_type":"classification","input_data":{"text":"Hello"},"ground_truth":{"label":"greeting"},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":1,"created_at":"2026-02-05T20:48:17.414761","updated_at":"2026-02-05T20:48:17.(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'PATCH' 'http://localhost:8000/api/v1/prompts/1' \pts/1' \
  -H 'Content-Type: application/json' \
  -d '{"test_case_name": "Updated Name"}'
{"test_case_name":"Updated Name","model_type":"nlp","input_type":"text","output_type":"classification","input_data":{"text":"Hello"},"ground_truth":{"label":"greeting"},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":1,"created_at":"2026-02-05T20:48:17.414761","updated_at":"2026-02-05T20:51:11.850406"}(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'PATCH' 'http://localhost:8000/api/v1/prompts/1'   -H 'Content-Type: application/json'   -d '{"test_case_name": "Updated Name"}'
{"test_case_name":"Updated Name","model_type":"nlp","input_type":"text","output_type":"classification","input_data":{"text":"Hello"},"ground_truth":{"label":"greeting"},"category":null,"tags":null,"difficulty":null,"origin":"human","is_verified":true,"test_case_metadata":{},"created_by":null,"id":1,"created_at":"2026-02-05T20:48:17.414761","updated_at":"2026-02-05T20:51:24(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'DELETE' 'http://localhost:8000/api/v1/prompts/1'rompts/1'
(ml-eval-framework) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/ml-evaluation-framework$ curl -X 'GET' 'http://localhost:8000/api/v1/prompts/1'

```


### Sprint 1 Success Criteria

✅ **PASS**: All database and API tests pass
✅ **PASS**: Manual CRUD operations work via curl
✅ **PASS**: JSONB columns store/retrieve complex data
✅ **PASS**: Foreign keys and cascades work
✅ **PASS**: User-first defaults are applied

---

## Sprint 2 Testing

**Sprint Goal**: Verify test suite loading, validation, and management

### Current Status

🟡 **Sprint 2 is ~30% complete** - Most tests are **not yet implemented**

### What Needs Testing

Sprint 2 tests will ensure:
1. ❌ Test suite files (JSON/YAML) can be parsed
2. ❌ TestSuiteManager class works correctly
3. ❌ Input validators catch invalid data
4. ❌ Output validators catch invalid formats
5. ❌ Suite versioning and metadata work
6. ❌ Duplicate detection prevents redundant data
7. ❌ Database integration stores loaded suites
8. ❌ Error reporting is comprehensive

### Implemented Tests

#### Manual Test: Load Suite Parsing

```bash
# Test basic JSON parsing (no database integration yet)
python scripts/load_suite.py data/example_suite.json

# Expected output:
# ✅ Successfully parsed 3 test cases from 'data/example_suite.json'.
# --- (Next step: Implement database interaction to add these test cases) ---
```

**What This Tests**:
- JSON file parsing
- File existence validation
- Top-level array validation

**What's Missing**:
- YAML support
- Database insertion
- Validation logic
- Error reporting

### Tests to Implement (Sprint 2 Future Work)

Location: `tests/test_suite/` (directory exists but empty)

#### 1. TestSuiteManager Tests

```bash
# Will be: pytest tests/test_suite/test_manager.py -v
```

**Future Test Cases**:
- `test_load_suite_from_json` - Parse JSON file
- `test_load_suite_from_yaml` - Parse YAML file
- `test_load_suite_to_database` - Insert into DB
- `test_detect_duplicates` - Prevent redundant data
- `test_suite_versioning` - Handle versions correctly
- `test_invalid_file_format` - Reject bad files

#### 2. Input Validator Tests

```bash
# Will be: pytest tests/test_suite/test_input_validators.py -v
```

**Future Test Cases**:
- `test_image_path_validator_valid` - Accept valid image paths
- `test_image_path_validator_invalid` - Reject non-existent paths
- `test_image_path_validator_extensions` - Validate file types
- `test_tabular_validator` - Check schema compliance
- `test_audio_path_validator` - Validate audio files
- `test_time_series_validator` - Check sequence data

#### 3. Output Validator Tests

```bash
# Will be: pytest tests/test_suite/test_output_validators.py -v
```

**Future Test Cases**:
- `test_classification_validator` - Check label format
- `test_bounding_box_validator` - Validate coordinates
- `test_regression_validator` - Check numerical values
- `test_text_output_validator` - Validate text format

### Manual Sprint 2 Testing (When Implemented)

```bash
# 1. Create test suite file
cat > /tmp/test_suite.json <<EOF
[
  {
    "test_case_name": "Test 1",
    "model_type": "nlp",
    "input_type": "text",
    "output_type": "classification",
    "input_data": {"text": "Sample"},
    "ground_truth": {"label": "sample"}
  }
]
EOF

# 2. Load suite into database
python scripts/load_suite.py /tmp/test_suite.json

# 3. Verify in database
docker exec -it ml_eval_postgres psql -U ml_user -d ml_eval_db \
  -c "SELECT test_case_name, model_type FROM test_cases WHERE test_case_name = 'Test 1';"
```

### Sprint 2 Success Criteria (Future)

❌ **TODO**: JSON and YAML parsing tests pass
❌ **TODO**: Validators reject invalid data
❌ **TODO**: Test suite loads into database
❌ **TODO**: Duplicate detection works
❌ **TODO**: Error reporting is comprehensive

---

## Sprint 3 Testing

**Sprint Goal**: Verify model query engine and end-to-end evaluation workflow

### What We're Testing

Sprint 3 tests ensure:
1. ✅ Model adapters work for different model types
2. ✅ Evaluators correctly assess outputs
3. ✅ EvaluationEngine orchestrates full workflow
4. ✅ Responses are stored with correct data
5. ✅ Evaluations link to responses correctly
6. ✅ ModelRun status is tracked accurately
7. ✅ Error handling works for failed evaluations

### Evaluation Engine Tests

Location: `tests/test_query_engine/test_engine.py`

```bash
pytest tests/test_query_engine/test_engine.py -v
```

#### Test Case 1: `test_run_evaluation_success`

**What It Tests**:
- Complete end-to-end evaluation workflow
- Multiple test cases processed successfully
- All database records created correctly

**The Workflow**:
1. Creates a ModelRun for "matrix_multiplication"
2. Creates 2 test cases with matrix data
3. Instantiates MatrixModel + LocalMatrixAdapter
4. Runs EvaluationEngine.run_evaluation()
5. Verifies:
   - ModelRun status → "completed"
   - total_cases = 2, completed_cases = 2, failed_cases = 0
   - 2 Response records created
   - 2 Evaluation records created
   - All evaluations pass (score = 1.0)

**Why It Matters**: This is the **core workflow** of the entire platform. If this test passes, Sprint 1-3 integration is working.

**Running the Test**:
```bash
pytest tests/test_query_engine/test_engine.py::test_run_evaluation_success -v -s
```

#### Test Case 2: `test_run_evaluation_with_failure`

**What It Tests**:
- Error handling when model fails
- Partial completion tracking
- Error messages stored correctly

**The Workflow**:
1. Creates ModelRun with 2 test cases
2. One test case has valid matrix dimensions
3. One test case has invalid dimensions (will fail)
4. Runs evaluation
5. Verifies:
   - ModelRun still completes (status = "completed")
   - completed_cases = 1, failed_cases = 1
   - Successful case has Response + Evaluation
   - Failed case has Response with error_message
   - Failed case has NO Evaluation

**Why It Matters**: Real models fail sometimes. We must handle errors gracefully and continue processing remaining test cases.

**Running the Test**:
```bash
pytest tests/test_query_engine/test_engine.py::test_run_evaluation_with_failure -v -s
```

### Manual Sprint 3 Testing

#### 1. Simple Matrix Model Test

```bash
# Terminal 1: Start API
uvicorn ml_eval.main:app --reload --port 8000

# Terminal 2: Create test case and run
# Step 1: Create model run
curl -X 'POST' 'http://localhost:8000/api/v1/runs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "MatrixModel-Test",
    "model_version": "1.0",
    "model_type": "matrix_multiplication"
  }'
# Note the "id" (e.g., 1)

# Step 2: Create test case
curl -X 'POST' 'http://localhost:8000/api/v1/prompts/' \
  -H 'Content-Type: application/json' \
  -d '{
    "test_case_name": "2x2 Identity Test",
    "model_type": "matrix_multiplication",
    "input_type": "json",
    "output_type": "json",
    "input_data": {
      "matrix_a": [[1, 0], [0, 1]],
      "matrix_b": [[5, 6], [7, 8]]
    },
    "ground_truth": {"result_matrix": [[5, 6], [7, 8]]}
  }'

# Step 3: Run evaluation
python scripts/run_evaluation.py 1

# Expected output:
# --- Setting up evaluation for ModelRun ID: 1 ---
# --- Initializing components ---
# ✅ Components initialized for model_type: matrix_multiplication.
# --- Running evaluation for ModelRun ID: 1 ---
# Starting evaluation for ModelRun 1 (MatrixModel-Test 1.0)...
# Evaluation for ModelRun 1 completed.
# 🎉 Evaluation complete for ModelRun ID: 1
#    - Total Cases: 1
#    - Completed: 1
#    - Failed: 0
```

#### 2. Time Series Model Test

```bash
# Step 1: Train model (if not already done)
python scripts/train_linear_time_series.py

# Step 2: Seed test cases
python scripts/seed_linear_test_cases.py

# Step 3: Create model run
curl -X 'POST' 'http://localhost:8000/api/v1/runs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "linear_model",
    "model_version": "1.0",
    "model_type": "time_series_linear"
  }'
# Note the run_id (e.g., 2)

# Step 4: Run evaluation
python scripts/run_evaluation.py 2

# Step 5: Generate report
python scripts/generate_report_time_series_v3.py 2

# Step 6: View report
ls -lh reports/
# Check for generated PNG and CSV files
```

#### 3. Verify Results in Database

```bash
docker exec -it ml_eval_postgres psql -U ml_user -d ml_eval_db

-- Check model run status
SELECT id, model_name, model_type, status, total_cases, completed_cases, failed_cases
FROM model_runs
ORDER BY id DESC
LIMIT 5;

-- Check responses
SELECT r.id, r.test_case_id, r.run_id, r.error_message,
       tc.test_case_name
FROM responses r
JOIN test_cases tc ON r.test_case_id = tc.id
WHERE r.run_id = 1;

-- Check evaluations
SELECT e.id, e.evaluator_type, e.score, e.passed, e.metric_name,
       r.test_case_id
FROM evaluations e
JOIN responses r ON e.response_id = r.id
WHERE r.run_id = 1;

\q
```

### Sprint 3 Success Criteria

✅ **PASS**: EvaluationEngine tests pass
✅ **PASS**: Manual evaluation completes successfully
✅ **PASS**: Responses stored with correct output_data
✅ **PASS**: Evaluations link to responses
✅ **PASS**: Error handling works for failed cases
✅ **PASS**: Multiple model types supported

---

## Integration Testing

### Full End-to-End Test

This test verifies all sprints working together:

```bash
# 1. Clean database
docker exec -it ml_eval_postgres psql -U ml_user -d ml_eval_db -c "TRUNCATE test_cases, model_runs, responses, evaluations CASCADE;"

# 2. Create test case via API
curl -X 'POST' 'http://localhost:8000/api/v1/prompts/' \
  -H 'Content-Type: application/json' \
  -d '{
    "test_case_name": "Integration Test",
    "model_type": "simple_match",
    "input_type": "text",
    "output_type": "json",
    "input_data": {"text": "test input"},
    "ground_truth": {"text": "test input", "processed": true}
  }'

# 3. Create model run
curl -X 'POST' 'http://localhost:8000/api/v1/runs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "SimpleModel",
    "model_version": "1.0",
    "model_type": "simple_match"
  }'

# 4. Run evaluation
python scripts/run_evaluation.py 1

# 5. Verify in database
docker exec -it ml_eval_postgres psql -U ml_user -d ml_eval_db \
  -c "SELECT 'test_cases:', COUNT(*) FROM test_cases UNION ALL SELECT 'model_runs:', COUNT(*) FROM model_runs UNION ALL SELECT 'responses:', COUNT(*) FROM responses UNION ALL SELECT 'evaluations:', COUNT(*) FROM evaluations;"

# Expected output:
#   test_cases: 1
#   model_runs: 1
#   responses: 1
#   evaluations: 1
```

---

## Manual Testing Procedures

### Daily Development Testing

```bash
# 1. Run unit tests
pytest -v

# 2. Start API server
uvicorn ml_eval.main:app --reload --port 8000

# 3. Test API health
curl http://localhost:8000/health

# 4. Create/test specific feature
# (use curl commands above)
```

### Before Committing Code

```bash
# 1. Run all tests
pytest

# 2. Check test coverage
pytest --cov=ml_eval --cov-report=term-missing

# 3. Run linter (if configured)
# flake8 ml_eval/ tests/

# 4. Verify migrations
alembic upgrade head
alembic downgrade -1
alembic upgrade head
```

---

## Troubleshooting

### Test Database Issues

**Problem**: Tests fail with "database does not exist"

```bash
# Solution: Create test database
PGPASSWORD=ml_password createdb -h localhost -p 5433 -U ml_user ml_eval_db_test
```

**Problem**: Tests hang or timeout

```bash
# Solution: Check database is running
docker ps | grep ml_eval_postgres

# Restart if needed
docker-compose restart db
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'ml_eval'`

```bash
# Solution: Install in development mode
pip install -e .

# Or ensure project root is in PYTHONPATH
export PYTHONPATH="/home/dell-linux-dev3/Projects/ml-evaluation-framework:$PYTHONPATH"
```

### Migration Issues

**Problem**: Alembic migrations fail

```bash
# Check current revision
alembic current

# Reset migrations (CAUTION: deletes data)
alembic downgrade base
alembic upgrade head
```

### Test Failures After Schema Changes

**Problem**: Tests fail after updating models

```bash
# 1. Generate new migration
alembic revision --autogenerate -m "Description of changes"

# 2. Apply migration
alembic upgrade head

# 3. Re-run tests
pytest
```

---

## Summary

### Current Test Coverage

| Sprint | Automated Tests | Manual Tests | Status |
|--------|----------------|--------------|---------|
| Sprint 1 | 15 tests | ✅ Complete | ✅ 100% |
| Sprint 2 | 0 tests | 🟡 Partial | 🟡 30% |
| Sprint 3 | 7 tests | ✅ Complete | ✅ 100% |

### Test Commands Quick Reference

```bash
# All tests
pytest

# Specific sprint
pytest tests/test_database/        # Sprint 1
pytest tests/test_query_engine/    # Sprint 3

# With coverage
pytest --cov=ml_eval

# Verbose output
pytest -v -s
```

---

**Last Updated**: 2026-02-05
**Next Update**: After Sprint 2 test implementation