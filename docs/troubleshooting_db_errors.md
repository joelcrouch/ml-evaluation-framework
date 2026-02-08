# Troubleshooting Guide

This document provides solutions to common issues you may encounter while developing and testing the ML Evaluation Framework.

---

## Database Issues

### Issue: `database "ml_eval_db_test" does not exist`

**Symptoms:**
```
psycopg2.OperationalError: connection to server at "localhost" (127.0.0.1), port 5433 failed:
FATAL:  database "ml_eval_db_test" does not exist
```

**Root Cause:**
The test database hasn't been created in the PostgreSQL Docker container.

**Solution:**

1. **Verify PostgreSQL container is running:**
   ```bash
   docker ps | grep postgres
   ```

   You should see a container named `ml_eval_postgres` running on port 5433.

2. **Check if test database exists:**
   ```bash
   docker exec ml_eval_postgres psql -U ml_user -d postgres -c "\l" | grep ml_eval
   ```

   You should see both `ml_eval_db` and `ml_eval_db_test`.

3. **Create test database if missing:**
   ```bash
   docker exec ml_eval_postgres psql -U ml_user -d postgres -c "CREATE DATABASE ml_eval_db_test;"
   ```

   **Note:** If you get "ERROR: database already exists", that's fine - it means it's already created.

4. **Verify connection:**
   ```bash
   docker exec ml_eval_postgres psql -U ml_user -d ml_eval_db_test -c "SELECT version();"
   ```

---

### Issue: Test database is out of sync with schema

**Symptoms:**
- Tests fail with `relation does not exist` errors
- Tests fail with column mismatch errors
- Migrations applied to dev database but not test database

**Solution:**

**Option 1: Drop and recreate test database (recommended for clean slate):**
```bash
# Drop test database
docker exec ml_eval_postgres psql -U ml_user -d postgres -c "DROP DATABASE IF EXISTS ml_eval_db_test;"

# Recreate test database
docker exec ml_eval_postgres psql -U ml_user -d postgres -c "CREATE DATABASE ml_eval_db_test;"

# Run tests (they will create tables automatically)
pytest -v
```

**Option 2: Apply migrations to test database:**
```bash
# Set environment to test database
export POSTGRES_DB=ml_eval_db_test

# Run migrations
alembic upgrade head

# Unset environment variable
unset POSTGRES_DB
```

---

### Issue: Cannot connect to PostgreSQL container

**Symptoms:**
```
connection to server at "localhost" (127.0.0.1), port 5433 failed:
Connection refused
```

**Solution:**

1. **Check if container is running:**
   ```bash
   docker ps | grep ml_eval_postgres
   ```

2. **If not running, start it:**
   ```bash
   docker start ml_eval_postgres
   ```

3. **If container doesn't exist, check docker-compose:**
   ```bash
   docker-compose ps
   docker-compose up -d
   ```

4. **Verify port mapping:**
   ```bash
   docker port ml_eval_postgres
   ```

   Should show: `5432/tcp -> 0.0.0.0:5433`

---

### Issue: Permission denied errors in PostgreSQL

**Symptoms:**
```
psycopg2.OperationalError: FATAL:  password authentication failed for user "ml_user"
```

**Solution:**

1. **Verify credentials in environment variables:**
   ```bash
   echo $POSTGRES_USER
   echo $POSTGRES_PASSWORD
   ```

   Should be `ml_user` and `ml_password`.

2. **Check if credentials match docker-compose.yml:**
   ```bash
   grep -A 5 "postgres:" docker-compose.yml
   ```

3. **Reset PostgreSQL user password (if needed):**
   ```bash
   docker exec ml_eval_postgres psql -U postgres -c "ALTER USER ml_user WITH PASSWORD 'ml_password';"
   ```

---

## Test Issues

### Issue: Tests fail with "TestSuiteManager cannot be collected" warning

**Symptoms:**
```
PytestCollectionWarning: cannot collect test class 'TestSuiteManager' because it has a __init__ constructor
```

**Impact:**
This is a **warning only** - it doesn't affect test execution. Pytest mistakenly thinks the `TestSuiteManager` class in `ml_eval/test_suite/manager.py` is a test class because it starts with "Test".

**Solution (optional):**
Rename the class to avoid the "Test" prefix (e.g., `SuiteManager`), but this is **not necessary** for functionality.

---

### Issue: Image path validation tests fail

**Symptoms:**
```
FileNotFoundError: Image file not found: data/seeded_images/sample_cat.jpg
```

**Solution:**

1. **Verify seeded images exist:**
   ```bash
   ls -la data/seeded_images/
   ```

2. **Use existing images in tests:**
   Update test fixtures to reference actual images:
   ```python
   "input_data": {"path": "data/seeded_images/roses/roses_1.jpg"}
   ```

3. **Regenerate seeded images if missing:**
   ```bash
   python scripts/seed_sample_data.py  # If this script exists
   ```

---

## Running Tests

### Quick Test Commands

```bash
# Run all tests
pytest -v

# Run specific test suite
pytest tests/test_suite/ -v

# Run with coverage
pytest --cov=ml_eval --cov-report=html

# Run specific test file
pytest tests/test_suite/test_manager.py -v

# Run specific test
pytest tests/test_suite/test_manager.py::test_load_json_file -v

# Stop on first failure
pytest -x

# Show print statements
pytest -s
```

---

## Environment Setup Issues

### Issue: Missing Python dependencies

**Symptoms:**
```
ModuleNotFoundError: No module named 'pyyaml'
```

**Solution:**

1. **Activate conda environment:**
   ```bash
   conda activate ml-eval-framework
   ```

2. **Install missing dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   pip list | grep yaml
   ```

---

### Issue: Wrong Python version or environment

**Symptoms:**
- Import errors for installed packages
- Tests run but use wrong database

**Solution:**

1. **Verify active environment:**
   ```bash
   which python
   conda env list
   ```

   Should show `ml-eval-framework` environment active.

2. **Deactivate and reactivate:**
   ```bash
   conda deactivate
   conda activate ml-eval-framework
   ```

3. **Verify Python version:**
   ```bash
   python --version  # Should be 3.11.x
   ```

---

## Docker Issues

### Issue: Docker daemon not running

**Symptoms:**
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

**Solution:**
```bash
# Start Docker service
sudo systemctl start docker

# Or restart Docker
sudo systemctl restart docker

# Enable Docker on boot (optional)
sudo systemctl enable docker
```

---

### Issue: Port 5433 already in use

**Symptoms:**
```
Error starting userland proxy: listen tcp4 0.0.0.0:5433: bind: address already in use
```

**Solution:**

1. **Find process using port 5433:**
   ```bash
   sudo lsof -i :5433
   ```

2. **Stop conflicting process or change port:**
   ```bash
   # Option 1: Kill the process
   sudo kill -9 <PID>

   # Option 2: Change docker-compose.yml port mapping
   # Change "5433:5432" to "5434:5432"
   ```

---

## Sprint 2 Specific Issues

### Issue: ValidationReport errors not showing

**Symptoms:**
- Tests pass but validation errors aren't displayed
- No feedback on what's wrong with test suite

**Solution:**

Use the `render()` method to display validation results:
```python
manager = TestSuiteManager()
report = manager.validate_suite(test_cases)
print(report.render())
```

---

### Issue: Duplicate detection not working

**Symptoms:**
- Same test case saved multiple times
- `skip_duplicates=True` not preventing duplicates

**Solution:**

1. **Check if test cases have unique names:**
   ```python
   # Duplicates detected by: test_case_name + model_type OR input_data hash
   ```

2. **Clear existing test data:**
   ```bash
   # Drop and recreate test database
   docker exec ml_eval_postgres psql -U ml_user -d postgres -c "DROP DATABASE ml_eval_db_test;"
   docker exec ml_eval_postgres psql -U ml_user -d postgres -c "CREATE DATABASE ml_eval_db_test;"
   ```

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check logs:**
   ```bash
   # Docker logs
   docker logs ml_eval_postgres

   # Application logs (if configured)
   tail -f logs/app.log
   ```

2. **Run tests with verbose output:**
   ```bash
   pytest -vv --tb=long
   ```

3. **Check database state:**
   ```bash
   docker exec -it ml_eval_postgres psql -U ml_user -d ml_eval_db_test

   # Inside psql:
   \dt                    # List tables
   \d+ prompts            # Describe prompts table
   SELECT COUNT(*) FROM prompts;  # Check data
   ```

4. **Verify environment configuration:**
   ```bash
   env | grep POSTGRES
   cat .env
   ```

---

## Preventive Measures

### Before running tests:

1. ✅ Verify Docker container is running
2. ✅ Check test database exists
3. ✅ Activate correct conda environment
4. ✅ Ensure no stale data in test database

### After making schema changes:

1. ✅ Drop and recreate test database
2. ✅ Run full test suite
3. ✅ Check for migration scripts needed

### Before committing code:

1. ✅ Run full test suite: `pytest -v`
2. ✅ Check test coverage: `pytest --cov=ml_eval`
3. ✅ Verify no warnings or errors

---

**Last Updated:** 2026-02-06
**Maintainer:** ML Eval Framework Team





