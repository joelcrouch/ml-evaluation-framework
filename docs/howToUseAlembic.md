### Why
So in this project, we initially create the db with setup_db.py.  That script maps teh cruurent ORM to the correct tables in our database.  
Wowee! Thats great!
But the terrbile thing about it is everytime we want to add in a new column/table to our database it wipes our database clean.  Not exactly what we signed up for.
So, off the shelf comes alembic.  
We can use alembic to update databaes in place. It will/can add a column/table to an existing database without having to create a bradn new one.  We could do that programmatically with SQL,but why not use a tested tool that we can plug and play?  We say yes.

Anyways.

### How To
1.  Initialize alembic
```
alembic init migrations
```
The command above (run in project root) will create a alembic.ini and a migrations directory.  

2.  Configure migrations/env.py
This script will find our python code(ml_eval), load creds from .env (project root), and read the sqlalchemy models. (base)
Replace whatevere is in migrations/env.py with 
```
# migrations/env.py
import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
from dotenv import load_dotenv

# ------------------------------------------------------------------------
# 1. Add your project root to the python path so imports work
# ------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ------------------------------------------------------------------------
# 2. Load Environment Variables & Import Models
# ------------------------------------------------------------------------
load_dotenv()  # Load .env file

# Import your Base and models so Alembic can detect changes
from ml_eval.database.models import Base
# (Ensure your models are imported in models.py or imported here so they register)

config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ------------------------------------------------------------------------
# 3. Set the Database URL dynamically from .env
# ------------------------------------------------------------------------
# This overrides the sqlalchemy.url in alembic.ini with your actual credentials
db_url = os.getenv("SQLALCHEMY_DATABASE_URL")
if not db_url:
    raise ValueError("SQLALCHEMY_DATABASE_URL not found in .env")
config.set_main_option("sqlalchemy.url", db_url)

# ------------------------------------------------------------------------
# 4. Set Target Metadata
# ------------------------------------------------------------------------
# This tells Alembic where your "Target" schema is (your Python models)
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

```


3.  Generate initial mgration
We already ran setup_db.py so we and with our screenshot on our living doc.md file, we can see the files were created in the database.  Cool.
Alembic doesn't know anything about that yet. So we do some more stuff

```
alembic revision --autogenerate -m "Initial schema'
```
Veify by looking at the output, and you should see it makeing test_prompts, model_runs etc.

Then you have to sync alembic, and this is crucial.  We have two options (blow up or add )

I think we should blow it up to make sure alembci works.  
 
- Option A (Clean Slate - Recommended for Dev): Drop the tables and let Alembic recreate them. This verifies your migration script works 100%.
```
# Enter postgres shell (or use a GUI tool)
docker exec -it ml_eval_postgres psql -U ml_user -d ml_eval_db
# Inside SQL shell:
DROP TABLE evaluations; DROP TABLE responses; DROP TABLE model_runs; DROP TABLE test_prompts;
\q

# Now apply the migration
alembic upgrade head

```

- Option B (fakeit)
```
alembic stamp head
```

#### how to use going forward

1. Modify ml-eval/database/models.py (eg add a new column)

2. generate a migration script
```
alembic revision --autogenerate -m "added cost column"
```

3.  REview the generated file in migrations/versions/ (sanity check)

4.  Apply the change

```
alembic upgrade head

```