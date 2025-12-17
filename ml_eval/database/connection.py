# ml_eval/database/connection.py

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError
# from .connection import engine
# Load environment variables from .env file
# This is crucial for securely loading the SQLALCHEMY_DATABASE_URL
load_dotenv()

# 1. Configuration
# Get the full database URL from the environment variable
# It is defined in the .env file as: postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DATABASE
SQLALCHEMY_DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL")

if not SQLALCHEMY_DATABASE_URL:
    # Raise an error if the connection string is missing, which is a critical failure
    raise ValueError(
        "SQLALCHEMY_DATABASE_URL not found. "
        "Please ensure your .env file is present and correctly configured."
    )

# 2. Database Engine Creation
# The engine is the central object that SQLAlchemy uses to communicate with the database.
# 'pool_pre_ping=True' ensures connections are valid before use.
# 'json_serializer' and 'json_deserializer' are used to handle Python dicts
# with the PostgreSQL JSONB columns.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    # In a real application, you may adjust pool size for production scalability.
    # For local development, default settings are fine.
)

# 3. Session Management
# SessionLocal is a class that represents a session factory. 
# Instances of SessionLocal will be the actual database sessions.
SessionLocal = sessionmaker(
    autocommit=False,  # We will manually commit transactions
    autoflush=False,   # No autoflushing of changes
    bind=engine        # Bind the session to the created engine
)

# 4. Base Class for ORM Models
# The declarative_base class is the base for all your ORM models.
# All your models in database/models.py must inherit from this Base.
Base = declarative_base()


# 5. Dependency for Session Handling
# This function is used as a session dependency in CRUD operations or FastAPI (later)
# It ensures that a session is created and then properly closed, regardless of success or failure.
def get_db():
    """Provides a database session to the caller and ensures it is closed afterward."""
    db = SessionLocal()
    try:
        # Yield the session object for use in CRUD functions
        yield db
    finally:
        # Ensure the session is closed in all cases
        db.close()

# Note: The Base object will be imported into `ml_eval/database/models.py` 
# for defining the TestCase, ModelRun, Response, and Evaluation tables.

def check_database_connection():
    """Performs a non-destructive check to ensure the database is reachable and accepting connections."""
    try:
        # Tries to connect and execute a simple query (SELECT 1 is a universal standard)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("✅ Database connection verified.")
    except OperationalError as e:
        # Re-raise a friendlier error to crash the application if the database is unreachable
        print("❌ FATAL ERROR: Database connection failed. Is PostgreSQL running and URL correct?")
        raise ConnectionError(f"Database unavailable: {e}")

from . import models