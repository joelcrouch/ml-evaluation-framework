#fastapin entyr point
from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# 1. Database Imports
# We use this to connect to the DB and create tables (only for the first time/testing)
from ml_eval.database.connection import engine
from ml_eval.database.models import Base
from ml_eval.database.connection import check_database_connection
# 2. Router/API Imports (Will be added in T2.2)
from ml_eval.routers import crud

# --- Application Lifespan Context ---

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Handles application startup and shutdown events.
    
    Startup: Ensures tables exist (Alembic handles migrations, but this is a failsafe/test quick start).
    """
    print("--- [STARTUP] Initializing database connection... ---")
    
    # Optional: Creates all tables defined in Base.metadata if they don't exist.
    # We rely on Alembic for production migrations, but this is helpful for local development.
    # Base.metadata.create_all(bind=engine) 
    # NEW: Actively check the connection using a non-destructive query
    check_database_connection()

    yield
    
    # Shutdown logic can go here later (e.g., closing resource pools)
    print("--- [SHUTDOWN] Application shutting down. ---")

# --- FastAPI App Initialization ---

from ml_eval.routers import crud

app = FastAPI(
    title="Universal ML Evaluation Framework API",
    description="The core API for managing test suites, model runs, and evaluations.",
    version="0.1.0",
    lifespan=lifespan
)

# --- Router Inclusion (Future Task T2.2) ---
app.include_router(crud.router, prefix="/api/v1", tags=["CRUD"])


# --- Root Endpoint (Health Check) ---

@app.get("/")
def read_root():
    """Confirms the API is running and healthy."""
    return {
        "message": "Welcome to the Universal ML Evaluation Framework API!",
        "status": "Running",
        "api_version": app.version
    }

# --- IMPORTANT: Database Session Dependency (Next Task T2.1.3) ---

# This function will be defined in the next step and used by all router endpoints
# to manage the database connection for each request.
# 
# from ml_eval.database.connection import SessionLocal
# def get_db():
#     # ... implementation goes here ...
#     pass