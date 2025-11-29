import sys
import os

# Adjust Python Path to find your 'ml_eval' package from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the necessary components
try:
    from ml_eval.database.connection import engine
    # Importing Base imports all defined models (TestPrompt, ModelRun, etc.)
    from ml_eval.database.models import Base 
    print("Imports successful.")
except ImportError as e:
    print(f"FATAL ERROR: Failed to import database modules. Check paths and dependencies.")
    print(f"Error: {e}")
    sys.exit(1)


def create_all_tables():
    """
    Connects to the database and creates all tables defined in Base's metadata.
    This effectively applies the schema from ml_eval/database/models.py.
    """
    print("--- Starting database table creation ---")
    
    # ⚠️ WARNING: This command is idempotent but destructive in that it wipes and recreates 
    # tables if they already exist without managing data. This is only for initial setup.
    # In production, you would use Alembic migrations.
    try:
        # Check connection before proceeding
        with engine.connect() as connection:
            print(f"Successfully connected to database: {connection.engine.url}")
            
        # Create all tables defined in Base
        Base.metadata.create_all(bind=engine)
        print("🎉 SUCCESS: All tables created successfully!")
        
    except Exception as e:
        print(f"🛑 ERROR: Failed to create database tables.")
        print(f"Details: {e}")
        sys.exit(1)


if __name__ == "__main__":
    create_all_tables()