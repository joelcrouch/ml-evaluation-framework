import sys
import os
import contextlib
from sqlalchemy.sql import text 
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError, ProgrammingError

# --- CRITICAL: Adjust Python Path to find your 'ml_eval' package ---
# This assumes you are running the script from the project root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the objects to test
try:
    # Adjust this path if your files are not under a directory named 'ml_eval'
    from ml_eval.database.connection import engine, SessionLocal, get_db, SQLALCHEMY_DATABASE_URL
except ImportError:
    print("FATAL ERROR: Could not import connection.py. Ensure your project structure and paths are correct.")
    sys.exit(1)


def test_connection_url():
    """Verify that the database URL is loaded and uses the correct port."""
    print("1. Testing .env configuration...")
    # Check that the URL is loaded and, importantly, uses the correct host port (5433)
    assert SQLALCHEMY_DATABASE_URL is not None, "SQLALCHEMY_DATABASE_URL is missing. Check your .env file."
    assert 'localhost:5433' in SQLALCHEMY_DATABASE_URL, f"URL should use port 5433, got: {SQLALCHEMY_DATABASE_URL}"
    print(f"✅ URL loaded and uses port 5433: {SQLALCHEMY_DATABASE_URL}")

def test_database_connectivity():
    """Attempt to execute a trivial query to confirm the database is reachable."""
    print("\n2. Testing database connectivity...")
    # Attempt to connect and execute a simple 'SELECT 1' query
    with engine.connect() as connection:
        # FIX: Wrap the SQL string in text()
        connection.execute(text("SELECT 1"))
    print("✅ Database connectivity test passed! Connection established.")

def test_session_management():
    """Test creating a session and ensuring it's a valid object."""
    print("\n3. Testing SQLAlchemy session creation...")
    # SessionLocal should create a valid session object
    db = SessionLocal()
    try:
        assert isinstance(db, Session), f"SessionLocal should return a Session object, got {type(db)}"
        print("✅ Session creation successful (returned a SQLAlchemy Session object).")
    finally:
        db.close()

def test_get_db_dependency():
    """Test the get_db generator function."""
    print("\n4. Testing get_db dependency function...")
    # Use context manager style to test the generator
    db_generator = get_db()
    session = next(db_generator)  # Get the session

    assert isinstance(session, Session), f"get_db should yield a Session object, got {type(session)}"
    print("✅ get_db yielded a valid Session object.")

    # This line attempts to execute the 'finally' block of the generator
    try:
        next(db_generator)
    except StopIteration:
        print("✅ get_db successfully closed the session (StopIteration occurred).")
    else:
        raise AssertionError("get_db generator should raise StopIteration after yielding session")


def main_test():
    """Run all connection tests."""
    print("--- Running Database Connection Tests ---")
    
    # Run tests and collect results
    results = [
        test_connection_url(),
        test_database_connectivity(),
        test_session_management(),
        test_get_db_dependency()
    ]

    if all(results):
        print("\n\n🎉 ALL 4 CONNECTION TESTS PASSED! Infrastructure is ready for ORM models.")
        sys.exit(0)
    else:
        print("\n\n🛑 ONE OR MORE CONNECTION TESTS FAILED. Please review the errors.")
        sys.exit(1)


if __name__ == "__main__":
    main_test()