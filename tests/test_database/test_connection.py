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
    if SQLALCHEMY_DATABASE_URL and 'localhost:5433' in SQLALCHEMY_DATABASE_URL:
        print(f"✅ URL loaded and uses port 5433: {SQLALCHEMY_DATABASE_URL}")
        return True
    else:
        print("❌ FAILED: SQLALCHEMY_DATABASE_URL is incorrect or missing. Check your .env file.")
        return False

def test_database_connectivity():
    """Attempt to execute a trivial query to confirm the database is reachable."""
    print("\n2. Testing database connectivity...")
    try:
        # Attempt to connect and execute a simple 'SELECT 1' query
        with engine.connect() as connection:
            # FIX: Wrap the SQL string in text()
            connection.execute(text("SELECT 1"))
        print("✅ Database connectivity test passed! Connection established.")
        return True
    except OperationalError as e:
        print("❌ FAILED: Database connection failed (OperationalError).")
        print(f"   -> Error: {e}")
        print("   -> Is your Docker PostgreSQL container running on port 5433?")
        return False
    except ProgrammingError as e:
        # This can happen if the database user/password/name is wrong but the port is open
        print("❌ FAILED: Database connection failed (ProgrammingError).")
        print(f"   -> Error: {e}")
        print("   -> Check your POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_DB in the .env file.")
        return False
    except Exception as e:
        print(f"❌ FAILED: An unexpected error occurred: {e}")
        return False

def test_session_management():
    """Test creating a session and ensuring it's a valid object."""
    print("\n3. Testing SQLAlchemy session creation...")
    try:
        # SessionLocal should create a valid session object
        db = SessionLocal()
        if isinstance(db, Session):
            print("✅ Session creation successful (returned a SQLAlchemy Session object).")
            db.close()
            return True
        else:
            print("❌ FAILED: SessionLocal did not return a SQLAlchemy Session object.")
            return False
    except Exception as e:
        print(f"❌ FAILED: Session creation failed: {e}")
        return False

def test_get_db_dependency():
    """Test the get_db generator function."""
    print("\n4. Testing get_db dependency function...")
    try:
        # Use context manager style to test the generator
        db_generator = get_db()
        session = next(db_generator)  # Get the session

        if isinstance(session, Session):
            print("✅ get_db yielded a valid Session object.")
            
            # This line attempts to execute the 'finally' block of the generator
            try:
                next(db_generator) 
            except StopIteration:
                print("✅ get_db successfully closed the session (StopIteration occurred).")
                return True
            except Exception as e:
                print(f"❌ FAILED: get_db failed to close session properly: {e}")
                return False
        else:
            print("❌ FAILED: get_db did not yield a Session object.")
            return False

    except OperationalError:
        print("❌ FAILED: get_db test failed (database is not available).")
        return False
    except Exception as e:
        print(f"❌ FAILED: An unexpected error occurred in get_db test: {e}")
        return False


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