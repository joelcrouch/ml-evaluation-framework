#!/usr/bin/env python3
"""
Database Initialization Script

This is the ONLY supported way to initialize the database schema.
It runs Alembic migrations to create all tables.

Usage:
    python scripts/init_db.py

Requirements:
    - PostgreSQL database must exist and be accessible
    - SQLALCHEMY_DATABASE_URL must be set in .env file
    - Alembic configuration must be present (alembic.ini, migrations/)
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_migrations():
    """Run Alembic migrations to initialize or update the database schema."""
    print("=" * 70)
    print("DATABASE INITIALIZATION - Alembic Migrations")
    print("=" * 70)

    # Verify environment setup
    try:
        from ml_eval.database.connection import SQLALCHEMY_DATABASE_URL, check_database_connection
        print(f"\n✅ Database URL loaded: {SQLALCHEMY_DATABASE_URL.split('@')[1] if '@' in SQLALCHEMY_DATABASE_URL else 'configured'}")
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please ensure your .env file exists and contains SQLALCHEMY_DATABASE_URL")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load database configuration: {e}")
        sys.exit(1)

    # Check database connection
    print("\n📡 Checking database connection...")
    try:
        check_database_connection()
    except Exception as e:
        print(f"\n❌ ERROR: Cannot connect to database")
        print(f"Details: {e}")
        print("\nTroubleshooting:")
        print("  1. Is PostgreSQL running?")
        print("  2. Is the database created?")
        print("  3. Are credentials in .env correct?")
        sys.exit(1)

    # Run Alembic migrations
    print("\n🔄 Running Alembic migrations (upgrade head)...")
    try:
        # Import Alembic here to avoid import issues
        from alembic.config import Config
        from alembic import command

        # Load Alembic configuration
        alembic_cfg = Config(str(project_root / "alembic.ini"))

        # Run migrations
        command.upgrade(alembic_cfg, "head")

        print("\n✅ SUCCESS: Database schema is up to date!")
        print("\n" + "=" * 70)
        print("Database is ready for use.")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ ERROR: Migration failed")
        print(f"Details: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that alembic.ini exists")
        print("  2. Check that migrations/ directory exists")
        print("  3. Run 'alembic current' to check migration state")
        print("  4. Run 'alembic history' to see available migrations")
        sys.exit(1)


if __name__ == "__main__":
    run_migrations()
