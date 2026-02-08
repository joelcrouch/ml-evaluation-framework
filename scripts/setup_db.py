#!/usr/bin/env python3
"""
DEPRECATED: This script is no longer supported.

This script previously used Base.metadata.create_all() to create database tables,
which bypasses Alembic migrations and causes schema drift.

=============================================================================
USE scripts/init_db.py INSTEAD
=============================================================================

To initialize your database, run:
    python scripts/init_db.py

This will run Alembic migrations, which is the ONLY supported way to create
and manage database schema.

Why this matters:
- Base.metadata.create_all() creates tables directly from Python models
- Alembic migrations track schema changes over time
- Using both causes schema drift and unpredictable behavior
- Tests, dev, and prod must all use the same schema creation method

For more information, see the project documentation on database management.
"""

import sys

def main():
    print("=" * 80)
    print("ERROR: This script is deprecated and should not be used.")
    print("=" * 80)
    print()
    print("Base.metadata.create_all() bypasses Alembic migrations and causes")
    print("schema drift between environments.")
    print()
    print("To initialize your database, use:")
    print()
    print("    python scripts/init_db.py")
    print()
    print("This will run Alembic migrations (the correct way to manage schema).")
    print("=" * 80)
    sys.exit(1)


if __name__ == "__main__":
    main()