"""Load a test suite into the ML Evaluation Framework database."""
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_eval.test_suite import TestSuiteManager
from ml_eval.database.connection import get_db


def main():
    """Main function to load test suites from files."""
    parser = argparse.ArgumentParser(
        description="Load a test suite into the ML Evaluation Framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load JSON test suite
  python scripts/load_suite.py data/example_suite.json

  # Load YAML test suite
  python scripts/load_suite.py data/example_suite.yaml

  # Skip duplicate test cases
  python scripts/load_suite.py data/example_suite.json --skip-duplicates

  # Include invalid test cases (not recommended)
  python scripts/load_suite.py data/example_suite.json --include-invalid
        """
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the test suite file (JSON or YAML)"
    )
    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="Skip test cases that already exist in the database"
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Attempt to save invalid test cases (may fail)"
    )

    args = parser.parse_args()

    # Initialize manager
    manager = TestSuiteManager()

    try:
        # Step 1: Load test suite from file
        print(f"📂 Loading test suite from: {args.file_path}")
        test_cases = manager.load_from_file(args.file_path)

        if not test_cases:
            print("⚠️  Warning: Test suite file is empty. No test cases to load.")
            sys.exit(0)

        print(f"✅ Successfully loaded {len(test_cases)} test cases")

        # Step 2: Extract metadata
        metadata = manager.get_suite_metadata(test_cases)
        print("\n📊 Suite Metadata:")
        print(f"  - Suite Name:    {metadata.get('suite_name') or 'N/A'}")
        print(f"  - Suite Version: {metadata.get('suite_version') or 'N/A'}")
        print(f"  - Total Cases:   {metadata['total_cases']}")
        print(f"  - Model Types:   {', '.join(metadata['model_types'])}")
        if metadata['tags']:
            print(f"  - Tags:          {', '.join(metadata['tags'])}")

        # Step 3: Validate test suite
        print("\n🔍 Validating test suite...")
        validation_report = manager.validate_suite(test_cases)

        # Step 4: Get database session
        db_session = next(get_db())

        try:
            # Step 5: Save to database
            print("\n💾 Saving test cases to database...")
            save_stats = manager.save_to_database(
                test_cases=test_cases,
                db_session=db_session,
                skip_duplicates=args.skip_duplicates,
                skip_invalid=not args.include_invalid
            )

            # Update report with duplicate count
            validation_report.duplicate_count = save_stats["skipped_duplicate"]

            # Step 6: Print results
            print("\n" + validation_report.render())

            print(f"\n💾 Database Save Results:")
            print(f"  - Saved:             {save_stats['saved']}")
            print(f"  - Skipped (invalid): {save_stats['skipped_invalid']}")
            print(f"  - Skipped (duplicate): {save_stats['skipped_duplicate']}")

            # Determine exit code
            if validation_report.has_critical_errors():
                print("\n⚠️  Test suite loaded with validation errors.")
                if save_stats['saved'] > 0:
                    print(f"✅ {save_stats['saved']} valid test cases were saved to the database.")
                sys.exit(1)  # Exit with error if there were validation issues
            else:
                print(f"\n✅ Success! All {save_stats['saved']} test cases loaded successfully.")
                sys.exit(0)

        finally:
            db_session.close()

    except FileNotFoundError as e:
        print(f"\n❌ File Error: {e}")
        sys.exit(2)
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
