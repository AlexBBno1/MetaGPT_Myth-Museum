"""
Migrate existing shorts folders to new naming convention.

This is a one-time migration script.

New format: {series_slug}_{topic_slug}/

Usage:
    python scripts/migrate_folders.py --dry-run   # Preview changes
    python scripts/migrate_folders.py             # Execute migration
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.folder_naming import (
    get_migration_plan,
    execute_migration,
    infer_series_from_folder,
)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate shorts folders to new naming")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )
    parser.add_argument(
        "--shorts-dir",
        type=Path,
        default=Path("outputs/shorts"),
        help="Path to shorts directory",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SHORTS FOLDER MIGRATION")
    print("=" * 60)
    print(f"Directory: {args.shorts_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print()
    
    # Get migration plan
    plan = get_migration_plan(args.shorts_dir)
    
    if not plan:
        print("No folders need migration.")
        return
    
    print(f"Found {len(plan)} folders to migrate:\n")
    
    # Show plan
    for action in plan:
        print(f"  {action['old_name']}")
        print(f"    -> {action['new_name']}")
        print(f"       Series: {action['series']}")
        print()
    
    # Execute or preview
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - No changes made")
        print("Run without --dry-run to execute migration")
        return
    
    # Confirm
    print("=" * 60)
    response = input("Proceed with migration? [y/N]: ")
    
    if response.lower() != 'y':
        print("Migration cancelled.")
        return
    
    # Execute
    print("\nExecuting migration...")
    results = execute_migration(plan, dry_run=False)
    
    # Report
    success = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    
    print()
    print("=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\nFailed migrations:")
        for r in results:
            if not r['success']:
                print(f"  {r['old_name']}: {r['error']}")


if __name__ == "__main__":
    main()
