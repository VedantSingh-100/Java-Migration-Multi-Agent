#!/usr/bin/env python3
"""
Batch migration script for all repositories in selected40.csv.
Resumes from last attempted repo if stopped and restarted.

Each migration runs as a separate subprocess for proper isolation.
"""
import argparse
import os
import sys
import time
import subprocess
import pandas as pd
from datetime import datetime


def run_single_migration(repo: str, base_commit: str, csv_path: str) -> tuple[bool, int]:
    """
    Run migration for a single repo as a subprocess.
    Returns (success, return_code).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    migrate_script = os.path.join(script_dir, "migrate_single_repo.py")

    cmd = [
        sys.executable,  # Use same Python interpreter
        migrate_script,
        repo,
        base_commit,
        "--csv", csv_path
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            timeout=7200,  # 2 hour timeout per repo
        )
        return result.returncode == 0, result.returncode
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: Migration exceeded 2 hour limit")
        return False, -1
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, -2


def main():
    parser = argparse.ArgumentParser(
        description="Migrate all repositories from selected40.csv"
    )
    parser.add_argument(
        "--csv",
        default="./docs/selected40.csv",
        help="Path to CSV file (default: ./docs/selected40.csv)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all attempted/migrated flags to False before starting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which repos would be migrated without actually migrating",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start from specific repo index (0-based)",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    total_repos = len(df)

    # Reset if requested
    if args.reset:
        df["attempted"] = False
        df["migrated"] = False
        df.to_csv(csv_path, index=False)
        print(f"Reset all {total_repos} repos to attempted=False, migrated=False")
        # Reload after reset
        df = pd.read_csv(csv_path)

    # Find repos that haven't been attempted yet
    pending = df[df["attempted"] == False]
    already_attempted = df[df["attempted"] == True]
    already_migrated = df[df["migrated"] == True]

    print("\n" + "=" * 70)
    print("BATCH MIGRATION STATUS")
    print("=" * 70)
    print(f"Total repositories:     {total_repos}")
    print(f"Already attempted:      {len(already_attempted)}")
    print(f"Already migrated:       {len(already_migrated)}")
    print(f"Pending (to migrate):   {len(pending)}")
    print("=" * 70 + "\n")

    if len(pending) == 0:
        print("All repositories have been attempted. Use --reset to start over.")
        sys.exit(0)

    if args.dry_run:
        print("DRY RUN - Would migrate the following repos:")
        for i, (idx, row) in enumerate(pending.iterrows()):
            print(f"  [{i+1}] {row['repo']} @ {row['base_commit'][:8]}")
        sys.exit(0)

    # Migrate each pending repo
    results = {"success": 0, "failed": 0}
    start_time = time.time()

    pending_list = list(pending.iterrows())

    for i, (idx, row) in enumerate(pending_list):
        if i < args.start_from:
            continue

        repo = row["repo"]
        base_commit = row["base_commit"]
        repo_num = len(already_attempted) + i + 1

        print("\n" + "=" * 70)
        print(f"[{repo_num}/{total_repos}] MIGRATING: {repo}")
        print(f"Base commit: {base_commit}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")

        try:
            success, return_code = run_single_migration(repo, base_commit, csv_path)

            if success:
                results["success"] += 1
                print(f"\n  Result: SUCCESS")
            else:
                results["failed"] += 1
                print(f"\n  Result: FAILED (exit code: {return_code})")

        except KeyboardInterrupt:
            print(f"\n\nInterrupted by user at repo: {repo}")
            print(f"Progress saved - resume by running script again.")
            print(f"(Repo {repo} was marked as attempted)")
            break

    # Final summary
    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60
    elapsed_hrs = elapsed / 3600

    print("\n" + "=" * 70)
    print("BATCH MIGRATION COMPLETE")
    print("=" * 70)
    if elapsed_hrs >= 1:
        print(f"Time elapsed:    {elapsed_hrs:.1f} hours")
    else:
        print(f"Time elapsed:    {elapsed_min:.1f} minutes")
    print(f"Successful:      {results['success']}")
    print(f"Failed:          {results['failed']}")
    print("=" * 70)

    # Reload and show final status
    df = pd.read_csv(csv_path)
    print(f"\nFinal CSV Status:")
    print(f"  Attempted: {len(df[df['attempted'] == True])}/{total_repos}")
    print(f"  Migrated:  {len(df[df['migrated'] == True])}/{total_repos}")


if __name__ == "__main__":
    main()
