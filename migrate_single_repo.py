#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
from loguru import logger
from src.utils.repo_utils import clone_and_prepare_repo
from src.utils.logging_config import setup_migration_logging, log_summary, log_console
from supervisor_orchestrator_refactored import SupervisorMigrationOrchestrator


def setup_java_environment(target_version: str = "21") -> bool:
    """
    Configure JAVA_HOME and PATH for the target Java version.
    Returns True if successful, False otherwise.
    """
    # Map of Java versions to their installation paths
    java_paths = {
        "21": "/usr/lib/jvm/java-21-openjdk",
        "17": "/usr/lib/jvm/java-17-openjdk",
        "11": "/usr/lib/jvm/java-11-openjdk",
        "8": "/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.452.b09-2.el9.x86_64",
    }

    # Get the path for the target version
    java_home = java_paths.get(target_version)

    if not java_home:
        print(f"ERROR: Unknown Java version: {target_version}")
        print(f"Supported versions: {list(java_paths.keys())}")
        return False

    # Check if the Java installation exists
    if not os.path.isdir(java_home):
        print(f"ERROR: Java {target_version} not installed at {java_home}")
        print("Please install Java using: sudo dnf install java-{version}-openjdk-devel")
        return False

    # Set JAVA_HOME
    os.environ["JAVA_HOME"] = java_home

    # Prepend Java bin to PATH
    java_bin = os.path.join(java_home, "bin")
    current_path = os.environ.get("PATH", "")

    # Remove any existing Java paths from PATH to avoid conflicts
    path_parts = current_path.split(os.pathsep)
    filtered_parts = [p for p in path_parts if "/jvm/" not in p]
    new_path = os.pathsep.join([java_bin] + filtered_parts)
    os.environ["PATH"] = new_path

    # Verify the setup
    java_executable = os.path.join(java_bin, "java")
    if not os.path.isfile(java_executable):
        print(f"ERROR: Java executable not found at {java_executable}")
        return False

    print(f"Java environment configured:")
    print(f"  JAVA_HOME = {java_home}")
    print(f"  Java bin  = {java_bin}")

    return True


def migrate(repo: str, base_commit: str, csv_path: str):
    log_summary(f"CSV STATUS UPDATE: Marking {repo} as attempted")

    df = pd.read_csv(csv_path)
    df.loc[df["repo"] == repo, "attempted"] = True
    df.to_csv(csv_path, index=False)

    # Get the absolute path to the migration script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Allow custom repos directory via environment variable (for model comparison experiments)
    repos_base = os.environ.get("REPOS_DIR", os.path.join(script_dir, "repositories"))
    dest_dir = os.path.join(repos_base, repo.replace("/", "_"))
    log_console(f"Preparing repository: {repo} at {base_commit}")
    log_summary(f"APPROACH: Cloning repository to {dest_dir}")

    repo_path = clone_and_prepare_repo(repo, base_commit, dest_dir)
    if not repo_path:
        log_console(f"Failed to prepare {repo}", "ERROR")
        log_summary(f"ERROR: Repository preparation failed for {repo}")
        return False

    log_summary(f"Repository prepared successfully at: {repo_path}")
    log_console("Starting migration orchestrator...")
    log_summary("APPROACH: Initializing SupervisorMigrationOrchestrator")

    # Set environment variables for test invariance verification
    # These are used by completion_tools.py to run final MigrationBench check
    os.environ['MIGRATION_BASE_COMMIT'] = base_commit
    os.environ['MIGRATION_REPO_PATH'] = repo_path
    log_summary(f"Set MIGRATION_BASE_COMMIT={base_commit}")
    log_summary(f"Set MIGRATION_REPO_PATH={repo_path}")

    # Import and reset token counter for this migration
    from supervisor_orchestrator_refactored import tc, MAX_LLM_CALLS
    tc.reset()
    log_summary(f"Token counter reset - LLM call limit set to {MAX_LLM_CALLS}")

    orchestrator = SupervisorMigrationOrchestrator()

    log_summary("APPROACH: Running multi-agent migration process")
    result = orchestrator.migrate_project(repo_path)
    success = result.get("success", False)
    status = result.get("status", "unknown")
    reason = result.get("result", "<no-result>")

    # Helper function to log token stats
    def log_token_stats_if_available():
        if "token_stats" in result:
            token_stats = result["token_stats"]
            log_console(f"\nTotal Migration Cost: ${token_stats['total_cost_usd']:.4f}")
            log_console(f"Total Tokens: {token_stats['total_tokens']:,}")
            log_console(f"LLM Calls: {token_stats['llm_calls']:,}")

            log_summary("\n" + "="*60)
            log_summary("FINAL TOKEN USAGE SUMMARY")
            log_summary("="*60)
            log_summary(f"LLM Calls:       {token_stats['llm_calls']:,}")
            log_summary(f"Prompt tokens:   {token_stats['prompt_tokens']:,}")
            log_summary(f"Response tokens: {token_stats['response_tokens']:,}")
            log_summary(f"Total tokens:    {token_stats['total_tokens']:,}")
            log_summary("-"*60)
            log_summary(f"Prompt cost:     ${token_stats['prompt_cost_usd']:.4f}")
            log_summary(f"Response cost:   ${token_stats['response_cost_usd']:.4f}")
            log_summary(f"TOTAL COST:      ${token_stats['total_cost_usd']:.4f}")
            log_summary("="*60)

    if success:
        if status == "partial":
            # PARTIAL_SUCCESS: Build passes, tests pass, but < 90% tasks complete
            log_console(f"Migration PARTIAL SUCCESS for {repo}", "SUCCESS")
            log_summary(f"RESULT: Migration partially complete")
            log_summary(f"STATUS: PARTIAL_SUCCESS")
            log_summary(f"DETAILS: {reason}")
            # Do NOT mark as migrated in CSV - only full SUCCESS counts
            log_summary(f"CSV STATUS: Kept as 'attempted' (partial success does not mark migrated)")
            log_token_stats_if_available()
        else:
            # Full SUCCESS: >= 90% tasks complete, build passes, tests pass
            log_console(f"Migration successful for {repo}", "SUCCESS")
            log_summary(f"RESULT: Migration completed successfully")
            log_summary(f"STATUS: SUCCESS")
            log_summary(f"DETAILS: {reason}")
            log_token_stats_if_available()
            # Mark migrated in CSV - only for full SUCCESS
            df.loc[df["repo"] == repo, "migrated"] = True
            df.to_csv(csv_path, index=False)
            log_summary(f"CSV STATUS UPDATE: Marked {repo} as migrated")
    else:
        # Handle different failure types
        limit_exceeded = result.get("limit_exceeded", False)

        if limit_exceeded:
            log_console(f"Migration stopped for {repo} - LLM call limit reached", "WARNING")
            log_summary(f"RESULT: Migration incomplete - LLM call limit exceeded")
            log_summary(f"STATUS: INCOMPLETE")
            log_summary(f"DETAILS: {reason}")
        elif status == "incomplete":
            log_console(f"Migration incomplete for {repo}", "WARNING")
            log_summary(f"RESULT: Migration incomplete")
            log_summary(f"STATUS: INCOMPLETE")
            log_summary(f"DETAILS: {reason}")
        else:
            log_console(f"Migration failed for {repo}", "ERROR")
            log_summary(f"RESULT: Migration failed")
            log_summary(f"STATUS: FAILURE")
            log_summary(f"FAILURE REASON: {reason}")  # Now contains actual error, not agent message

        log_token_stats_if_available()

    print(f"Result for {repo}: success={success}, status={status}, detail={reason}")
    if result.get("limit_exceeded"):
        print(f"⚠ WARNING: {result.get('warning', 'LLM call limit exceeded')}")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run migration for a single repo and update selected.csv if successful."
    )
    parser.add_argument("repo", help="Repository, e.g., 'owner/name'")
    parser.add_argument("base_commit", help="Base commit SHA")
    parser.add_argument(
        "--csv",
        default="./selected.csv",
        help="Path to selected.csv (will be updated on success)",
    )
    parser.add_argument(
        "--target-java-version",
        default=os.environ.get("TARGET_JAVA_VERSION", "21"),
        help="Target Java version for migration (default: 21). Set via CLI or TARGET_JAVA_VERSION env var.",
    )
    args = parser.parse_args()

    # Set the target Java version in environment for all components to use
    os.environ["TARGET_JAVA_VERSION"] = args.target_java_version

    # Setup Java environment BEFORE anything else
    if not setup_java_environment(args.target_java_version):
        print(f"FATAL: Failed to configure Java {args.target_java_version} environment")
        print("Migration cannot proceed without proper Java setup.")
        sys.exit(1)

    # Setup structured logging
    log_files = setup_migration_logging(args.repo)

    log_summary(f"MIGRATION TARGET: {args.repo}")
    log_summary(f"BASE COMMIT: {args.base_commit}")
    log_summary(f"CSV FILE: {args.csv}")
    log_summary(f"TARGET JAVA VERSION: {args.target_java_version}")
    log_console(f"Starting migration for {args.repo}")
    log_console(f"Target Java version: {args.target_java_version}")

    try:
        ok = migrate(args.repo, args.base_commit, args.csv)

        if ok:
            log_summary("FINAL RESULT: SUCCESS")
        else:
            log_summary("FINAL RESULT: FAILURE")

        log_summary("MIGRATION SESSION ENDED")
        log_summary("=" * 80)
        exit(0 if ok else 1)

    except KeyboardInterrupt:
        log_console("Migration interrupted by user", "WARNING")
        log_summary("INTERRUPTION: User cancelled migration")
        exit(130)
    except Exception as e:
        log_console(f"Unexpected error: {e}", "ERROR")
        log_summary(f"UNEXPECTED ERROR: {str(e)}")
        exit(1)