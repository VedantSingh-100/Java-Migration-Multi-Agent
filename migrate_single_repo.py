#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from loguru import logger
from src.utils.repo_utils import clone_and_prepare_repo
from src.utils.logging_config import setup_migration_logging, log_summary, log_console
from supervisor_orchestrator_refactored import SupervisorMigrationOrchestrator


def migrate(repo: str, base_commit: str, csv_path: str):
    log_summary(f"CSV STATUS UPDATE: Marking {repo} as attempted")

    df = pd.read_csv(csv_path)
    df.loc[df["repo"] == repo, "attempted"] = True
    df.to_csv(csv_path, index=False)

    # Get the absolute path to the migration script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Repositories should be in migration/repositories (sibling to this script)
    dest_dir = os.path.join(script_dir, "repositories", repo.replace("/", "_"))
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
    args = parser.parse_args()

    # Setup structured logging
    log_files = setup_migration_logging(args.repo)

    log_summary(f"MIGRATION TARGET: {args.repo}")
    log_summary(f"BASE COMMIT: {args.base_commit}")
    log_summary(f"CSV FILE: {args.csv}")
    log_console(f"Starting migration for {args.repo}")

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