from pathlib import Path
from typing import List, Optional
import os
import shutil
from git import Repo, GitCommandError
from langchain_core.tools import tool

# State files that must be preserved across git operations
STATE_FILES = [
    'TODO.md',
    'VISIBLE_TASKS.md',
    'COMPLETED_ACTIONS.md',
    'CURRENT_STATE.md',
    'analysis.md',
    'ERROR_HISTORY.md'
]


def _backup_state_files(repo_path: str) -> dict:
    """Backup state files before git operations that might remove them."""
    backups = {}
    for filename in STATE_FILES:
        filepath = os.path.join(repo_path, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    backups[filename] = f.read()
            except Exception:
                pass
    return backups


def _restore_state_files(repo_path: str, backups: dict):
    """Restore state files after git operations."""
    for filename, content in backups.items():
        filepath = os.path.join(repo_path, filename)
        try:
            with open(filepath, 'w') as f:
                f.write(content)
        except Exception:
            pass


@tool
def list_branches(repo_path: str) -> str:
    """List all branches in the repository."""
    try:
        repo = Repo(repo_path)
        branches = [b.name for b in repo.branches]
        return "Branches:\n" + "\n".join(branches)
    except Exception as e:
        return f"Error listing branches: {e}"


@tool
def create_branch(repo_path: str, branch_name: str) -> str:
    """Create a new branch."""
    try:
        repo = Repo(repo_path)
        repo.git.branch(branch_name)
        return f"Created branch '{branch_name}'"
    except Exception as e:
        return f"Error creating branch: {e}"


@tool
def checkout_branch(repo_path: str, branch_name: str) -> str:
    """Checkout (switch to) a branch. State files (TODO.md, VISIBLE_TASKS.md, etc.) are preserved."""
    try:
        # PROTECTION: Backup state files before checkout
        # This prevents loss of migration state when switching branches
        backups = _backup_state_files(repo_path)
        backed_up_files = list(backups.keys())

        repo = Repo(repo_path)
        repo.git.checkout(branch_name)

        # PROTECTION: Restore state files after checkout
        if backups:
            _restore_state_files(repo_path, backups)
            return f"Checked out branch '{branch_name}'. Preserved state files: {', '.join(backed_up_files)}"

        return f"Checked out branch '{branch_name}'"
    except Exception as e:
        # Even on error, try to restore state files if we backed them up
        if 'backups' in locals() and backups:
            _restore_state_files(repo_path, backups)
        return f"Error checking out branch: {e}"


@tool
def commit_changes(repo_path: str, message: str, add_all: bool = True) -> str:
    """
    Stage changes (all by default) and commit with the given message.
    Set add_all=False to require explicit git.add calls beforehand.

    IMPORTANT: This tool verifies test method preservation before committing.
    If test methods have been renamed, deleted, or modified inappropriately,
    the commit will be BLOCKED with an error message.
    """
    try:
        # STEP 1: Verify test preservation BEFORE committing
        from src.utils.test_verifier import verify_test_preservation_before_commit

        is_valid, verification_msg = verify_test_preservation_before_commit(repo_path)

        if not is_valid:
            # Block the commit - test preservation violated
            return f"""COMMIT BLOCKED - TEST PRESERVATION VIOLATION

{verification_msg}

Your commit was NOT created. You must:
1. Revert changes to test files
2. Fix the APPLICATION code instead
3. Use @Disabled annotation if a test truly cannot work

DO NOT rename, delete, or rewrite test methods."""

        # STEP 2: Proceed with commit if verification passed
        repo = Repo(repo_path)
        if add_all:
            repo.git.add(A=True)
        commit = repo.index.commit(message)
        return f"Committed changes: {commit.hexsha} - {message}"
    except ImportError:
        # If verifier not available, proceed without verification (backwards compatibility)
        try:
            repo = Repo(repo_path)
            if add_all:
                repo.git.add(A=True)
            commit = repo.index.commit(message)
            return f"Committed changes: {commit.hexsha} - {message}"
        except Exception as e:
            return f"Error committing changes: {e}"
    except Exception as e:
        return f"Error committing changes: {e}"


@tool
def tag_checkpoint(repo_path: str, tag_name: str, message: Optional[str] = None) -> str:
    """Create an annotated tag as a checkpoint."""
    try:
        repo = Repo(repo_path)
        repo.create_tag(path=tag_name, message=message or f"Checkpoint {tag_name}")
        return f"Created tag '{tag_name}'"
    except Exception as e:
        return f"Error tagging checkpoint: {e}"


@tool
def get_status(repo_path: str) -> str:
    """Get git status for the working directory."""
    try:
        repo = Repo(repo_path)
        return repo.git.status()
    except Exception as e:
        return f"Error getting status: {e}"


@tool
def get_log(repo_path: str, max_entries: int = 10) -> str:
    """Show the most recent commits (default 10)."""
    try:
        repo = Repo(repo_path)
        logs = repo.git.log(f"-n{max_entries}", "--oneline")
        return logs
    except Exception as e:
        return f"Error getting log: {e}"


# Collect all Git tools
git_tools = [
    list_branches,
    create_branch,
    checkout_branch,
    commit_changes,
    tag_checkpoint,
    get_status,
    get_log,
]