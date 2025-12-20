import os
import subprocess
from typing import Optional


def clone_and_prepare_repo(github_repo: str, base_commit: str, dest_dir: str, branch_name: str = "migration-base") -> Optional[str]:
    """
    Clone a GitHub repo, checkout to base_commit, and create a new branch.
    Returns the path to the cloned repo or None if failed.

    IMPORTANT: If repo exists, it resets to base_commit and cleans all migration state files
    to ensure a fresh migration run.
    """
    repo_url = f"https://github.com/{github_repo}.git"
    if os.path.exists(dest_dir):
        print(f"[REPO_UTILS] Repository already exists at {dest_dir}, resetting to base commit")
        orig_cwd = os.getcwd()
        try:
            os.chdir(dest_dir)

            # Clean up any uncommitted changes first
            subprocess.run(["git", "reset", "--hard"], check=True)
            subprocess.run(["git", "clean", "-fd"], check=True)

            # Fetch latest from origin (in case base_commit needs it)
            subprocess.run(["git", "fetch", "origin"], check=False)  # Don't fail if offline

            # Checkout to base commit
            subprocess.run(["git", "checkout", base_commit], check=True)
            print(f"[REPO_UTILS] Reset to base commit: {base_commit}")

            # Delete existing migration branch if it exists, then recreate
            branch_check = subprocess.run(["git", "branch", "--list", branch_name], capture_output=True, text=True)
            if branch_check.stdout.strip():
                subprocess.run(["git", "branch", "-D", branch_name], check=True)
                print(f"[REPO_UTILS] Deleted old branch: {branch_name}")

            # Create fresh migration branch
            subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            print(f"[REPO_UTILS] Created fresh branch: {branch_name}")

            os.chdir(orig_cwd)
        except subprocess.CalledProcessError as e:
            print(f"[REPO_UTILS] Error resetting repo: {e}")
            os.chdir(orig_cwd)
            return None

        # Delete old migration state files to ensure fresh run
        state_files = ["TODO.md", "CURRENT_STATE.md", "COMPLETED_ACTIONS.md",
                       "analysis.md", "VISIBLE_TASKS.md", "ERROR_HISTORY.md", "MigrationReport.md"]
        for fname in state_files:
            fpath = os.path.join(dest_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"[REPO_UTILS] Removed old state file: {fname}")

        # Create fresh empty state files
        for fname in ["TODO.md", "CURRENT_STATE.md"]:
            fpath = os.path.join(dest_dir, fname)
            with open(fpath, "w") as f:
                f.write(f"# {fname}\n\n")

        print(f"[REPO_UTILS] Repository reset complete, ready for fresh migration")
        return dest_dir
    orig_cwd = os.getcwd()
    try:
        subprocess.run(["git", "clone", repo_url, dest_dir], check=True)
        os.chdir(dest_dir)
        subprocess.run(["git", "checkout", base_commit], check=True)
        subprocess.run(["git", "checkout", "-b", branch_name], check=True)
        # Ensure TODO.md and CURRENT_STATE.md exist in the project path
        for fname in ["TODO.md", "CURRENT_STATE.md"]:
            fpath = os.path.join(dest_dir, fname)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            if not os.path.exists(fpath):
                with open(fpath, "w") as f:
                    f.write(f"# {fname}\n\n")
        os.chdir(orig_cwd)
        return dest_dir
    except subprocess.CalledProcessError as e:
        print(f"Error cloning/preparing repo {github_repo}: {e}")
        os.chdir(orig_cwd)
        return None