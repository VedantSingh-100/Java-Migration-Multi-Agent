import os
import subprocess
from typing import Optional


def clone_and_prepare_repo(github_repo: str, base_commit: str, dest_dir: str, branch_name: str = "migration-base") -> Optional[str]:
    """
    Clone a GitHub repo, checkout to base_commit, and create a new branch.
    Returns the path to the cloned repo or None if failed.
    """
    repo_url = f"https://github.com/{github_repo}.git"
    if os.path.exists(dest_dir):
        # Ensure we're on a proper branch (not detached HEAD)
        orig_cwd = os.getcwd()
        try:
            os.chdir(dest_dir)
            # Check current branch state
            result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
            current_branch = result.stdout.strip()

            if not current_branch:  # Empty means detached HEAD
                print(f"[REPO_UTILS] Repository in detached HEAD state, checking out to {branch_name}")
                # Check if branch exists
                branch_check = subprocess.run(["git", "branch", "--list", branch_name], capture_output=True, text=True)
                if branch_check.stdout.strip():
                    # Branch exists, checkout to it
                    subprocess.run(["git", "checkout", branch_name], check=True)
                else:
                    # Branch doesn't exist, create it from current HEAD
                    subprocess.run(["git", "checkout", "-b", branch_name], check=True)
                print(f"[REPO_UTILS] Now on branch: {branch_name}")

            os.chdir(orig_cwd)
        except subprocess.CalledProcessError as e:
            print(f"[REPO_UTILS] Warning: Could not fix branch state: {e}")
            os.chdir(orig_cwd)

        # Ensure TODO.md and CURRENT_STATE.md exist in the project path
        for fname in ["TODO.md", "CURRENT_STATE.md"]:
            fpath = os.path.join(dest_dir, fname)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            if not os.path.exists(fpath):
                with open(fpath, "w") as f:
                    f.write(f"# {fname}\n\n")
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