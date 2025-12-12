"""
LangChain tools for file operations in migration agents
"""

from pathlib import Path
from langchain_core.tools import tool
from src.utils.logging_config import log_summary
import re


@tool
def read_file(file_path: str) -> str:
    """Read content from a file."""
    try:
        content = Path(file_path).read_text(encoding='utf-8')
        log_summary(f"FILE READ SUCCESS: {len(content)} characters from {file_path}")
        return content
    except Exception as e:
        log_summary(f"FILE READ ERROR: {str(e)} for {file_path}")
        return f"Error reading file: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file.
    Call this function with a file_path and content properly else you'll get the following error

    Field required [type=missing, input_value={'file_path': '*.java'}, input_type=dict]
    """
    try:
        Path(file_path).write_text(content, encoding='utf-8')
        log_summary(f"FILE WRITE SUCCESS: {file_path}")
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        log_summary(f"FILE WRITE ERROR: {str(e)} for {file_path}")
        return f"Error writing file: {str(e)}"


@tool
def find_replace(file_path: str, find_text: str, replace_text: str) -> str:
    """Find and replace text in a file."""
    try:
        content = Path(file_path).read_text(encoding='utf-8')
        count = content.count(find_text)
        if count > 0:
            new_content = content.replace(find_text, replace_text)
            Path(file_path).write_text(new_content, encoding='utf-8')
            log_summary(f"FIND REPLACE SUCCESS: {count} replacements in {file_path} - '{find_text}' -> '{replace_text}'")
            return f"Made {count} replacements in {file_path}"
        else:
            log_summary(f"FIND REPLACE NO MATCH: No occurrences of '{find_text}' found in {file_path} (intended replace: '{replace_text}')")
            return f"No occurrences of '{find_text}' found in {file_path}"
    except Exception as e:
        log_summary(f"FIND REPLACE ERROR: {str(e)} for {file_path} - '{find_text}' -> '{replace_text}'")
        return f"Error: {str(e)}"


@tool
def list_java_files(directory: str) -> str:
    """List all Java files in a directory and subdirectories."""
    try:
        java_files = [str(f) for f in Path(directory).rglob("*.java")]
        log_summary(f"LIST JAVA FILES SUCCESS: Found {len(java_files)} files in {directory}")
        return f"Found {len(java_files)} Java files:\n" + "\n".join(java_files)
    except Exception as e:
        log_summary(f"LIST JAVA FILES ERROR: {str(e)} for {directory}")
        return f"Error: {str(e)}"


@tool
def search_files(directory: str, pattern: str) -> str:
    """Search for a regex pattern in Java files."""
    try:
        matches = []
        java_files = [str(f) for f in Path(directory).rglob("*.java")]

        for file_path in java_files:
            content = Path(file_path).read_text(encoding='utf-8')
            for line_num, line in enumerate(content.split('\n'), 1):
                if re.search(pattern, line):
                    matches.append(f"{file_path}:{line_num}: {line.strip()}")

        if matches:
            log_summary(f"SEARCH FILES SUCCESS: {len(matches)} matches for '{pattern}' in {directory}")
            return f"Found {len(matches)} matches:\n" + "\n".join(matches[:20])
        else:
            log_summary(f"SEARCH FILES NO MATCHES: Pattern '{pattern}' not found in {directory}")
            return f"No matches found for pattern '{pattern}'"
    except Exception as e:
        log_summary(f"SEARCH FILES ERROR: {str(e)} for pattern '{pattern}' in {directory}")
        return f"Error: {str(e)}"


@tool
def file_exists(file_path: str) -> str:
    """Check if a file exists."""
    exists = str(Path(file_path).exists())
    log_summary(f"FILE EXISTS RESULT: {file_path} exists={exists}")
    return exists


@tool
def revert_test_files(repo_path: str) -> str:
    """
    Revert ALL modified test files back to their last committed state.

    USE THIS TOOL when you've accidentally modified test files and need to undo
    your changes. This is the proper way to recover from test preservation violations.

    This tool will:
    1. Find all modified test files (files matching *Test.java, *Tests.java, etc.)
    2. Revert them to HEAD (last commit)
    3. Report which files were reverted

    Args:
        repo_path: Path to the repository root

    Returns:
        Summary of reverted files or error message
    """
    import subprocess
    import os

    try:
        # Find modified files
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return f"Error getting git status: {result.stderr}"

        # Parse modified files and filter for test files
        test_patterns = ['Test.java', 'Tests.java', 'TestCase.java', 'IT.java', 'ITCase.java']
        modified_test_files = []

        for line in result.stdout.split('\n'):
            if not line or len(line) < 4:
                continue
            # Git status --porcelain format: "XY filename" where:
            # - X is index status (position 0)
            # - Y is work tree status (position 1)
            # - Space (position 2)
            # - Filename (position 3 onwards)
            file_path = line[3:].strip()

            # Handle renamed files (format: "R  old -> new")
            if ' -> ' in file_path:
                file_path = file_path.split(' -> ')[1]

            # Check if it's a test file
            if any(file_path.endswith(pattern) for pattern in test_patterns):
                modified_test_files.append(file_path)

        if not modified_test_files:
            log_summary("REVERT TEST FILES: No modified test files found")
            return "No modified test files found. Nothing to revert."

        # Revert each test file
        reverted = []
        errors = []

        for test_file in modified_test_files:
            full_path = os.path.join(repo_path, test_file)

            # Check if file exists (might be deleted)
            revert_result = subprocess.run(
                ["git", "checkout", "HEAD", "--", test_file],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if revert_result.returncode == 0:
                reverted.append(test_file)
                log_summary(f"REVERTED TEST FILE: {test_file}")
            else:
                errors.append(f"{test_file}: {revert_result.stderr}")

        # Build response
        response_lines = [
            "=" * 60,
            "TEST FILE REVERT COMPLETE",
            "=" * 60,
            "",
        ]

        if reverted:
            response_lines.append(f"✅ Successfully reverted {len(reverted)} test file(s):")
            for f in reverted:
                response_lines.append(f"   - {f}")
            response_lines.append("")

        if errors:
            response_lines.append(f"❌ Failed to revert {len(errors)} file(s):")
            for e in errors:
                response_lines.append(f"   - {e}")
            response_lines.append("")

        response_lines.extend([
            "Next steps:",
            "1. Run 'mvn compile' to check for compilation errors",
            "2. Fix APPLICATION code (not tests) to resolve any errors",
            "3. If a test truly cannot work, add @Disabled annotation",
            "=" * 60,
        ])

        return "\n".join(response_lines)

    except subprocess.TimeoutExpired:
        return "Error: Git command timed out"
    except Exception as e:
        log_summary(f"REVERT TEST FILES ERROR: {str(e)}")
        return f"Error reverting test files: {str(e)}"


# Collect all file tools
file_tools = [read_file, write_file, find_replace, list_java_files, search_files, file_exists, revert_test_files]