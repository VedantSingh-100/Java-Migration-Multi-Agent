"""
LangChain tools for file operations in migration agents
"""

from pathlib import Path
from langchain_core.tools import tool
from src.utils.logging_config import log_summary
import re
import os


# Global project path - set by orchestrator before tool execution
_current_project_path: str = None


# =============================================================================
# JAVA VERSION GUARDRAIL FOR FIND_REPLACE
# =============================================================================

def _get_target_java_version() -> int:
    """Get the configured target Java version as an integer."""
    target_version = os.environ.get("TARGET_JAVA_VERSION", "21")
    return int(target_version) if target_version.isdigit() else 21


# Extended patterns to catch more Java version formats
JAVA_VERSION_PATTERNS = [
    # Standard Maven properties
    r'<java\.version>\s*(\d+)\s*</java\.version>',
    r'<maven\.compiler\.source>\s*(\d+)\s*</maven\.compiler\.source>',
    r'<maven\.compiler\.target>\s*(\d+)\s*</maven\.compiler\.target>',
    r'<maven\.compiler\.release>\s*(\d+)\s*</maven\.compiler\.release>',
    r'<release>\s*(\d+)\s*</release>',
    r'<source>\s*(\d+)\s*</source>',
    r'<target>\s*(\d+)\s*</target>',
    # Handle 1.x format (e.g., 1.8)
    r'<java\.version>\s*1\.(\d+)\s*</java\.version>',
    r'<maven\.compiler\.source>\s*1\.(\d+)\s*</maven\.compiler\.source>',
    r'<maven\.compiler\.target>\s*1\.(\d+)\s*</maven\.compiler\.target>',
]


def _check_java_version_downgrade_in_replacement(find_text: str, replace_text: str) -> tuple:
    """
    Check if a find_replace operation is attempting to downgrade Java version.

    Returns:
        (is_blocked, message) - True if downgrade should be blocked
    """
    target_int = _get_target_java_version()
    target_version = os.environ.get("TARGET_JAVA_VERSION", "21")

    for pattern in JAVA_VERSION_PATTERNS:
        # Check if find_text contains a Java version tag
        find_match = re.search(pattern, find_text, re.IGNORECASE)
        replace_match = re.search(pattern, replace_text, re.IGNORECASE)

        if find_match and replace_match:
            old_version = int(find_match.group(1))
            new_version = int(replace_match.group(1))

            # Block if: we're lowering the version AND it goes below target
            if new_version < old_version and new_version < target_int:
                msg = (
                    f"🚫 BLOCKED: Java version downgrade detected in find_replace! "
                    f"Attempting to change from {old_version} to {new_version}. "
                    f"Target version is {target_version}. "
                    f"Downgrading is forbidden. Fix dependency versions instead."
                )
                log_summary(f"GUARDRAIL BLOCKED FIND_REPLACE: Java downgrade {old_version} -> {new_version}")
                return True, msg

    return False, ""


def _check_java_version_in_content(content: str, file_path: str) -> tuple:
    """
    Check if file content contains Java version below target.
    Used by write_file to prevent writing pom.xml with downgraded Java version.

    Returns:
        (is_blocked, message) - True if content should be blocked
    """
    # Only check pom.xml files
    if not file_path.endswith('pom.xml'):
        return False, ""

    target_int = _get_target_java_version()
    target_version = os.environ.get("TARGET_JAVA_VERSION", "21")

    versions_found = []

    for pattern in JAVA_VERSION_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            version = int(match.group(1))
            versions_found.append(version)

            if version < target_int:
                msg = (
                    f"🚫 BLOCKED: write_file attempting to write pom.xml with Java version {version}! "
                    f"Target version is {target_version}. "
                    f"Downgrading Java version is forbidden. "
                    f"Fix dependency versions instead of lowering Java version."
                )
                log_summary(f"GUARDRAIL BLOCKED WRITE_FILE: Java version {version} < target {target_int} in pom.xml")
                return True, msg

    return False, ""


def set_project_path(path: str):
    """Set the current project path for file operations.

    This is called by the orchestrator at migration start to ensure
    all file operations are constrained to the project directory.
    """
    global _current_project_path
    _current_project_path = path
    log_summary(f"FILE_OPS: Project path set to '{path}'")


def _resolve_path(file_path: str) -> str:
    """Resolve file_path relative to project_path if not absolute.

    - Relative paths: prefix with project_path
    - Absolute paths within project: pass through
    - Absolute paths outside project: BLOCKED (raises ValueError)

    This prevents agents from accidentally reading/writing files
    outside the repository being migrated.
    """
    global _current_project_path

    if not _current_project_path:
        # No project context, use path as-is (backwards compatibility)
        return file_path

    path = Path(file_path)
    project = Path(_current_project_path).resolve()

    # If relative path, prefix with project_path
    if not path.is_absolute():
        resolved = project / file_path
        log_summary(f"PATH RESOLVED: '{file_path}' -> '{resolved}'")
        return str(resolved)

    # If absolute, verify it's within project - BLOCK if outside
    resolved = path.resolve()
    try:
        resolved.relative_to(project)
        return str(resolved)
    except ValueError:
        # Path is outside project - BLOCK this operation
        error_msg = f"BLOCKED: Path '{file_path}' is outside project root '{project}'"
        log_summary(f"PATH BLOCKED: {error_msg}")
        raise ValueError(error_msg)


@tool
def read_file(file_path: str) -> str:
    """Read content from a file."""
    try:
        resolved_path = _resolve_path(file_path)
        content = Path(resolved_path).read_text(encoding='utf-8')
        log_summary(f"FILE READ SUCCESS: {len(content)} characters from {resolved_path}")
        return content
    except Exception as e:
        log_summary(f"FILE READ ERROR: {str(e)} for {file_path}")
        return f"Error reading file: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file.
    Call this function with a file_path and content properly else you'll get the following error

    Field required [type=missing, input_value={'file_path': '*.java'}, input_type=dict]

    NOTE: Writing pom.xml with Java version below TARGET_JAVA_VERSION will be BLOCKED.
    """
    try:
        resolved_path = _resolve_path(file_path)

        # GUARDRAIL: Check for Java version downgrade in pom.xml content
        is_blocked, block_msg = _check_java_version_in_content(content, resolved_path)
        if is_blocked:
            return block_msg

        Path(resolved_path).write_text(content, encoding='utf-8')
        log_summary(f"FILE WRITE SUCCESS: {resolved_path}")
        return f"Successfully wrote to {resolved_path}"
    except Exception as e:
        log_summary(f"FILE WRITE ERROR: {str(e)} for {file_path}")
        return f"Error writing file: {str(e)}"


@tool
def find_replace(file_path: str, find_text: str, replace_text: str) -> str:
    """Find and replace text in a file. BLOCKS Java version downgrades."""
    # GUARDRAIL: Check for Java version downgrade attempts
    is_blocked, block_msg = _check_java_version_downgrade_in_replacement(find_text, replace_text)
    if is_blocked:
        return block_msg

    try:
        resolved_path = _resolve_path(file_path)
        content = Path(resolved_path).read_text(encoding='utf-8')
        count = content.count(find_text)
        if count > 0:
            new_content = content.replace(find_text, replace_text)
            Path(resolved_path).write_text(new_content, encoding='utf-8')
            log_summary(f"FIND REPLACE SUCCESS: {count} replacements in {resolved_path} - '{find_text}' -> '{replace_text}'")
            return f"Made {count} replacements in {resolved_path}"
        else:
            log_summary(f"FIND REPLACE NO MATCH: No occurrences of '{find_text}' found in {resolved_path} (intended replace: '{replace_text}')")
            return f"No occurrences of '{find_text}' found in {resolved_path}"
    except Exception as e:
        log_summary(f"FIND REPLACE ERROR: {str(e)} for {file_path} - '{find_text}' -> '{replace_text}'")
        return f"Error: {str(e)}"


@tool
def list_java_files(directory: str) -> str:
    """List all Java files in a directory and subdirectories."""
    try:
        resolved_dir = _resolve_path(directory)
        java_files = [str(f) for f in Path(resolved_dir).rglob("*.java")]
        log_summary(f"LIST JAVA FILES SUCCESS: Found {len(java_files)} files in {resolved_dir}")
        return f"Found {len(java_files)} Java files:\n" + "\n".join(java_files)
    except Exception as e:
        log_summary(f"LIST JAVA FILES ERROR: {str(e)} for {directory}")
        return f"Error: {str(e)}"


@tool
def search_files(directory: str, pattern: str) -> str:
    """Search for a regex pattern in Java files."""
    try:
        resolved_dir = _resolve_path(directory)
        matches = []
        java_files = [str(f) for f in Path(resolved_dir).rglob("*.java")]

        for file_path in java_files:
            content = Path(file_path).read_text(encoding='utf-8')
            for line_num, line in enumerate(content.split('\n'), 1):
                if re.search(pattern, line):
                    matches.append(f"{file_path}:{line_num}: {line.strip()}")

        if matches:
            log_summary(f"SEARCH FILES SUCCESS: {len(matches)} matches for '{pattern}' in {resolved_dir}")
            return f"Found {len(matches)} matches:\n" + "\n".join(matches[:20])
        else:
            log_summary(f"SEARCH FILES NO MATCHES: Pattern '{pattern}' not found in {resolved_dir}")
            return f"No matches found for pattern '{pattern}'"
    except Exception as e:
        log_summary(f"SEARCH FILES ERROR: {str(e)} for pattern '{pattern}' in {directory}")
        return f"Error: {str(e)}"


@tool
def file_exists(file_path: str) -> str:
    """Check if a file exists."""
    try:
        resolved_path = _resolve_path(file_path)
        exists = str(Path(resolved_path).exists())
        log_summary(f"FILE EXISTS RESULT: {resolved_path} exists={exists}")
        return exists
    except Exception as e:
        log_summary(f"FILE EXISTS ERROR: {str(e)} for {file_path}")
        return f"Error: {str(e)}"


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