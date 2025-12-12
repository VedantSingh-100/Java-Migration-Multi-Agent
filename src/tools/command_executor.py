"""
LangChain tools for command execution in migration agents
"""

import subprocess
import re
import time
import os
import shutil
import xml.etree.ElementTree as ET
from src.utils.logging_config import log_summary
from langchain_core.tools import tool

# Import unified error classification system (Single Source of Truth)
from src.orchestrator.error_handler import unified_classifier, MavenErrorType


# Add blocked commands for safety
BLOCKED_COMMANDS = {
    'rm', 'del', 'rmdir', 'unlink',      # File deletion
    'format', 'fdisk', 'mkfs',           # Disk operations
    'sudo', 'su', 'doas',                # Privilege escalation
    'chmod', 'chown', 'chgrp',           # Permission changes
    'kill', 'killall', 'pkill',          # Process termination
    'shutdown', 'reboot', 'halt',        # System control
    'dd', 'shred',                       # Data destruction
    'mount', 'umount',                   # File system operations
    'crontab',                           # Scheduled tasks
    'systemctl', 'service',              # Service management
    'apt', 'yum', 'dnf', 'pacman',       # Package managers
    'pip', 'npm', 'gem', 'cargo',        # Language package managers
    'wget', 'curl',                      # Network downloads (can be dangerous)
    'ssh', 'scp', 'ftp', 'telnet',       # Remote access
    'nc', 'netcat', 'nmap',              # Network tools
    'docker', 'podman',                  # Container operations
}

BLOCKED_PATTERNS = [
    r'rm\s+.*-rf',                       # rm -rf specifically
    r'rm\s+.*-r.*f',                     # rm with recursive and force flags
    r'>\s*/dev/',                        # Writing to device files
    r'>\s*/etc/',                        # Writing to system config
    r'>\s*/usr/',                        # Writing to system directories
    r'>\s*/var/log/',                    # Writing to system logs
    r'>\s*/proc/',                       # Writing to proc filesystem
    r'>\s*/sys/',                        # Writing to sys filesystem
    r'\|\s*sh',                          # Piping to shell
    r'\|\s*bash',                        # Piping to bash
    r'\$\(',                             # Command substitution
    r'`.*`',                             # Backtick command substitution
    r'&&\s*(rm|del|format)',             # Command chaining with dangerous commands
    r';\s*(rm|del|format)',              # Command separation with dangerous commands
    r'\|\s*(rm|del|format)',             # Piping to dangerous commands
    r'--eval',                           # Eval flags
    r'-e\s+.*rm',                        # Execute flags with rm
    # Git operations that can remove state files (TODO.md, VISIBLE_TASKS.md, etc.)
    r'git\s+stash',                      # git stash removes uncommitted state files
    r'git\s+checkout\s+[^-]',            # git checkout <branch> can lose state files
    r'git\s+reset\s+--hard',             # git reset --hard removes uncommitted changes
    r'git\s+clean',                      # git clean removes untracked files
    r'git\s+restore\s+--staged',         # git restore can undo staged state files
]

ALLOWED_COMMANDS = {
    'mvn', 'maven',                      # Maven build tool
    'git',                               # Git version control
    'java', 'javac',                     # Java compiler/runtime
    'ls', 'dir',                         # Directory listing
    'cat', 'type',                       # File content viewing
    'grep', 'find', 'findstr',           # Text search
    'echo',                              # Text output
    'pwd', 'cd',                         # Directory navigation
    'mkdir',                             # Directory creation (safe)
    'cp', 'copy',                        # File copying (safer than mv)
    'head', 'tail',                      # File content viewing
    'wc',                                # Word/line counting
    'sort', 'uniq',                      # Text processing
    'which', 'where',                    # Command location
    'env', 'printenv',                   # Environment variables
}

# Cache for Maven path discovery
_maven_path_cache = None

def get_maven_env() -> dict:
    """
    Get environment dict with Maven properly configured.

    Searches for Maven in:
    1. MAVEN_HOME environment variable
    2. Common installation paths
    3. User's home directory

    Returns environment dict to pass to subprocess.
    """
    global _maven_path_cache

    env = os.environ.copy()

    # If already found Maven, use cached path
    if _maven_path_cache:
        maven_bin = os.path.join(_maven_path_cache, 'bin')
        env['PATH'] = maven_bin + os.pathsep + env.get('PATH', '')
        env['MAVEN_HOME'] = _maven_path_cache
        return env

    # Check if mvn is already in PATH
    existing_mvn = shutil.which('mvn')
    if existing_mvn:
        log_summary(f"MAVEN_ENV: Found mvn in PATH: {existing_mvn}")
        return env

    # Search for Maven installation
    home_dir = os.path.expanduser('~')
    search_paths = [
        # Check MAVEN_HOME first
        os.environ.get('MAVEN_HOME', ''),
        # User's home directory (common pattern)
        os.path.join(home_dir, 'apache-maven-3.9.11'),
        os.path.join(home_dir, 'apache-maven-3.9.9'),
        os.path.join(home_dir, 'apache-maven-3.9.8'),
        os.path.join(home_dir, 'apache-maven-3.9.6'),
        os.path.join(home_dir, 'maven'),
        os.path.join(home_dir, '.maven'),
        os.path.join(home_dir, '.sdkman/candidates/maven/current'),
        # System paths
        '/usr/local/maven',
        '/usr/local/apache-maven',
        '/opt/maven',
        '/opt/apache-maven',
        '/usr/share/maven',
    ]

    for maven_home in search_paths:
        if not maven_home:
            continue
        mvn_bin = os.path.join(maven_home, 'bin', 'mvn')
        if os.path.isfile(mvn_bin) and os.access(mvn_bin, os.X_OK):
            log_summary(f"MAVEN_ENV: Found Maven at {maven_home}")
            _maven_path_cache = maven_home
            maven_bin_dir = os.path.join(maven_home, 'bin')
            env['PATH'] = maven_bin_dir + os.pathsep + env.get('PATH', '')
            env['MAVEN_HOME'] = maven_home
            return env

    # Also check for Maven wrapper in project (will be set per-project)
    log_summary("MAVEN_ENV: Maven not found in standard locations, will try mvnw if available")
    return env


def get_maven_command(base_command: str, project_path: str = None) -> str:
    """
    Get the appropriate Maven command, preferring Maven wrapper if available.

    Args:
        base_command: The maven command (e.g., "mvn compile -B")
        project_path: Optional project path to check for mvnw

    Returns:
        Command string with appropriate Maven executable
    """
    # Check for Maven wrapper in project
    if project_path:
        mvnw_path = os.path.join(project_path, 'mvnw')
        if os.path.isfile(mvnw_path) and os.access(mvnw_path, os.X_OK):
            log_summary(f"MAVEN_CMD: Using Maven wrapper at {mvnw_path}")
            return base_command.replace('mvn ', './mvnw ', 1)

    # Check if we have Maven in our discovered path
    env = get_maven_env()
    maven_home = env.get('MAVEN_HOME')
    if maven_home:
        mvn_path = os.path.join(maven_home, 'bin', 'mvn')
        log_summary(f"MAVEN_CMD: Using Maven at {mvn_path}")
        return base_command.replace('mvn ', f'{mvn_path} ', 1)

    # Fall back to plain mvn (hope it's in PATH)
    return base_command


def is_command_safe(command: str) -> tuple[bool, str]:
    """Check if a command is safe to execute."""
    # Normalize command for checking
    cmd_lower = command.lower().strip()

    # Check for blocked patterns first (more specific)
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, cmd_lower):
            return False, f"Blocked pattern detected: {pattern}"

    # Extract the base command (first word)
    cmd_parts = cmd_lower.split()
    if not cmd_parts:
        return False, "Empty command"

    base_command = cmd_parts[0]

    # Remove path prefixes to get actual command
    base_command = base_command.split('/')[-1].split('\\')[-1]

    # Check if base command is explicitly blocked
    if base_command in BLOCKED_COMMANDS:
        return False, f"Blocked command: {base_command}"

    # Check if base command is in allowed list
    if base_command not in ALLOWED_COMMANDS:
        return False, f"Command not in allowed list: {base_command}. Allowed: {sorted(ALLOWED_COMMANDS)}"

    # Additional safety checks for specific commands
    if base_command == 'git':
        # Allow most git commands but block some dangerous ones
        git_blocked = ['clean -fd', 'reset --hard', 'push --force']
        for blocked in git_blocked:
            if blocked in cmd_lower:
                return False, f"Blocked git operation: {blocked}"

    elif base_command in ['cp', 'copy']:
        # Block copying to system directories
        system_dirs = ['/etc/', '/usr/', '/var/', '/sys/', '/proc/', 'c:\\windows\\', 'c:\\program files\\']
        for sys_dir in system_dirs:
            if sys_dir in cmd_lower:
                return False, f"Blocked copy to system directory: {sys_dir}"
    elif base_command == 'mkdir':
        # Block creating directories in system locations
        system_dirs = ['/etc/', '/usr/', '/var/', '/sys/', '/proc/', 'c:\\windows\\', 'c:\\program files\\']
        for sys_dir in system_dirs:
            if sys_dir in cmd_lower:
                return False, f"Blocked mkdir in system directory: {sys_dir}"

    return True, "Command is safe"


# NOTE: Error classification has been moved to src/orchestrator/error_handler.py
# The UnifiedErrorClassifier (unified_classifier singleton) is now the Single Source of Truth
# for all Maven error classification. See MavenErrorType enum for all supported error types.


def add_public_repositories_to_pom(pom_path: str) -> bool:
    """
    Add public repository fallbacks to pom.xml using XML parsing.
    Returns True if successful, False otherwise.
    """
    try:
        # Register namespace to preserve it in output
        ET.register_namespace('', 'http://maven.apache.org/POM/4.0.0')
        tree = ET.parse(pom_path)
        root = tree.getroot()

        ns = {'maven': 'http://maven.apache.org/POM/4.0.0'}

        # Check if repositories already exist
        repos = root.find('maven:repositories', ns)
        if repos is None:
            repos = ET.SubElement(root, '{http://maven.apache.org/POM/4.0.0}repositories')

        # Public repositories to add as fallbacks
        public_repos = [
            {"id": "maven-central-fallback", "url": "https://repo1.maven.org/maven2"},
            {"id": "spring-releases", "url": "https://repo.spring.io/release"},
            {"id": "spring-milestones", "url": "https://repo.spring.io/milestone"},
        ]

        # Check if these repos already exist
        existing_ids = set()
        for existing_repo in repos.findall('maven:repository', ns):
            repo_id = existing_repo.find('maven:id', ns)
            if repo_id is not None and repo_id.text:
                existing_ids.add(repo_id.text)

        # Add only non-existing repositories
        added_count = 0
        for repo_info in public_repos:
            if repo_info["id"] not in existing_ids:
                repo = ET.SubElement(repos, '{http://maven.apache.org/POM/4.0.0}repository')
                repo_id = ET.SubElement(repo, '{http://maven.apache.org/POM/4.0.0}id')
                repo_id.text = repo_info["id"]
                repo_url = ET.SubElement(repo, '{http://maven.apache.org/POM/4.0.0}url')
                repo_url.text = repo_info["url"]
                added_count += 1

        if added_count > 0:
            tree.write(pom_path, encoding='utf-8', xml_declaration=True)
            log_summary(f"POM_MODIFY: Added {added_count} public repositories to pom.xml")
            return True
        else:
            log_summary(f"POM_MODIFY: No new repositories added (already present)")
            return True

    except Exception as e:
        log_summary(f"POM_MODIFY_ERROR: Failed to modify pom.xml: {str(e)}")
        return False


def handle_authorization_error(command: str, cwd: str, timeout: int) -> tuple:
    """
    Handle 403/401 errors by adding public repository fallbacks.
    Returns: (success: bool, output: str)
    """
    log_summary(f"MVN_403_HANDLER: Authorization error detected, adding public repo fallbacks")

    # Get Maven environment
    maven_env = get_maven_env()

    pom_path = os.path.join(cwd, "pom.xml")
    backup_path = pom_path + ".backup-maven-fix"

    # Check if pom.xml exists
    if not os.path.exists(pom_path):
        log_summary(f"MVN_403_HANDLER: pom.xml not found at {pom_path}")
        return False, "pom.xml not found"

    # Backup original pom
    try:
        shutil.copy2(pom_path, backup_path)
        log_summary(f"MVN_403_HANDLER: Backed up pom.xml to {backup_path}")
    except Exception as e:
        log_summary(f"MVN_403_HANDLER: Failed to backup pom.xml: {str(e)}")
        return False, f"Failed to backup pom.xml: {str(e)}"

    try:
        # Add public repositories to pom.xml
        if not add_public_repositories_to_pom(pom_path):
            raise Exception("Failed to add repositories to pom.xml")

        log_summary(f"MVN_403_HANDLER: Added public repository fallbacks to pom.xml")

        # Try caching dependencies with public repos + SSL bypass
        cache_command = get_maven_command("mvn dependency:go-offline -B -Dmaven.wagon.http.ssl.insecure=true -Dmaven.wagon.http.ssl.allowall=true -Dmaven.wagon.http.ssl.ignore.validity.dates=true", cwd)
        log_summary(f"MVN_403_HANDLER: Attempting to cache dependencies with public repos")

        cache_result = subprocess.run(
            cache_command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,
            env=maven_env
        )

        if cache_result.returncode == 0:
            log_summary(f"MVN_403_HANDLER: Successfully cached dependencies with public repos")

            # Restore original pom
            shutil.move(backup_path, pom_path)
            log_summary(f"MVN_403_HANDLER: Restored original pom.xml")

            # Run command offline
            log_summary(f"MVN_403_HANDLER: Running command in offline mode")
            offline_result = subprocess.run(
                get_maven_command(command + " -o", cwd),
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                env=maven_env
            )

            if offline_result.returncode == 0:
                log_summary(f"MVN_403_HANDLER: Successfully completed in offline mode")
                output = f"Return code: 0 (offline mode after 403 fix)\n"
                output += f"Note: Resolved 403/authorization errors using public repositories\n"
                if offline_result.stdout:
                    output += f"STDOUT:\n{offline_result.stdout[-2000:]}\n"
                if offline_result.stderr:
                    output += f"STDERR:\n{offline_result.stderr[-2000:]}\n"
                return True, output
            else:
                log_summary(f"MVN_403_HANDLER: Offline mode failed after caching")
        else:
            log_summary(f"MVN_403_HANDLER: Failed to cache dependencies with public repos")

        # If we get here, something failed - restore original pom
        if os.path.exists(backup_path):
            shutil.move(backup_path, pom_path)
            log_summary(f"MVN_403_HANDLER: Restored original pom.xml after failure")

        return False, "Failed to resolve dependencies even with public repositories"

    except Exception as e:
        # Always restore on error
        if os.path.exists(backup_path):
            shutil.move(backup_path, pom_path)
            log_summary(f"MVN_403_HANDLER: Restored original pom.xml after exception")
        log_summary(f"MVN_403_HANDLER: Exception: {str(e)}")
        return False, f"Exception during 403 handling: {str(e)}"


def run_maven_with_retry(command: str, cwd: str, timeout: int = 300, max_retries: int = 2) -> str:
    """
    Run a Maven command with multi-layered error handling for SSL, authorization, and dependency issues.

    Strategy:
    1. Try to resolve dependencies first with retries
    2. Execute the main Maven command
    3. If failed, classify error type (SSL, 403/401, 404, etc.)
    4. Apply error-specific fixes:
       - SSL errors: Bypass SSL verification and cache deps, then run offline
       - 403/401 errors: Add public repositories, cache deps, run offline
       - 404/Missing: Report to agent for manual intervention
    5. Return detailed results with context about what happened

    This creates a progressive fallback system that handles most common Maven errors automatically.
    """
    log_summary(f"MVN_WITH_RETRY: {command} (max_retries={max_retries})")

    # Get Maven environment (finds Maven in non-standard locations)
    maven_env = get_maven_env()
    log_summary(f"MVN_WITH_RETRY: MAVEN_HOME={maven_env.get('MAVEN_HOME', 'not set')}")

    # Get proper Maven command (uses absolute path or mvnw if available)
    command = get_maven_command(command, cwd)
    log_summary(f"MVN_WITH_RETRY: Resolved command: {command[:80]}...")

    # Step 1: Attempt dependency resolution first
    dep_resolved = False
    ssl_error_detected = False

    dep_command = get_maven_command("mvn dependency:resolve -B", cwd)
    log_summary(f"MVN_DEPENDENCY: Attempting to resolve dependencies first")

    for attempt in range(max_retries):
        log_summary(f"MVN_DEPENDENCY: Attempt {attempt + 1}/{max_retries}")

        dep_result = subprocess.run(
            dep_command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,
            env=maven_env
        )

        combined_output = (dep_result.stdout or "") + (dep_result.stderr or "")

        if dep_result.returncode == 0:
            log_summary(f"MVN_DEPENDENCY: Successfully resolved dependencies")
            dep_resolved = True
            break

        # Use unified classifier for dependency resolution errors
        dep_error_type = unified_classifier.classify(combined_output, dep_result.returncode)
        log_summary(f"MVN_DEPENDENCY: Error type: {dep_error_type.value}")

        if dep_error_type == MavenErrorType.SSL_CERTIFICATE:
            log_summary(f"MVN_DEPENDENCY: SSL error detected in dependency resolution")
            ssl_error_detected = True

            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                log_summary(f"MVN_DEPENDENCY: Retrying after {sleep_time}s...")
                time.sleep(sleep_time)
        else:
            # Non-SSL error, don't retry
            log_summary(f"MVN_DEPENDENCY: Non-SSL error ({dep_error_type.value}), skipping retries")
            break

    # Step 2: Try the actual Maven command
    log_summary(f"MVN_COMMAND: Executing main command: {command}")

    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=True,
        env=maven_env
    )

    def tail(text, n=500):
        lines = text.splitlines()
        return "\n".join(lines[-n:])

    # Step 3: Classify error using unified classifier (Single Source of Truth)
    combined_output = (result.stdout or "") + (result.stderr or "")
    error_type = unified_classifier.classify(combined_output, result.returncode)
    log_summary(f"MVN_ERROR_CLASSIFICATION: {error_type.value}")

    # Step 4: If command failed, apply appropriate fix based on error type
    if result.returncode != 0:
        # Check if this is an infrastructure error that can be auto-fixed
        is_ssl_error = error_type == MavenErrorType.SSL_CERTIFICATE
        is_auth_error = error_type in [MavenErrorType.AUTH_401, MavenErrorType.AUTH_403]

        # Handle SSL errors (with or without auth errors)
        if is_ssl_error:
            log_summary(f"MVN_SSL_BYPASS: Attempting to cache dependencies with SSL verification disabled")

            # Use go-offline to download all dependencies with SSL checks disabled
            cache_command = get_maven_command("mvn dependency:go-offline -B -Dmaven.wagon.http.ssl.insecure=true -Dmaven.wagon.http.ssl.allowall=true -Dmaven.wagon.http.ssl.ignore.validity.dates=true", cwd)
            cache_result = subprocess.run(
                cache_command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                env=maven_env
            )

            cache_output = (cache_result.stdout or "") + (cache_result.stderr or "")

            if cache_result.returncode == 0:
                log_summary(f"MVN_SSL_BYPASS: Successfully cached dependencies to .m2 repository")

                # Step 5: Now try offline mode with cached dependencies
                log_summary(f"MVN_OFFLINE: Attempting offline mode with cached dependencies")
                offline_command = command + " -o"
                offline_result = subprocess.run(
                    offline_command,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=True,
                    env=maven_env
                )

                if offline_result.returncode == 0:
                    log_summary(f"MVN_OFFLINE: Successfully completed in offline mode")
                    output = f"Return code: 0 (offline mode after SSL bypass)\n"
                    output += f"Note: SSL errors bypassed by caching dependencies, then ran offline\n"
                    if offline_result.stdout:
                        output += f"STDOUT:\n{tail(offline_result.stdout)}\n"
                    if offline_result.stderr:
                        output += f"STDERR:\n{tail(offline_result.stderr)}\n"
                    return output
                else:
                    log_summary(f"MVN_OFFLINE: Offline mode failed even after caching")
            else:
                log_summary(f"MVN_SSL_BYPASS: Failed to cache dependencies")
                # Check if cache failure was due to auth error - if so, try 403 handler
                cache_error_type = unified_classifier.classify(cache_output, cache_result.returncode)
                if cache_error_type in [MavenErrorType.AUTH_401, MavenErrorType.AUTH_403]:
                    log_summary(f"MVN_SSL_BYPASS: Detected {cache_error_type.value} during caching, trying auth handler")
                    success, handler_output = handle_authorization_error(command, cwd, timeout)
                    if success:
                        return handler_output

        # Handle 401/403 authorization errors directly
        elif is_auth_error:
            log_summary(f"MVN_AUTH_ERROR: Detected {error_type.value}, attempting to add public repositories")
            success, handler_output = handle_authorization_error(command, cwd, timeout)
            if success:
                return handler_output

    # Step 6: Return the result (success or final failure)
    output = f"Return code: {result.returncode}\n"

    if result.returncode != 0:
        # Use unified classifier categories for warning messages
        output += f"ERROR_TYPE: {error_type.value}\n"

        # Infrastructure errors (auto-fix attempted)
        if error_type == MavenErrorType.SSL_CERTIFICATE:
            output += "WARNING: SSL certificate errors detected during Maven operations\n"
            output += "NOTE: Attempted SSL bypass and offline mode but failed\n"
        elif error_type in [MavenErrorType.AUTH_401, MavenErrorType.AUTH_403]:
            output += "WARNING: Authorization errors (401/403) detected during Maven operations\n"
            output += "NOTE: Attempted adding public repositories but failed\n"
        elif error_type == MavenErrorType.NETWORK_TIMEOUT:
            output += "WARNING: Network timeout during Maven operations\n"

        # Dependency errors
        elif error_type == MavenErrorType.ARTIFACT_NOT_FOUND:
            output += "WARNING: Artifacts not found\n"
            output += "NOTE: Some dependencies may not be available in configured repositories\n"
        elif error_type == MavenErrorType.DEPENDENCY_CONFLICT:
            output += "WARNING: Dependency version conflict detected\n"
        elif error_type == MavenErrorType.VERSION_MISMATCH:
            output += "WARNING: Version mismatch detected\n"

        # Build errors (need error_expert)
        elif error_type == MavenErrorType.COMPILATION_ERROR:
            output += "WARNING: Compilation errors detected\n"
            output += "NOTE: Java source code needs fixes - route to error_expert\n"
        elif error_type == MavenErrorType.TEST_FAILURE:
            output += "WARNING: Test failures detected\n"
            output += "NOTE: Tests need investigation - route to error_expert\n"
        elif error_type == MavenErrorType.POM_SYNTAX_ERROR:
            output += "WARNING: POM syntax error detected\n"
            output += "NOTE: pom.xml needs fixes - route to error_expert\n"

        # Migration-specific errors
        elif error_type == MavenErrorType.JAVA_VERSION_ERROR:
            output += "WARNING: Java version configuration error\n"
        elif error_type == MavenErrorType.SPRING_MIGRATION_ERROR:
            output += "WARNING: Spring migration error detected\n"
        elif error_type == MavenErrorType.JAKARTA_MIGRATION_ERROR:
            output += "WARNING: Jakarta namespace migration error detected\n"

        # Unknown/fallback
        else:
            output += "WARNING: Maven command failed\n"

    if result.stdout:
        output += f"STDOUT:\n{tail(result.stdout)}\n"
    if result.stderr:
        output += f"STDERR:\n{tail(result.stderr)}\n"

    if result.returncode != 0:
        log_summary(f"MVN_COMMAND: Failed with return code {result.returncode}, error_type={error_type.value}")
    else:
        log_summary(f"MVN_COMMAND: Success")

    return output


@tool
def run_command(command: str, cwd: str = ".", timeout: int = 300) -> str:
    """Run a shell command and return the result (last 500 lines or less)."""
    log_summary(f"COMMAND: {command} (cwd: {cwd})")

    is_safe, reason = is_command_safe(command)
    if not is_safe:
        log_summary(f"COMMAND BLOCKED: {reason}")
        return f"Command blocked for safety reasons: {reason}"

    try:
        # Get proper environment (especially for Maven commands)
        env = os.environ.copy()
        actual_command = command

        # If this is a Maven command, use proper Maven path
        if command.strip().startswith('mvn ') or command.strip() == 'mvn':
            env = get_maven_env()
            actual_command = get_maven_command(command, cwd)
            log_summary(f"COMMAND: Resolved Maven command: {actual_command[:80]}...")

        # Use shell=True for commands with complex quoting
        result = subprocess.run(
            actual_command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,
            env=env
        )

        def tail(text, n=500):
            lines = text.splitlines()
            return "\n".join(lines[-n:])

        output = f"Return code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{tail(result.stdout)}\n"
        if result.stderr:
            output += f"STDERR:\n{tail(result.stderr)}\n"

        if result.returncode == 1:
            error_info = ""
            if result.stderr:
                error_info += f"STDERR: {tail(result.stderr)}"
            if result.stdout:
                if error_info:
                    error_info += " | "
                error_info += f"STDOUT: {tail(result.stdout)}"
            if not error_info:
                error_info = "No error output"
            log_summary(f"COMMAND Return code {result.returncode} - {error_info}")
        else:
            log_summary(f"COMMAND SUCCESS")

        return output
    except subprocess.TimeoutExpired:
        log_summary(f"COMMAND TIMEOUT: {command} exceeded {timeout}s")
        return f"Command timed out after {timeout} seconds"
    except Exception as e:
        log_summary(f"COMMAND EXCEPTION: {str(e)}")
        return f"Error: {str(e)}"


@tool
def mvn_compile(project_path: str) -> str:
    """
    Run Maven compile in the specified project directory.
    Automatically handles SSL errors with dependency resolution retries and offline fallback.
    """
    try:
        return run_maven_with_retry("mvn compile -B", project_path, timeout=600, max_retries=2)
    except subprocess.TimeoutExpired:
        log_summary(f"MVN_COMPILE TIMEOUT in {project_path}")
        return f"Maven compile timed out after 600 seconds"
    except Exception as e:
        log_summary(f"MVN_COMPILE EXCEPTION: {str(e)}")
        return f"Maven compile error: {str(e)}"


@tool
def mvn_test(project_path: str) -> str:
    """
    Run Maven test in the specified project directory.
    Automatically handles SSL errors with dependency resolution retries and offline fallback.
    """
    try:
        return run_maven_with_retry("mvn test -B", project_path, timeout=600, max_retries=2)
    except subprocess.TimeoutExpired:
        log_summary(f"MVN_TEST TIMEOUT in {project_path}")
        return f"Maven test timed out after 600 seconds"
    except Exception as e:
        log_summary(f"MVN_TEST EXCEPTION: {str(e)}")
        return f"Maven test error: {str(e)}"


@tool
def mvn_rewrite_run(project_path: str) -> str:
    """[EXECUTES RECIPES] Run ALL configured OpenRewrite recipes using 'mvn rewrite:run'.

    This tool ACTUALLY EXECUTES the recipes configured in pom.xml and applies code changes.
    Use this after configuring recipes with configure_openrewrite_recipes() or add_openrewrite_plugin().

    Returns the Maven output showing which files were modified.
    """
    return run_command.invoke({"command": "mvn rewrite:run", "cwd": project_path})


@tool
def git_status(project_path: str) -> str:
    """Get git status for the specified project directory."""
    return run_command.invoke({"command": "git status --porcelain", "cwd": project_path})


@tool
def git_add_all(project_path: str) -> str:
    """Git add all changes in the specified project directory."""
    return run_command.invoke({"command": "git add .", "cwd": project_path})


@tool
def git_commit(project_path: str, message: str) -> str:
    """Git commit with message in the specified project directory."""
    return run_command.invoke({"command": f'git commit -m "{message}"', "cwd": project_path})


@tool
def mvn_rewrite_run_recipe(project_path: str, recipe_name: str) -> str:
    """[EXECUTES SPECIFIC RECIPE] Run a single OpenRewrite recipe using 'mvn rewrite:run -Drewrite.activeRecipes=...'.

    This tool ACTUALLY EXECUTES a specific recipe by name and applies code changes.
    Use this to run one recipe at a time instead of all configured recipes.

    Example recipe names: 'org.openrewrite.java.migrate.Java8toJava11', 'org.openrewrite.java.spring.boot3.UpgradeSpringBoot_3_0'
    """
    return run_command.invoke({"command": f"mvn rewrite:run -Drewrite.activeRecipes={recipe_name}", "cwd": project_path})


@tool
def mvn_rewrite_dry_run(project_path: str) -> str:
    """[DRY RUN - NO CHANGES] Preview what changes OpenRewrite would make without actually applying them.

    This runs 'mvn rewrite:dryRun' to see what files would be changed without modifying anything.
    Use this to preview changes before running mvn_rewrite_run().
    """
    return run_command.invoke({"command": "mvn rewrite:dryRun", "cwd": project_path})


# Collect all command tools
command_tools = [run_command, mvn_compile, mvn_test, mvn_rewrite_run, mvn_rewrite_run_recipe, mvn_rewrite_dry_run, git_status, git_add_all, git_commit]