"""
Tool Registry for Migration Orchestrator

This module handles:
- Tool set definitions for each agent (analysis, execution, error, supervisor)
- Tool wrapping for tracking, protection, and deduplication
- File access controls

Each agent type has a specific set of tools tailored to its role:
- Analysis: Read-only + state file creation (TODO.md, CURRENT_STATE.md, analysis.md)
- Execution: Modify files, run OpenRewrite, git operations
- Error: Diagnostic only (read files, run builds, check git)
- Supervisor: Read-only + state management
"""

import os
from datetime import datetime, timedelta
from typing import List, Set, Callable, Any

from langchain_core.tools import StructuredTool

from src.utils.logging_config import log_agent, log_summary
from .constants import (
    TRACKED_TOOLS,
    COMMIT_TOOLS,
    PROTECTED_STATE_FILES,
    ERROR_AGENT_BLOCKED_FILES
)


# =============================================================================
# TOOL SET DEFINITIONS
# =============================================================================

# Analysis agent tools - for analyzing project and creating migration plan
ANALYSIS_TOOL_NAMES: Set[str] = {
    # Read tools for inspection
    'read_file',
    'read_pom',
    'find_all_poms',
    'list_java_files',
    'search_files',
    'file_exists',
    'get_java_version',
    'list_dependencies',
    'get_available_recipes',
    'suggest_recipes_for_java_version',
    'web_search_tool',
    'call_openrewrite_agent',
    'get_status',
    'get_log',
    'list_branches',

    # State management
    'write_file',
    'check_migration_state',

    # Completion marking
    'mark_analysis_complete',
}

# Execution agent tools - for executing migration tasks
EXECUTION_TOOL_NAMES: Set[str] = {
    # File operations (read for context, write for changes)
    'read_file', 'write_file', 'find_replace', 'file_exists',
    # Read pom for targeted checks before modification
    'read_pom',
    # Maven operations (execution - these MODIFY the project)
    'configure_openrewrite_recipes', 'update_java_version',
    'add_openrewrite_plugin', 'update_spring_boot_version',
    # OpenRewrite execution
    'mvn_rewrite_run', 'mvn_rewrite_run_recipe', 'mvn_rewrite_dry_run',
    # Git operations (write)
    'create_branch', 'checkout_branch', 'commit_changes',
    'git_add_all', 'git_commit', 'git_status', 'tag_checkpoint',
    # Build validation
    'mvn_compile', 'mvn_test',
    # Command execution (for edge cases)
    'run_command',
    # Completion marking
    'mark_execution_complete',
}

# Error agent tools - for diagnosing and fixing errors
ERROR_TOOL_NAMES: Set[str] = {
    # Read tools - will be wrapped to block state files
    'read_file', 'file_exists',
    # Diagnostic tools
    'mvn_compile', 'mvn_test',
    # Git operations (diagnostic)
    'git_status', 'get_log', 'list_branches'
    # REMOVED: run_command (too dangerous - agent did manual sed/find edits)
    # REMOVED: check_migration_state (confuses agent about current phase)
}

# Supervisor agent tools - read-only for oversight
SUPERVISOR_TOOL_NAMES: Set[str] = {
    'read_file',
    'file_exists',
    'list_java_files',
    'read_pom',
    'check_migration_state',
}

# Files that analysis agent is allowed to write
ANALYSIS_ALLOWED_FILES = ['TODO.md', 'CURRENT_STATE.md', 'analysis.md']


# =============================================================================
# TOOL WRAPPERS
# =============================================================================

class ToolWrapper:
    """
    Wraps tools with tracking, protection, and deduplication logic.

    This class provides methods to wrap tools for different agent types,
    adding appropriate restrictions and tracking.
    """

    def __init__(self, project_path: str = None, action_logger=None, task_manager=None):
        """
        Args:
            project_path: Path to project directory
            action_logger: ActionLogger instance for logging tool calls
            task_manager: TaskManager instance for auto-sync after commits
        """
        self.project_path = project_path
        self.action_logger = action_logger
        self.task_manager = task_manager
        self.recent_tracked_tools = []
        self.verification_window_seconds = 60

        # Session-scoped tracking for loop prevention
        self.session_commits = []  # Track commits made THIS session
        self.task_attempts = {}  # task_description -> attempt count
        log_agent("[TOOL_WRAPPER] Session tracking initialized")

    def set_project_path(self, project_path: str):
        """Update the project path"""
        self.project_path = project_path

    def wrap_analysis_tool(self, tool) -> StructuredTool:
        """
        Wrap analysis agent tools with restrictions.

        Analysis agent should ONLY write to state files:
        - TODO.md (migration plan)
        - CURRENT_STATE.md (project status)
        - analysis.md (analysis notes)

        Should NOT modify project files (pom.xml, .java files, etc.)
        """
        tool_name = tool.name

        # Only wrap write_file - other tools are safe
        if tool_name != 'write_file':
            return tool

        original_func = tool.func

        def restricted_write_file(*args, **kwargs):
            file_path = kwargs.get('file_path', args[0] if args else '')

            # Check if writing to allowed file
            is_allowed = any(file_path.endswith(allowed_file) for allowed_file in ANALYSIS_ALLOWED_FILES)

            if not is_allowed:
                log_agent(f"[ANALYSIS_BLOCK] Analysis agent tried to write to '{file_path}' - BLOCKED")
                log_agent(f"[ANALYSIS_BLOCK] Analysis agent can only write to: {', '.join(ANALYSIS_ALLOWED_FILES)}")
                return f"""ERROR: Analysis agent cannot modify project files.

FORBIDDEN: File modification blocked

You attempted to write to: {file_path}

Analysis agent can ONLY create these state files:
- TODO.md (migration plan with sequential tasks)
- CURRENT_STATE.md (project status and dependencies)
- analysis.md (analysis notes and recommendations)

You CANNOT modify project files such as:
- pom.xml (execution agent will handle this)
- .java files (execution agent will handle this)
- Any other project files

Your role is to ANALYZE and PLAN, not to execute changes.
The execution agent will handle all project modifications."""

            # Log allowed write
            log_agent(f"[ANALYSIS_WRITE] Analysis agent writing to allowed file: {file_path}")
            return original_func(*args, **kwargs)

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            func=restricted_write_file,
            args_schema=tool.args_schema
        )

    def wrap_error_read_file(self, tool) -> StructuredTool:
        """
        Wrap read_file for error agent to block state files.

        Error agent should ONLY see:
        - Error messages (injected in prompt)
        - Project code files (pom.xml, .java files)
        - Error history

        Error agent should NOT see:
        - TODO.md (it's not executing tasks)
        - CURRENT_STATE.md (irrelevant to error fixing)
        - VISIBLE_TASKS.md (irrelevant to error fixing)
        - COMPLETED_ACTIONS.md (irrelevant to error fixing)
        """
        original_func = tool.func

        def error_read_file_blocked(*args, **kwargs):
            file_path = kwargs.get('file_path', args[0] if args else '')

            # Block state files
            for blocked in ERROR_AGENT_BLOCKED_FILES:
                if blocked in file_path:
                    log_agent(f"[ERROR_BLOCK] Error agent tried to read {blocked} - BLOCKED")
                    return f"ERROR: {blocked} is not accessible to error agent. Focus on fixing the build error only."

            # Allow reading ERROR_HISTORY.md and project files
            return original_func(*args, **kwargs)

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            func=error_read_file_blocked,
            args_schema=tool.args_schema
        )

    def wrap_execution_tool(self, tool, on_commit_success: Callable = None) -> StructuredTool:
        """
        Wrap execution tools with tracking and protection.

        Features:
        - File protection (COMPLETED_ACTIONS.md, TODO.md, VISIBLE_TASKS.md)
        - Deduplication (prevent repeating completed actions)
        - Action logging to COMPLETED_ACTIONS.md
        - Auto-sync after commits (mark task complete, update VISIBLE_TASKS.md)

        Args:
            tool: The tool to wrap
            on_commit_success: Callback to call after successful commit
        """
        original_func = tool.func
        tool_name = tool.name

        # Special handling for write_file and find_replace - protect system files
        if tool_name in ['write_file', 'find_replace']:
            return self._wrap_file_modify_tool(tool)

        # Special handling for read_file - redirect TODO.md to VISIBLE_TASKS.md
        if tool_name == 'read_file':
            return self._wrap_read_file_with_redirect(tool)

        # Special handling for run_command - block TODO.md access
        if tool_name == 'run_command':
            return self._wrap_run_command_with_todo_block(tool)

        # For non-tracked tools, return as-is
        if tool_name not in TRACKED_TOOLS:
            return tool

        # Wrap tracked tools with tracking and deduplication
        return self._wrap_tracked_tool(tool, on_commit_success)

    def _wrap_file_modify_tool(self, tool) -> StructuredTool:
        """Wrap file modification tools with protection"""
        original_func = tool.func
        tool_name = tool.name

        def file_modify_protected(*args, **kwargs):
            file_path = kwargs.get('file_path', '')

            # PROTECTION 1: Block writes to COMPLETED_ACTIONS.md (system-managed)
            if 'COMPLETED_ACTIONS.md' in file_path:
                log_agent(f"[PROTECT] Blocked {tool_name} to COMPLETED_ACTIONS.md - file is system-managed")
                return "COMPLETED_ACTIONS.md is a system-managed, append-only file for tracking completed actions. You cannot modify it directly."

            # PROTECTION 2: Block TODO.md access
            if 'TODO.md' in file_path and 'VISIBLE_TASKS.md' not in file_path:
                log_agent(f"[PROTECT] BLOCKED: {tool_name} to TODO.md - execution agent cannot access this file")
                return (
                    "BLOCKED: TODO.md is a system-managed file. You cannot access it directly.\n\n"
                    "YOUR WORKFLOW:\n"
                    "1. Read VISIBLE_TASKS.md to see your current task\n"
                    "2. Execute that task\n"
                    "3. Commit with commit_changes or git_commit\n"
                    "4. System AUTOMATICALLY marks task complete in TODO.md\n"
                    "5. System AUTOMATICALLY updates VISIBLE_TASKS.md with next task\n\n"
                    "You do NOT need to mark tasks - the system handles this deterministically."
                )

            # PROTECTION 3: Block VISIBLE_TASKS.md modification
            if 'VISIBLE_TASKS.md' in file_path:
                log_agent(f"[PROTECT] BLOCKED: {tool_name} to VISIBLE_TASKS.md - agent cannot modify task list")
                return (
                    "BLOCKED: VISIBLE_TASKS.md is READ-ONLY for you.\n\n"
                    "You CANNOT mark tasks complete or modify this file.\n\n"
                    "THE CORRECT WORKFLOW:\n"
                    "1. Read VISIBLE_TASKS.md to see your CURRENT task\n"
                    "2. Execute ONLY that task\n"
                    "3. Commit with commit_changes or git_commit\n"
                    "4. System AUTOMATICALLY updates VISIBLE_TASKS.md with next task\n\n"
                    "DO NOT try to mark tasks yourself."
                )

            # PROTECTION 4: Prevent analysis.md/CURRENT_STATE.md overwrites
            if tool_name == 'write_file' and (
                'analysis.md' in file_path or 'CURRENT_STATE.md' in file_path
            ):
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            existing_content = f.read()
                        if len(existing_content) > 50:
                            log_agent(f"[PROTECT] Blocked write_file to existing {os.path.basename(file_path)}")
                            return (
                                f"{os.path.basename(file_path)} already exists. This file was created during initial analysis. "
                                f"Do NOT recreate or overwrite it."
                            )
                    except Exception:
                        pass

            # Allow all other file operations
            return original_func(*args, **kwargs)

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            func=file_modify_protected,
            args_schema=tool.args_schema
        )

    def _wrap_read_file_with_redirect(self, tool) -> StructuredTool:
        """Wrap read_file to redirect TODO.md to VISIBLE_TASKS.md"""
        original_func = tool.func

        def read_file_with_task_redirect(*args, **kwargs):
            file_path = kwargs.get('file_path', args[0] if args else '')

            # REDIRECT: If reading TODO.md, read VISIBLE_TASKS.md instead
            if 'TODO.md' in file_path:
                log_agent(f"[READ_REDIRECT] Agent tried to read TODO.md - redirecting to VISIBLE_TASKS.md")
                visible_tasks_path = file_path.replace('TODO.md', 'VISIBLE_TASKS.md')
                if 'file_path' in kwargs:
                    kwargs['file_path'] = visible_tasks_path
                elif args:
                    args = (visible_tasks_path,) + args[1:]
                return original_func(*args, **kwargs)

            return original_func(*args, **kwargs)

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            func=read_file_with_task_redirect,
            args_schema=tool.args_schema
        )

    def _wrap_run_command_with_todo_block(self, tool) -> StructuredTool:
        """Wrap run_command to block TODO.md access"""
        original_func = tool.func

        def run_command_with_todo_block(*args, **kwargs):
            command = kwargs.get('command', args[0] if args else '')

            # BLOCK: Prevent TODO.md access via shell commands
            if 'TODO.md' in command:
                log_agent(f"[CMD_BLOCK] Execution agent tried to access TODO.md via run_command - BLOCKED")
                return """ERROR: TODO.md is not accessible to execution agent.

Use VISIBLE_TASKS.md instead - it contains your current task.

Your task tracking works like this:
1. Read VISIBLE_TASKS.md to see your CURRENT TASK
2. Complete that task
3. Commit your changes with git_commit
4. The system automatically marks the task complete
5. VISIBLE_TASKS.md is regenerated with the next task"""

            return original_func(*args, **kwargs)

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            func=run_command_with_todo_block,
            args_schema=tool.args_schema
        )

    def _wrap_tracked_tool(self, tool, on_commit_success: Callable = None) -> StructuredTool:
        """Wrap tracked tools with deduplication and logging"""
        original_func = tool.func
        tool_name = tool.name

        def wrapped_func(*args, **kwargs):
            # Check for deduplication
            if self.project_path:
                skip_msg = self._check_deduplication(tool_name, args, kwargs)
                if skip_msg:
                    return skip_msg

            # Smart git_commit: Auto-stage if nothing staged
            if tool_name == 'git_commit':
                self._auto_stage_if_needed(kwargs)

            # Execute the tool
            start_time = datetime.now()
            try:
                result = original_func(*args, **kwargs)
                error = None
            except Exception as e:
                result = None
                error = str(e)
                log_agent(f"[TOOL_ERROR] {tool_name} raised exception: {e}")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Determine success
            is_success = self._determine_success(tool_name, result)
            result_str = str(result) if result is not None else ""

            # Check for "nothing to commit" scenario
            is_nothing_to_commit = self._is_nothing_to_commit(tool_name, result)

            # Log action
            if self.action_logger:
                self.action_logger.log_action(
                    tool_name=tool_name,
                    success=is_success,
                    duration=duration,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    error=error
                )

            # Track for verification
            self.recent_tracked_tools.append({
                'tool_name': tool_name,
                'timestamp': datetime.now(),
                'success': is_success
            })

            # Handle commit scenarios
            if tool_name in COMMIT_TOOLS:
                if is_success:
                    # Successful commit - record and trigger auto-sync
                    log_agent(f"[AUTO_SYNC] ✅ Successful commit detected ({tool_name}) - triggering task sync")

                    # Record session commit
                    commit_hash = ""
                    if 'Committed changes:' in result_str:
                        # Extract hash from "Committed changes: {hash} - {message}"
                        parts = result_str.split('Committed changes:')[1].strip().split(' ')
                        if parts:
                            commit_hash = parts[0]
                    current_task = self._get_current_task_from_visible()
                    if commit_hash and current_task:
                        self._record_session_commit(commit_hash, current_task)

                    # LOG TASK COMPLETION to COMPLETED_ACTIONS.md
                    if current_task and self.action_logger:
                        self.action_logger.log_task_completion(current_task, commit_hash)
                        log_agent(f"[TASK_LOG] Logged task completion: {current_task[:60]}...")

                    if on_commit_success:
                        on_commit_success()

                elif is_nothing_to_commit:
                    # "Nothing to commit" - run verification flow
                    log_agent(f"[AUTO_SYNC] ⚠ 'Nothing to commit' detected - running task verification")
                    result_str = self._handle_nothing_to_commit(result_str)
                    return result_str

            return result if result is not None else f"Error: {error}"

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            func=wrapped_func,
            args_schema=tool.args_schema
        )

    def _check_deduplication(self, tool_name: str, args, kwargs) -> str:
        """Check if action was already completed. Returns skip message or None."""
        completed_actions_path = os.path.join(self.project_path, "COMPLETED_ACTIONS.md")

        try:
            if not os.path.exists(completed_actions_path):
                return None

            with open(completed_actions_path, 'r') as f:
                completed_content = f.read()

            action_pattern = TRACKED_TOOLS.get(tool_name, '')

            # Special handling for create_branch
            if tool_name == 'create_branch':
                branch_name = kwargs.get('branch_name', '')
                if branch_name and (
                    f"branch: {branch_name}" in completed_content or
                    f"branch '{branch_name}'" in completed_content
                ):
                    log_agent(f"[DEDUPE] Branch '{branch_name}' already created - skipping")
                    return f"Branch '{branch_name}' was already created. Skipping duplicate creation."

            # Special handling for git_commit
            elif tool_name == 'git_commit':
                return self._check_commit_deduplication(kwargs, completed_content)

            # Generic deduplication for other tools
            elif action_pattern and action_pattern in completed_content:
                log_agent(f"[DEDUPE] Action '{action_pattern}' already completed - skipping")
                if tool_name == 'add_openrewrite_plugin':
                    return "OpenRewrite plugin is already configured. Skipping duplicate addition."
                elif tool_name == 'configure_openrewrite_recipes':
                    return "OpenRewrite recipes are already configured. Skipping."
                elif tool_name == 'update_java_version':
                    return "Java version already updated. Skipping."
                elif tool_name in ['mvn_rewrite_run', 'mvn_rewrite_run_recipe']:
                    return "OpenRewrite migration already executed. Skipping."
                else:
                    return "Action already completed. Skipping duplicate."

        except Exception as e:
            log_agent(f"[DEDUPE] Could not check for deduplication: {str(e)}")

        return None

    def _check_commit_deduplication(self, kwargs, completed_content: str) -> str:
        """Check if commit was already made or failed too many times"""
        commit_msg = kwargs.get('message', '')
        if not commit_msg:
            return None

        # Check git history
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'log', '--oneline', '-10', '--format=%s'],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                recent_commits = result.stdout.strip().split('\n')
                if commit_msg in recent_commits:
                    log_agent(f"[DEDUPE] Commit '{commit_msg[:60]}...' already in git history")
                    return f"Commit with message '{commit_msg}' was already made."
        except Exception:
            pass

        # Check for recent failed attempts
        import re
        recent_failed = []
        for line in completed_content.split('\n'):
            if 'git_commit | FAILED' in line and commit_msg[:30] in line:
                match = re.search(r'\[(\d+)\] (\d{2}:\d{2}:\d{2})', line)
                if match:
                    recent_failed.append(match.group(2))

        if len(recent_failed) >= 2:
            log_agent(f"[DEDUPE] Blocking git_commit - failed {len(recent_failed)} times recently")
            return (
                f"git_commit with message '{commit_msg}' failed {len(recent_failed)} times recently. "
                f"Fix the underlying issue before retrying."
            )

        return None

    def _auto_stage_if_needed(self, kwargs):
        """Auto-stage files if nothing is staged before commit"""
        project_path = kwargs.get('project_path', self.project_path)
        if not project_path:
            return

        try:
            import subprocess
            # Check if there are staged changes
            status_result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=5
            )

            if status_result.returncode == 0:
                staged_files = status_result.stdout.strip()
                if not staged_files:
                    # Check if there are changes to stage
                    changes_result = subprocess.run(
                        ['git', 'status', '--porcelain'],
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )

                    if changes_result.returncode == 0 and changes_result.stdout.strip():
                        log_agent(f"[SMART_COMMIT] No staged changes, auto-staging files")
                        subprocess.run(
                            ['git', 'add', '-A'],
                            cwd=project_path,
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
        except Exception as e:
            log_agent(f"[SMART_COMMIT] Could not check/stage changes: {e}")

    def _determine_success(self, tool_name: str, result) -> bool:
        """Determine if tool execution was successful"""
        result_str = str(result) if result is not None else ""

        if tool_name == 'git_commit':
            # git_commit uses run_command, returns "Return code: 0" on success
            return 'Return code: 0' in result_str
        elif tool_name == 'commit_changes':
            # commit_changes (from git_operations.py) returns "Committed changes: {hash} - {message}"
            # on success, or "Error committing changes: ..." on failure
            return 'Committed changes:' in result_str
        elif tool_name == 'create_branch':
            return 'Created branch' in result_str or 'Checked out branch' in result_str
        elif tool_name == 'add_openrewrite_plugin':
            return 'added' in result_str.lower() or 'success' in result_str.lower()
        elif tool_name == 'configure_openrewrite_recipes':
            return 'Configured' in result_str or 'recipes' in result_str.lower()
        elif tool_name in ['mvn_rewrite_run', 'mvn_rewrite_run_recipe', 'mvn_rewrite_dry_run']:
            return 'Return code: 0' in result_str and 'BUILD SUCCESS' in result_str
        else:
            return (
                'success' in result_str.lower() or
                'completed' in result_str.lower() or
                'Return code: 0' in result_str
            )

    def _is_nothing_to_commit(self, tool_name: str, result) -> bool:
        """Check if commit tool returned 'nothing to commit'"""
        if tool_name not in COMMIT_TOOLS:
            return False
        result_str = str(result).lower() if result is not None else ""
        return 'nothing to commit' in result_str or 'working tree clean' in result_str

    # =========================================================================
    # TASK VERIFICATION FUNCTIONS
    # =========================================================================

    def _increment_task_attempt(self, task_description: str) -> int:
        """Increment and return the attempt count for a task"""
        if task_description not in self.task_attempts:
            self.task_attempts[task_description] = 0
        self.task_attempts[task_description] += 1
        count = self.task_attempts[task_description]
        log_agent(f"[LOOP_TRACK] Task attempt #{count}: {task_description[:60]}...")
        return count

    def _get_task_attempt_count(self, task_description: str) -> int:
        """Get the current attempt count for a task"""
        return self.task_attempts.get(task_description, 0)

    def _record_session_commit(self, commit_hash: str, task_description: str):
        """Record a commit made in this session"""
        self.session_commits.append({
            'hash': commit_hash,
            'task': task_description,
            'timestamp': datetime.now()
        })
        log_agent(f"[SESSION] Recorded commit {commit_hash[:8]} for task: {task_description[:50]}...")

    def _verify_task_complete(self, task_description: str) -> tuple:
        """
        Verify if a task is actually complete by checking file state.

        Returns:
            (is_verified: bool, reason: str)
        """
        import subprocess
        import re

        if not self.project_path:
            return False, "No project path set"

        task_lower = task_description.lower()

        # Match task to verification function using patterns
        verifications = [
            (r'add.*openrewrite|openrewrite.*plugin|configure.*recipe', self._verify_openrewrite_setup),
            (r'java\s*21|java\s*version.*21|UpgradeToJava21', self._verify_java_21),
            (r'spring\s*boot\s*3|UpgradeSpringBoot', self._verify_spring_boot_3),
            (r'jakarta|javax.*to.*jakarta|JavaxMigration', self._verify_jakarta_migration),
            (r'junit.*4.*to.*5|junit.*5|junit.*migration|JUnit4to5', self._verify_junit5_migration),
            (r'verify.*compil|compilation', self._verify_compilation),
            (r'verify.*test|run.*test', self._verify_tests),
            (r'verify.*import|import.*migrat', self._verify_imports_generic),
            (r'create.*branch|branch.*creat', self._verify_branch_exists),
            (r'baseline|establish.*build', self._verify_compilation),
        ]

        for pattern, verify_func in verifications:
            if re.search(pattern, task_lower, re.IGNORECASE):
                try:
                    return verify_func()
                except Exception as e:
                    log_agent(f"[VERIFY] Error in {verify_func.__name__}: {e}")
                    return False, f"Verification error: {e}"

        # Default: can't verify, assume not complete
        return False, f"No verification available for task type"

    def _verify_openrewrite_setup(self) -> tuple:
        """Verify OpenRewrite plugin is configured in pom.xml"""
        import os
        pom_path = os.path.join(self.project_path, "pom.xml")
        if not os.path.exists(pom_path):
            return False, "pom.xml not found"

        with open(pom_path, 'r') as f:
            content = f.read()

        if 'openrewrite-maven-plugin' in content or 'rewrite-maven-plugin' in content:
            return True, "OpenRewrite plugin found in pom.xml"
        return False, "OpenRewrite plugin not found in pom.xml"

    def _verify_java_21(self) -> tuple:
        """Verify Java version is set to 21"""
        import os
        pom_path = os.path.join(self.project_path, "pom.xml")
        if not os.path.exists(pom_path):
            return False, "pom.xml not found"

        with open(pom_path, 'r') as f:
            content = f.read()

        # Check for java.version property
        if '<java.version>21</java.version>' in content:
            return True, "Java version 21 configured via java.version property"
        if '<maven.compiler.source>21</maven.compiler.source>' in content:
            return True, "Java version 21 configured via maven.compiler.source"
        if '<release>21</release>' in content:
            return True, "Java version 21 configured via release tag"

        return False, "Java version 21 not found in pom.xml"

    def _verify_spring_boot_3(self) -> tuple:
        """Verify Spring Boot 3.x is configured"""
        import os
        pom_path = os.path.join(self.project_path, "pom.xml")
        if not os.path.exists(pom_path):
            return False, "pom.xml not found"

        with open(pom_path, 'r') as f:
            content = f.read()

        import re
        # Check for spring-boot version 3.x
        match = re.search(r'spring-boot.*?(\d+\.\d+\.\d+)', content, re.IGNORECASE | re.DOTALL)
        if match:
            version = match.group(1)
            if version.startswith('3.'):
                return True, f"Spring Boot {version} configured"

        return False, "Spring Boot 3.x not found in pom.xml"

    def _verify_jakarta_migration(self) -> tuple:
        """Verify javax imports have been migrated to jakarta"""
        import subprocess
        import os

        # Search for remaining javax.persistence or javax.servlet imports
        result = subprocess.run(
            ['grep', '-r', '-l', 'javax.persistence\|javax.servlet', '--include=*.java', '.'],
            cwd=self.project_path,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0 or not result.stdout.strip():
            return True, "No javax.persistence/servlet imports found"

        files_with_javax = result.stdout.strip().split('\n')
        return False, f"Found javax imports in {len(files_with_javax)} files"

    def _verify_junit5_migration(self) -> tuple:
        """Verify JUnit 5 migration is complete"""
        import subprocess
        import os

        # Check for JUnit 5 imports (positive check)
        junit5_result = subprocess.run(
            ['grep', '-r', 'org.junit.jupiter', '--include=*.java', '.'],
            cwd=self.project_path,
            capture_output=True,
            text=True,
            timeout=30
        )

        has_junit5 = junit5_result.returncode == 0 and junit5_result.stdout.strip()

        # Check for old JUnit 4 patterns (negative check)
        junit4_result = subprocess.run(
            ['grep', '-r', '-l', '@RunWith\|org.junit.Test\|org.junit.Before', '--include=*.java', '.'],
            cwd=self.project_path,
            capture_output=True,
            text=True,
            timeout=30
        )

        has_junit4 = junit4_result.returncode == 0 and junit4_result.stdout.strip()

        if has_junit5 and not has_junit4:
            return True, "JUnit 5 imports present, no JUnit 4 patterns found"
        elif has_junit5 and has_junit4:
            files = junit4_result.stdout.strip().split('\n')
            return False, f"JUnit 4 patterns still in {len(files)} files"
        elif not has_junit5:
            return False, "No JUnit 5 imports found"

        return False, "JUnit migration status unclear"

    def _verify_compilation(self) -> tuple:
        """Verify project compiles successfully"""
        import subprocess

        result = subprocess.run(
            ['mvn', 'compile', '-q', '-B'],
            cwd=self.project_path,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            return True, "Project compiles successfully"
        return False, f"Compilation failed: {result.stderr[:200]}"

    def _verify_tests(self) -> tuple:
        """Verify tests pass"""
        import subprocess

        result = subprocess.run(
            ['mvn', 'test', '-q', '-B'],
            cwd=self.project_path,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            return True, "Tests pass successfully"
        return False, f"Tests failed: {result.stderr[:200]}"

    def _verify_imports_generic(self) -> tuple:
        """Generic import verification - just check compilation"""
        return self._verify_compilation()

    def _verify_branch_exists(self) -> tuple:
        """Verify migration branch exists"""
        import subprocess

        result = subprocess.run(
            ['git', 'branch', '--list', 'migration*'],
            cwd=self.project_path,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            return True, f"Migration branch exists: {result.stdout.strip()}"
        return False, "No migration branch found"

    def _get_current_task_from_visible(self) -> str:
        """Get current task from VISIBLE_TASKS.md"""
        import os
        visible_path = os.path.join(self.project_path, "VISIBLE_TASKS.md")
        if not os.path.exists(visible_path):
            return ""

        try:
            with open(visible_path, 'r') as f:
                content = f.read()

            if 'CURRENT TASK' in content:
                current_section = content.split('CURRENT TASK')[1]
                if 'UPCOMING' in current_section:
                    current_section = current_section.split('UPCOMING')[0]

                for line in current_section.split('\n'):
                    line = line.strip()
                    if line.startswith('- [') and ']' in line:
                        task = line.split(']', 1)[1].strip()
                        return task
        except Exception as e:
            log_agent(f"[VERIFY] Error reading VISIBLE_TASKS.md: {e}")

        return ""

    def _handle_nothing_to_commit(self, result_str: str) -> str:
        """
        Handle 'nothing to commit' scenario with verification.

        Returns modified result string or triggers auto-sync if verified.
        """
        if not self.task_manager:
            return result_str

        current_task = self._get_current_task_from_visible()
        if not current_task:
            log_agent("[AUTO_SYNC] ⚠ 'Nothing to commit' but no current task found")
            return result_str

        attempt_count = self._increment_task_attempt(current_task)
        is_verified, verify_reason = self._verify_task_complete(current_task)

        if is_verified:
            log_agent(f"[AUTO_SYNC] ✅ Task verified complete despite no commit: {verify_reason}")
            # Trigger task completion via task_manager
            if hasattr(self.task_manager, 'update_visible_tasks_file'):
                from .state import StateFileManager
                sfm = StateFileManager(self.project_path)
                self.task_manager.update_visible_tasks_file(sfm, mark_current_complete=True)

                # LOG TASK COMPLETION to COMPLETED_ACTIONS.md
                if self.action_logger:
                    self.action_logger.log_task_completion(current_task, "VERIFIED_NO_COMMIT")
                    log_agent(f"[TASK_LOG] Logged verified task completion: {current_task[:60]}...")

                return f"{result_str}\n\n[AUTO-VERIFIED] Task marked complete: {verify_reason}"
        else:
            log_agent(f"[AUTO_SYNC] ⚠ Task not verified: {verify_reason}")

            # Force complete after 5 attempts to prevent infinite loop
            if attempt_count >= 5:
                log_agent(f"[LOOP_BREAK] 🔄 Task attempted {attempt_count} times - forcing completion")
                if hasattr(self.task_manager, 'update_visible_tasks_file'):
                    from .state import StateFileManager
                    sfm = StateFileManager(self.project_path)
                    self.task_manager.update_visible_tasks_file(sfm, mark_current_complete=True)

                    # LOG TASK COMPLETION to COMPLETED_ACTIONS.md
                    if self.action_logger:
                        self.action_logger.log_task_completion(current_task, "FORCE_COMPLETED")
                        log_agent(f"[TASK_LOG] Logged force-completed task: {current_task[:60]}...")

                    return f"{result_str}\n\n[FORCE-COMPLETED] Task force-completed after {attempt_count} attempts"

        return result_str

    def has_recent_tracked_tool_call(self) -> tuple:
        """
        Check if a tracked tool was successfully called recently.

        Returns:
            (bool, str): (has_recent_call, tool_name_or_reason)
        """
        if not self.recent_tracked_tools:
            return False, "No tracked tools have been called yet"

        now = datetime.now()
        cutoff_time = now - timedelta(seconds=self.verification_window_seconds)

        # Filter to recent calls within the window
        recent_calls = [
            tool_call for tool_call in self.recent_tracked_tools
            if tool_call['timestamp'] > cutoff_time
        ]

        if not recent_calls:
            oldest = self.recent_tracked_tools[-1]
            seconds_ago = (now - oldest['timestamp']).total_seconds()
            return False, f"Last tracked tool was {oldest['tool_name']} called {seconds_ago:.0f}s ago (outside {self.verification_window_seconds}s window)"

        # Return the most recent one
        most_recent = recent_calls[-1]
        seconds_ago = (now - most_recent['timestamp']).total_seconds()
        return True, f"{most_recent['tool_name']} called {seconds_ago:.0f}s ago"


    def wrap_error_tool(self, tool) -> StructuredTool:
        """
        Wrap error agent tools with restrictions.

        Error agent should NOT see state files (TODO.md, VISIBLE_TASKS.md, etc.)
        Error agent should ONLY see project code files and error messages.
        """
        tool_name = tool.name

        # Only wrap read_file - other tools are safe
        if tool_name == 'read_file':
            return self.wrap_error_read_file(tool)

        return tool


def get_tools_for_agent(agent_type: str, all_tools: List, tool_wrapper: ToolWrapper = None, on_commit_success: Callable = None) -> List:
    """
    Get the appropriate tool set for an agent type.

    Args:
        agent_type: One of 'analysis', 'execution', 'error', 'supervisor'
        all_tools: List of all available tools
        tool_wrapper: ToolWrapper instance for wrapping tools
        on_commit_success: Callback for commit success (execution only)

    Returns:
        List of tools for the agent
    """
    tool_names = {
        'analysis': ANALYSIS_TOOL_NAMES,
        'execution': EXECUTION_TOOL_NAMES,
        'error': ERROR_TOOL_NAMES,
        'supervisor': SUPERVISOR_TOOL_NAMES,
    }.get(agent_type, set())

    # Filter tools by name
    tools = [tool for tool in all_tools if tool.name in tool_names]

    if not tool_wrapper:
        return tools

    # Wrap tools based on agent type
    if agent_type == 'analysis':
        wrapped = [tool_wrapper.wrap_analysis_tool(tool) for tool in tools]
        log_agent(f"[TOOLS] Analysis agent has {len(wrapped)} tools (read + write state files)")
        log_agent(f"[TOOLS] write_file restricted to: {', '.join(ANALYSIS_ALLOWED_FILES)}")
        return wrapped

    elif agent_type == 'execution':
        wrapped = [tool_wrapper.wrap_execution_tool(tool, on_commit_success) for tool in tools]
        log_agent(f"[TOOLS] Execution agent has {len(wrapped)} tools (write + execute)")
        log_agent(f"[TOOLS] Execution tools: {sorted([t.name for t in wrapped])}")
        return wrapped

    elif agent_type == 'error':
        wrapped = []
        for tool in tools:
            if tool.name == 'read_file':
                wrapped.append(tool_wrapper.wrap_error_read_file(tool))
            else:
                wrapped.append(tool)
        log_agent(f"[TOOLS] Error agent has {len(wrapped)} tools (diagnostic only)")
        log_agent(f"[TOOLS] Error agent CANNOT read: {', '.join(ERROR_AGENT_BLOCKED_FILES)}")
        return wrapped

    else:
        log_agent(f"[TOOLS] {agent_type} agent has {len(tools)} tools")
        return tools
