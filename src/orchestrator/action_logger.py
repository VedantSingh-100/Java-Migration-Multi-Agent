"""
Action Logger for Migration Orchestrator

This module handles:
- Logging completed actions to COMPLETED_ACTIONS.md
- Formatting tool results for logging
- Task-level completion tracking
- Action history management

COMPLETED_ACTIONS.md serves as external memory that persists across
context window limits. It's system-managed and agents cannot modify it.
"""

import os
import re
from datetime import datetime
from typing import Any, Optional

from src.utils.logging_config import log_agent


class ActionLogger:
    """
    Logs migration actions to COMPLETED_ACTIONS.md for external memory tracking.

    Features:
    - Tool-level action logging (what tool did what)
    - Task-level completion logging (what was accomplished)
    - Result formatting for human readability
    - Error extraction for actionable feedback
    """

    def __init__(self, project_path: str = None):
        """
        Args:
            project_path: Path to project directory
        """
        self.project_path = project_path

    def set_project_path(self, project_path: str):
        """Update the project path"""
        self.project_path = project_path

    def log_action(self, tool_name: str, success: bool, duration: float = 0,
                   args: tuple = None, kwargs: dict = None,
                   result: Any = None, error: str = None):
        """
        Log completed action to COMPLETED_ACTIONS.md.

        Args:
            tool_name: Name of the tool that was executed
            success: Whether the tool execution was successful
            duration: Execution time in seconds
            args: Positional arguments passed to tool
            kwargs: Keyword arguments passed to tool
            result: Tool execution result
            error: Error message if failed
        """
        if not self.project_path:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "SUCCESS" if success else "FAILED"

        # Format the action entry
        entry = f"[{timestamp}] {tool_name}: {status}"
        if duration:
            entry += f" ({duration:.1f}s)"

        # Add brief result/error info
        if error:
            entry += f" - Error: {error[:100]}"
        elif result:
            formatted = self.format_result(result, tool_name)
            if formatted and len(formatted) < 100:
                entry += f" - {formatted}"

        self._append_to_file("COMPLETED_ACTIONS.md", entry)

        # SELF-HEALING: Ensure file has proper structure after every action
        self._ensure_file_structure()

    def log_task_completion(self, task_description: str, commit_hash: str = None):
        """
        Log TASK-LEVEL completion to COMPLETED_ACTIONS.md.

        This creates high-level task tracking that survives context trimming.
        Different from tool-level logging - this shows WHAT was accomplished.

        Args:
            task_description: Description of the completed task
            commit_hash: Git commit hash if available
        """
        if not self.project_path:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Build task completion entry
        commit_info = f" (commit: {commit_hash[:7]})" if commit_hash else ""
        entry = f"[{timestamp}] TASK COMPLETED: {task_description}{commit_info}"

        # Append to special TASK COMPLETIONS section
        completed_actions_path = os.path.join(self.project_path, "COMPLETED_ACTIONS.md")
        try:
            # Read existing content
            content = ""
            if os.path.exists(completed_actions_path):
                with open(completed_actions_path, 'r') as f:
                    content = f.read()

            # Find or create TASK COMPLETIONS section
            if '=== TASK COMPLETIONS ===' not in content:
                # Insert section after header
                if '=== ACTION LOG ===' in content:
                    parts = content.split('=== ACTION LOG ===')
                    content = parts[0] + "=== TASK COMPLETIONS ===\n\n=== ACTION LOG ===" + parts[1]
                else:
                    content = "=== TASK COMPLETIONS ===\n\n" + content

            # Insert new completion entry
            if '=== TASK COMPLETIONS ===' in content and '=== ACTION LOG ===' in content:
                parts = content.split('=== ACTION LOG ===')
                task_section = parts[0]
                action_section = '=== ACTION LOG ===' + parts[1]

                # Add entry to task section
                new_content = task_section.rstrip() + "\n" + entry + "\n\n" + action_section
            else:
                new_content = content + "\n" + entry

            with open(completed_actions_path, 'w') as f:
                f.write(new_content)

            log_agent(f"[TASK_LOG] Logged task completion: {task_description[:60]}...")
        except Exception as e:
            log_agent(f"[TASK_LOG] Error logging task completion: {e}")

    def format_result(self, result: Any, tool_name: str) -> str:
        """
        Format tool result for logging based on tool type.

        Args:
            result: The tool execution result
            tool_name: Name of the tool

        Returns:
            Human-readable result string
        """
        result_str = str(result)

        if tool_name == 'create_branch':
            if 'Created branch' in result_str:
                return "Created and checked out new branch"
            elif 'Checked out' in result_str:
                return "Checked out existing branch"
            return "Branch operation completed"

        elif tool_name == 'git_commit':
            if '] ' in result_str:
                # Extract commit hash and message
                parts = result_str.split('] ', 1)
                if len(parts) > 1:
                    return parts[1].split('\n')[0][:100]
            return "Commit successful"

        elif tool_name in ['mvn_rewrite_run_recipe', 'mvn_rewrite_run']:
            # Try to count changes
            change_count = result_str.count('Changes have been made')
            if change_count > 0:
                return f"Recipe executed, {change_count} files changed"
            return "Recipe executed"

        elif tool_name == 'add_openrewrite_plugin':
            return "OpenRewrite plugin added to pom.xml"

        elif tool_name == 'configure_openrewrite_recipes':
            # Try to extract recipe count
            if 'with' in result_str and 'recipe' in result_str:
                match = re.search(r'with (\d+)', result_str)
                if match:
                    return f"Configured {match.group(1)} recipes"
            return "Recipes configured"

        elif tool_name in ['mvn_compile', 'mvn_test']:
            if 'BUILD SUCCESS' in result_str:
                return "Build successful"
            elif 'Return code: 0' in result_str:
                return "Completed successfully"
            return "Maven command executed"

        elif tool_name == 'write_file':
            # Extract filename
            if 'Successfully wrote to' in result_str:
                path = result_str.replace('Successfully wrote to', '').strip()
                filename = path.split('/')[-1] if '/' in path else path
                return f"Wrote {filename}"
            return "File written"

        elif tool_name == 'find_replace':
            match = re.search(r'(\d+) replacement', result_str)
            if match:
                count = match.group(1)
                return f"{count} replacement(s) made"
            return "Text replaced"

        elif tool_name == 'git_add_all':
            return "Staged all changes"

        elif tool_name == 'run_command':
            if len(result_str) > 100:
                return result_str[:97] + "..."
            return "Command executed"

        # Default: truncate to 100 chars
        if len(result_str) > 100:
            return result_str[:97] + "..."
        return result_str

    def extract_actionable_error(self, tool_name: str, result: Any, error: Any) -> str:
        """
        Extract actionable error messages instead of generic failures.

        Args:
            tool_name: Name of the tool that failed
            result: Tool execution result
            error: Error message

        Returns:
            Human-readable error message with actionable advice
        """
        result_str = str(result) if result else ""
        error_str = str(error) if error else ""
        combined = result_str + " " + error_str

        # git_commit specific errors
        if tool_name == 'git_commit':
            if 'nothing to commit' in combined.lower():
                return "Nothing to commit (working directory clean). No changes were staged."
            elif 'not a git repository' in combined.lower():
                return "Not a git repository. Run git init first."
            elif 'no changes added' in combined.lower():
                return "No changes added to commit. Use git add to stage files first."
            elif 'please tell me who you are' in combined.lower():
                return "Git user.name/user.email not configured."

        # Maven compilation errors
        elif tool_name == 'mvn_compile':
            if 'compilation error' in combined.lower() or 'cannot find symbol' in combined.lower():
                # Try to extract the actual error
                lines = combined.split('\n')
                for i, line in enumerate(lines):
                    if 'error:' in line.lower():
                        # Get 2 lines of context
                        context = '\n'.join(lines[max(0, i):min(len(lines), i + 2)])
                        return f"Compilation failed: {context[:200]}"
                return "Compilation error - check imports and dependencies"
            elif 'build failure' in combined.lower():
                return "Maven build failed - check pom.xml syntax"

        # Maven test errors
        elif tool_name == 'mvn_test':
            if 'test failures' in combined.lower() or 'tests run:' in combined.lower():
                # Extract failure count
                match = re.search(r'Failures: (\d+)', combined)
                if match:
                    return f"{match.group(1)} test(s) failed. Check test output for details."
                return "Tests failed - check test output"

        # OpenRewrite errors
        elif tool_name in ['mvn_rewrite_run', 'mvn_rewrite_run_recipe']:
            if 'recipe not found' in combined.lower():
                return "OpenRewrite recipe not found. Check recipe name in pom.xml."
            elif 'no recipe specified' in combined.lower():
                return "No recipe specified. Use configure_openrewrite_recipes first."
            elif 'build failure' in combined.lower():
                return "OpenRewrite execution failed. Check pom.xml OpenRewrite configuration."

        # File operation errors
        elif tool_name == 'write_file':
            if 'permission denied' in combined.lower():
                return "Permission denied writing file. Check file permissions."
            elif 'no such file or directory' in combined.lower():
                return "Directory does not exist. Create parent directories first."

        elif tool_name == 'find_replace':
            if 'not found' in combined.lower():
                return "Text to replace not found in file. Check exact string match."

        elif tool_name == 'git_add_all':
            if 'not a git repository' in combined.lower():
                return "Not a git repository. Initialize git first."

        elif tool_name == 'run_command':
            if 'command not found' in combined.lower():
                return "Command not found. Check if tool is installed."
            elif 'return code:' in combined.lower():
                match = re.search(r'return code: (\d+)', combined.lower())
                if match:
                    return f"Command failed with exit code {match.group(1)}"

        # Generic fallback - but try to extract useful info
        if error_str:
            return error_str[:300]
        elif 'unsuccessful' in combined.lower():
            return f"{tool_name} returned unsuccessful result. Check tool output for details."
        else:
            return combined[:300] if combined else "Tool execution failed"

    def _ensure_file_structure(self):
        """
        SELF-HEALING: Ensure COMPLETED_ACTIONS.md has proper template structure.

        This is called after every log_action() to ensure the file always has:
        - Header section
        - TASK COMPLETIONS section
        - ACTION LOG section

        If the file was corrupted, lost, or recreated without structure,
        this rebuilds it while preserving existing content.

        Ported from supervisor_orchestrator.py _update_state_header() method.
        """
        if not self.project_path:
            return

        completed_actions_path = os.path.join(self.project_path, "COMPLETED_ACTIONS.md")

        # Read existing content
        existing_actions = ""
        existing_task_completions = ""
        raw_content = ""

        if os.path.exists(completed_actions_path):
            try:
                with open(completed_actions_path, 'r') as f:
                    raw_content = f.read()
            except Exception as e:
                log_agent(f"[SELF_HEAL] Error reading COMPLETED_ACTIONS.md: {e}")
                return

            # Check if file already has proper structure
            has_task_section = '=== TASK COMPLETIONS ===' in raw_content
            has_action_section = '=== ACTION LOG ===' in raw_content

            if has_task_section and has_action_section:
                # File has proper structure - no healing needed
                return

            log_agent("[SELF_HEAL] COMPLETED_ACTIONS.md missing proper structure - rebuilding")

            # Extract existing content to preserve
            if has_task_section and has_action_section:
                # Extract TASK COMPLETIONS section
                task_start = raw_content.find('=== TASK COMPLETIONS ===') + len('=== TASK COMPLETIONS ===')
                task_end = raw_content.find('=== ACTION LOG ===')
                existing_task_completions = raw_content[task_start:task_end].strip()

                # Extract action log
                existing_actions = raw_content.split('=== ACTION LOG ===', 1)[1].strip()
            elif has_action_section:
                # Only ACTION LOG exists
                existing_actions = raw_content.split('=== ACTION LOG ===', 1)[1].strip()
            else:
                # No structure - treat entire content as action log entries
                # Filter to only keep lines that look like action entries
                action_lines = []
                for line in raw_content.split('\n'):
                    line = line.strip()
                    if line.startswith('[') and ']:' in line:
                        action_lines.append(line)
                existing_actions = '\n'.join(action_lines)

        # Build proper header structure
        header = """# Completed Actions Log

This file tracks all migration actions performed by the system.
It is system-managed and cannot be modified by agents.

=== TASK COMPLETIONS ===

=== ACTION LOG ===
"""

        # Rebuild file with structure + preserved content
        try:
            with open(completed_actions_path, 'w') as f:
                full_content = header

                # Insert existing task completions between sections
                if existing_task_completions:
                    parts = full_content.split('=== ACTION LOG ===')
                    full_content = parts[0] + existing_task_completions + "\n\n=== ACTION LOG ===" + (parts[1] if len(parts) > 1 else "")

                # Append existing action log entries
                if existing_actions:
                    full_content = full_content.rstrip() + "\n" + existing_actions + "\n"

                f.write(full_content)

            log_agent("[SELF_HEAL] COMPLETED_ACTIONS.md structure restored successfully")
        except Exception as e:
            log_agent(f"[SELF_HEAL] Error rebuilding COMPLETED_ACTIONS.md: {e}")

    def _append_to_file(self, filename: str, content: str) -> bool:
        """
        Append content to file in project directory.

        Args:
            filename: Name of file in project directory
            content: Content to append

        Returns:
            True if successful
        """
        if not self.project_path:
            return False

        filepath = os.path.join(self.project_path, filename)
        try:
            with open(filepath, 'a') as f:
                f.write(content + "\n")
            return True
        except Exception as e:
            log_agent(f"[ACTION_LOG] Error appending to {filename}: {str(e)}", "ERROR")
            return False

    def get_action_count(self) -> int:
        """
        Get count of logged actions from COMPLETED_ACTIONS.md.

        Returns:
            Number of logged actions
        """
        if not self.project_path:
            return 0

        completed_actions_path = os.path.join(self.project_path, "COMPLETED_ACTIONS.md")
        if not os.path.exists(completed_actions_path):
            return 0

        try:
            with open(completed_actions_path, 'r') as f:
                content = f.read()
            # Count lines that look like action entries
            return len([line for line in content.split('\n')
                       if line.startswith('[') and ']:' in line])
        except Exception:
            return 0

    def get_recent_actions(self, count: int = 5) -> list:
        """
        Get most recent logged actions.

        Args:
            count: Number of recent actions to retrieve

        Returns:
            List of recent action strings
        """
        if not self.project_path:
            return []

        completed_actions_path = os.path.join(self.project_path, "COMPLETED_ACTIONS.md")
        if not os.path.exists(completed_actions_path):
            return []

        try:
            with open(completed_actions_path, 'r') as f:
                content = f.read()

            # Extract action lines
            action_lines = [line for line in content.split('\n')
                          if line.startswith('[') and ']:' in line]

            return action_lines[-count:] if action_lines else []
        except Exception:
            return []


def initialize_completed_actions_file(project_path: str):
    """
    Initialize COMPLETED_ACTIONS.md file with header structure.

    Args:
        project_path: Path to project directory
    """
    completed_actions_path = os.path.join(project_path, "COMPLETED_ACTIONS.md")

    if not os.path.exists(completed_actions_path):
        header = """# Completed Actions Log

This file tracks all migration actions performed by the system.
It is system-managed and cannot be modified by agents.

=== TASK COMPLETIONS ===

=== ACTION LOG ===
"""
        try:
            with open(completed_actions_path, 'w') as f:
                f.write(header)
            log_agent(f"[ACTION_LOG] Initialized COMPLETED_ACTIONS.md")
        except Exception as e:
            log_agent(f"[ACTION_LOG] Error initializing COMPLETED_ACTIONS.md: {e}")
