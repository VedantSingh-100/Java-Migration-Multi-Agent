"""
Error Handler for Migration Orchestrator

This module handles:
- Build/test error detection
- Error history tracking (ERROR_HISTORY.md)
- Stuck loop detection
- Error resolution tracking

Errors are tracked in ERROR_HISTORY.md to prevent infinite retry loops
and help diagnose persistent issues.
"""

import os
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Optional

from langchain_core.messages import BaseMessage

from src.utils.logging_config import log_agent, log_summary
from .constants import MAX_ERROR_ATTEMPTS, MAX_LOOPS_WITHOUT_PROGRESS


class ErrorHandler:
    """
    Handles error detection, tracking, and resolution for the migration process.

    Features:
    - Build/test error detection from messages
    - Error history logging to ERROR_HISTORY.md
    - Stuck loop detection
    - Action tracking for progress monitoring
    """

    def __init__(self, project_path: str = None, action_window_size: int = 10):
        """
        Args:
            project_path: Path to project directory
            action_window_size: Number of recent actions to track for loop detection
        """
        self.project_path = project_path
        self.action_window_size = action_window_size
        self.recent_actions = []

    def set_project_path(self, project_path: str):
        """Update the project path"""
        self.project_path = project_path

    def detect_build_error(self, messages: List[BaseMessage]) -> Tuple[bool, str, str]:
        """
        Check if messages contain build errors from Maven or compilation.

        Args:
            messages: List of messages to check

        Returns:
            (has_error: bool, error_summary: str, error_type: str)
            error_type: 'compile', 'test', or 'none'
        """
        # Patterns for compile errors (immediately route to error_expert)
        COMPILE_ERROR_PATTERNS = [
            'cannot find symbol',
            'compilation error',
            'package .* does not exist',
            'class .* does not exist',
            'incompatible types',
            'method .* cannot be applied',
            'non-static .* cannot be referenced',
            'unreported exception',
        ]

        # Patterns for test failures (retry once before routing to error_expert)
        TEST_FAILURE_PATTERNS = [
            r'Tests run:.*Failures: [1-9]',
            r'Tests run:.*Errors: [1-9]',
            r'There are test failures',
            r'Failed tests:',
            r'Tests in error:',
            r'Test .* FAILED',
            r'testCompile.*FAILED',
        ]

        import re

        for msg in reversed(messages):
            msg_content = ""
            msg_name = ""

            if isinstance(msg, dict):
                msg_content = str(msg.get('content', ''))
                msg_name = msg.get('name', '')
            elif hasattr(msg, 'content'):
                msg_content = str(msg.content)
                msg_name = getattr(msg, 'name', '')

            # Only check Maven/build related messages
            if not msg_name or not ('mvn' in msg_name.lower() or 'compile' in msg_name.lower() or 'test' in msg_name.lower()):
                continue

            # Check for BUILD FAILURE/ERROR indicator
            if 'BUILD FAILURE' not in msg_content and 'BUILD ERROR' not in msg_content and '[ERROR]' not in msg_content:
                continue

            # Extract error summary
            error_lines = [line for line in msg_content.split('\n') if 'ERROR' in line or 'FAILURE' in line or 'Failed' in line]
            error_summary = '\n'.join(error_lines[:5]) if error_lines else msg_content[:500]

            # Check for test failures FIRST (more specific)
            for pattern in TEST_FAILURE_PATTERNS:
                if re.search(pattern, msg_content, re.IGNORECASE):
                    log_agent(f"[ERROR_DETECT] Test failure detected: {pattern}")
                    return True, error_summary, 'test'

            # Check for compile errors
            for pattern in COMPILE_ERROR_PATTERNS:
                if re.search(pattern, msg_content, re.IGNORECASE):
                    log_agent(f"[ERROR_DETECT] Compile error detected: {pattern}")
                    return True, error_summary, 'compile'

            # Generic build failure (treat as compile error for immediate handling)
            log_agent(f"[ERROR_DETECT] Generic build failure detected")
            return True, error_summary, 'compile'

        return False, "", "none"

    def log_error_attempt(self, error: str, attempt_num: int,
                          was_successful: bool, attempted_fixes: List[str] = None):
        """
        Record error resolution attempt to ERROR_HISTORY.md.

        Args:
            error: Error message/description
            attempt_num: Which attempt number (1, 2, 3)
            was_successful: Whether the error was resolved
            attempted_fixes: List of fix descriptions
        """
        if not self.project_path:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        error_snippet = error[:200] if error else "Unknown error"

        # Check if this is a new error or continuation
        error_history = self._read_error_history()
        if f"## Error #{attempt_num}" not in error_history:
            # New error - create header
            self._append_to_error_history(
                f"\n## [x] Error #{attempt_num}: {error_snippet} [{timestamp}]"
            )

        # Record attempt result with details
        status = "RESOLVED" if was_successful else "FAILED"
        self._append_to_error_history(
            f"- [x] Attempt {attempt_num}: {status} [{timestamp}]"
        )

        # Log what fixes were attempted (if provided)
        if attempted_fixes:
            for fix in attempted_fixes:
                self._append_to_error_history(f"  - Tried: {fix}")

        log_agent(f"[ERROR_HISTORY] Logged error attempt #{attempt_num}: {status}")

    def is_error_duplicate(self, error_message: str) -> bool:
        """
        Check if this error has already been attempted too many times.

        Args:
            error_message: The error message to check

        Returns:
            True if this error should be skipped (already maxed attempts)
        """
        error_history = self._read_error_history()
        error_signature = error_message[:100] if error_message else ""

        # Count how many times this error signature appears
        if error_signature:
            count = error_history.count(error_signature)
            if count >= MAX_ERROR_ATTEMPTS:
                log_agent(f"[ERROR_DEDUPE] Error already attempted {count} times - skipping")
                return True

        return False

    def detect_stuck_loop(self) -> Tuple[bool, str]:
        """
        Detect if agent is stuck in a repetitive loop.

        Analyzes recent actions to identify patterns:
        1. Same tool called repeatedly (3+ times in window)
        2. No completed actions logged in recent window
        3. Same TODO item attempted repeatedly without completion

        Returns:
            (is_stuck: bool, reason: str)
        """
        if len(self.recent_actions) < self.action_window_size:
            return (False, "Not enough action history yet")

        # Get last N actions
        last_n = self.recent_actions[-self.action_window_size:]

        # Pattern 1: Same tool called too many times
        tool_counts = Counter(a.get('tool_name') for a in last_n if a.get('tool_name'))
        for tool, count in tool_counts.items():
            if count >= 3:
                log_agent(f"[LOOP_DETECT] Tool '{tool}' called {count} times in last {self.action_window_size} actions", "WARNING")
                return (True, f"Tool '{tool}' called {count} times in last {self.action_window_size} actions")

        # Pattern 2: No completed actions logged
        completions_logged = sum(1 for a in last_n if a.get('logged_to_completed'))
        if completions_logged == 0:
            log_agent(f"[LOOP_DETECT] No actions logged to COMPLETED_ACTIONS in last {self.action_window_size} calls", "WARNING")
            return (True, f"No progress: 0 completions in last {self.action_window_size} actions")

        # Pattern 3: Same TODO item attempted repeatedly
        todo_counts = Counter(a.get('todo_item') for a in last_n if a.get('todo_item'))
        for todo_item, count in todo_counts.items():
            if count >= 3:
                log_agent(f"[LOOP_DETECT] TODO '{todo_item[:50]}...' attempted {count} times", "WARNING")
                return (True, f"TODO item attempted {count} times without completion")

        return (False, "Agent making progress")

    def track_action(self, tool_name: str, todo_item: str = None, logged_to_completed: bool = False):
        """
        Track an action for loop detection.

        Args:
            tool_name: Name of the tool that was called
            todo_item: Current TODO item being worked on (if known)
            logged_to_completed: Whether this action was logged to COMPLETED_ACTIONS.md
        """
        action = {
            'tool_name': tool_name,
            'todo_item': todo_item,
            'logged_to_completed': logged_to_completed,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        self.recent_actions.append(action)

        # Keep only last 20 actions in memory (2x window size for history)
        if len(self.recent_actions) > self.action_window_size * 4:
            self.recent_actions = self.recent_actions[-self.action_window_size * 4:]

    def get_error_count_from_state(self, state: dict) -> int:
        """Get current error count from state"""
        return state.get('error_count', 0)

    def has_max_error_attempts(self, state: dict) -> bool:
        """Check if maximum error attempts have been reached"""
        return state.get('error_count', 0) >= MAX_ERROR_ATTEMPTS

    def should_route_to_error_agent(self, state: dict) -> bool:
        """
        Determine if error agent should be invoked.

        Args:
            state: Current workflow state

        Returns:
            True if error agent should handle the error
        """
        has_error = state.get('has_build_error', False)
        error_count = state.get('error_count', 0)

        if not has_error:
            return False

        if error_count >= MAX_ERROR_ATTEMPTS:
            log_agent(f"[ERROR] Max error attempts ({MAX_ERROR_ATTEMPTS}) reached - not routing to error agent")
            return False

        return True

    def _read_error_history(self) -> str:
        """Read ERROR_HISTORY.md content"""
        if not self.project_path:
            return ""

        error_history_path = os.path.join(self.project_path, "ERROR_HISTORY.md")
        if not os.path.exists(error_history_path):
            return ""

        try:
            with open(error_history_path, 'r') as f:
                return f.read()
        except Exception:
            return ""

    def _append_to_error_history(self, content: str):
        """Append content to ERROR_HISTORY.md"""
        if not self.project_path:
            return

        error_history_path = os.path.join(self.project_path, "ERROR_HISTORY.md")
        try:
            with open(error_history_path, 'a') as f:
                f.write(content + "\n")
        except Exception as e:
            log_agent(f"[ERROR_HISTORY] Error writing to file: {e}")


class StuckDetector:
    """
    Detects when the migration process is stuck and not making progress.

    Tracks progress through TODO completion and action logging.
    """

    def __init__(self, max_loops_without_progress: int = MAX_LOOPS_WITHOUT_PROGRESS):
        """
        Args:
            max_loops_without_progress: Number of loops without progress before considered stuck
        """
        self.max_loops_without_progress = max_loops_without_progress
        self.last_todo_count = 0
        self.loops_without_progress = 0

    def check_progress(self, current_todo_count: int) -> Tuple[bool, str]:
        """
        Check if progress is being made based on TODO completion.

        Args:
            current_todo_count: Current number of completed tasks

        Returns:
            (made_progress: bool, status_message: str)
        """
        if current_todo_count > self.last_todo_count:
            self.loops_without_progress = 0
            self.last_todo_count = current_todo_count
            return (True, f"Progress made: {current_todo_count} tasks completed")
        else:
            self.loops_without_progress += 1
            if self.loops_without_progress >= self.max_loops_without_progress:
                return (False, f"No progress for {self.loops_without_progress} loops (stuck)")
            else:
                return (True, f"No change this loop ({self.loops_without_progress}/{self.max_loops_without_progress})")

    def is_stuck(self) -> bool:
        """Check if the process is considered stuck"""
        return self.loops_without_progress >= self.max_loops_without_progress

    def reset(self):
        """Reset the stuck detector state"""
        self.last_todo_count = 0
        self.loops_without_progress = 0


def initialize_error_history_file(project_path: str):
    """
    Initialize ERROR_HISTORY.md file.

    Args:
        project_path: Path to project directory
    """
    error_history_path = os.path.join(project_path, "ERROR_HISTORY.md")

    if not os.path.exists(error_history_path):
        header = """# Error History

This file tracks error resolution attempts during migration.
It helps prevent infinite retry loops and diagnose persistent issues.

"""
        try:
            with open(error_history_path, 'w') as f:
                f.write(header)
            log_agent(f"[ERROR_HISTORY] Initialized ERROR_HISTORY.md")
        except Exception as e:
            log_agent(f"[ERROR_HISTORY] Error initializing file: {e}")
