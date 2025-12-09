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
        Check if the MOST RECENT build tool result contains errors.

        IMPORTANT: This follows the LangGraph best practice of checking the CURRENT
        state of the most recent tool result, NOT scanning all historical messages
        for any error. A successful build after a failed build means NO ERROR.

        See: https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph
        "Don't scan messages for tool success—instead, use direct state updates
        that reflect tool outcomes"

        Args:
            messages: List of messages to check

        Returns:
            (has_error: bool, error_summary: str, error_type: str)
            error_type: 'compile', 'test', 'pom', or 'none'
        """
        # Patterns for POM/configuration errors (need manual intervention or specific fixes)
        POM_ERROR_PATTERNS = [
            r'malformed\s+pom',
            r'non[\-\s]?parseable\s+pom',
            r'unrecognised\s+tag',
            r'unrecognized\s+tag',  # alternate spelling
            r'problems?\s+were\s+encountered\s+while\s+processing\s+the\s+pom',
            r'the\s+build\s+could\s+not\s+read',
            r'dependencies\.dependency\.version.*is\s+missing',
            r'dependency\.version.*for.*is\s+missing',
            r'\'dependencies\.dependency\.version\'',
            r'project\.version.*is\s+missing',
            r'invalid\s+pom',
            r'error\s+parsing',
            r'xml\s+parsing\s+error',
            r'premature\s+end\s+of\s+file',
            r'content\s+is\s+not\s+allowed\s+in\s+prolog',
            r'must\s+be\s+terminated\s+by\s+the\s+matching',  # XML tag mismatch
            r'element.*not\s+allowed\s+here',
            r'cvc-complex-type',  # XML schema validation error
            r'could\s+not\s+find\s+artifact',
            r'could\s+not\s+resolve\s+dependencies',
            r'failure\s+to\s+find',
        ]

        # Patterns for compile errors (immediately route to error_expert)
        COMPILE_ERROR_PATTERNS = [
            r'cannot\s+find\s+symbol',
            r'compilation\s+error',
            r'package\s+.*\s+does\s+not\s+exist',
            r'class\s+.*\s+does\s+not\s+exist',
            r'incompatible\s+types',
            r'method\s+.*\s+cannot\s+be\s+applied',
            r'non-static\s+.*\s+cannot\s+be\s+referenced',
            r'unreported\s+exception',
            r'error:\s+\[',  # javac error format
            r'cannot\s+be\s+applied\s+to',
            r'is\s+not\s+abstract\s+and\s+does\s+not\s+override',
            r'has\s+private\s+access',
            r'cannot\s+access',
            r'bad\s+operand',
            r'illegal\s+start\s+of\s+expression',
            r'reached\s+end\s+of\s+file\s+while\s+parsing',
            r'unclosed\s+string\s+literal',
        ]

        # Patterns for test failures (retry once before routing to error_expert)
        TEST_FAILURE_PATTERNS = [
            r'Tests\s+run:.*Failures:\s*[1-9]',
            r'Tests\s+run:.*Errors:\s*[1-9]',
            r'There\s+are\s+test\s+failures',
            r'Failed\s+tests:',
            r'Tests\s+in\s+error:',
            r'Test\s+.*\s+FAILED',
            r'testCompile.*FAILED',
            r'java\.lang\.AssertionError',
            r'org\.junit\..*AssertionFailedError',
            r'expected:<.*>\s+but\s+was:<.*>',
        ]

        # Patterns that indicate a SUCCESSFUL build (no errors)
        SUCCESS_PATTERNS = [
            r'BUILD\s+SUCCESS',
            r'Return\s+code:\s*0',
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

            # Check if this is a build-related message (mvn_compile, mvn_test, etc.)
            is_build_tool_result = self._is_build_tool_result(msg_content, msg_name)

            if not is_build_tool_result:
                continue

            # FOUND A BUILD TOOL RESULT - This is the MOST RECENT one
            # Check SUCCESS first - if the most recent build succeeded, NO ERROR
            for pattern in SUCCESS_PATTERNS:
                if re.search(pattern, msg_content, re.IGNORECASE):
                    log_agent(f"[ERROR_DETECT] Most recent build SUCCEEDED (pattern: {pattern})")
                    return False, "", "none"

            # Check for failure indicators
            has_failure_indicator = (
                'BUILD FAILURE' in msg_content or
                'BUILD ERROR' in msg_content or
                '[ERROR]' in msg_content or
                'COMPILATION ERROR' in msg_content or
                'Failed to execute goal' in msg_content or
                'Return code: 1' in msg_content
            )

            if not has_failure_indicator:
                # Build tool result found but no failure indicators - treat as success
                log_agent(f"[ERROR_DETECT] Most recent build tool result has no failure indicators - treating as success")
                return False, "", "none"

            # Extract error summary - get more context
            error_lines = []
            for line in msg_content.split('\n'):
                if any(kw in line for kw in ['ERROR', 'FAILURE', 'Failed', 'error:', 'cannot', 'missing']):
                    error_lines.append(line.strip())
            error_summary = '\n'.join(error_lines[:10]) if error_lines else msg_content[:800]

            # Check for POM errors FIRST (most specific, need different handling)
            for pattern in POM_ERROR_PATTERNS:
                if re.search(pattern, msg_content, re.IGNORECASE):
                    log_agent(f"[ERROR_DETECT] POM/configuration error detected: {pattern}")
                    log_summary(f"POM ERROR: {pattern} - requires configuration fix")
                    return True, error_summary, 'pom'

            # Check for test failures SECOND (more specific than compile)
            for pattern in TEST_FAILURE_PATTERNS:
                if re.search(pattern, msg_content, re.IGNORECASE):
                    log_agent(f"[ERROR_DETECT] Test failure detected: {pattern}")
                    return True, error_summary, 'test'

            # Check for compile errors THIRD
            for pattern in COMPILE_ERROR_PATTERNS:
                if re.search(pattern, msg_content, re.IGNORECASE):
                    log_agent(f"[ERROR_DETECT] Compile error detected: {pattern}")
                    return True, error_summary, 'compile'

            # Generic build failure (treat as compile error for immediate handling)
            log_agent(f"[ERROR_DETECT] Generic build failure detected (no specific pattern matched)")
            return True, error_summary, 'compile'

        # No build tool results found in messages
        log_agent(f"[ERROR_DETECT] No build tool results found in messages")
        return False, "", "none"

    def _is_build_tool_result(self, content: str, name: str) -> bool:
        """
        Check if a message is a build tool result (mvn_compile, mvn_test, etc.)

        Args:
            content: Message content
            name: Message/tool name

        Returns:
            True if this is a build tool result
        """
        # Check by name first (most reliable)
        if name:
            name_lower = name.lower()
            build_tool_names = ['mvn_compile', 'mvn_test', 'mvn_rewrite', 'maven']
            if any(tool in name_lower for tool in build_tool_names):
                return True

        # Check by content patterns (for messages without clear names)
        build_content_indicators = [
            'Return code:',  # Our Maven tools return this
            'BUILD SUCCESS',
            'BUILD FAILURE',
            '[INFO] BUILD',
            'mvn compile',
            'mvn test',
            '[INFO] --- maven-compiler-plugin',
            '[INFO] --- maven-surefire-plugin',
        ]

        return any(indicator in content for indicator in build_content_indicators)

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
