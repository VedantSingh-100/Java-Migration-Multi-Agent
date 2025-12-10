"""
State Management for Migration Orchestrator

This module contains:
- State class (defines the shared state schema for LangGraph)
- State file operations (reading/writing TODO.md, COMPLETED_ACTIONS.md, etc.)
- Phase determination logic

IMPORTANT: The State class inherits from AgentState which uses `add_messages`
reducer. This means messages ACCUMULATE across nodes rather than being replaced.
This is a key design decision that affects how message pruning works.
"""

import os
from typing import Any
from langgraph.prebuilt.chat_agent_executor import AgentState
from src.utils.logging_config import log_agent

from .constants import MAX_SUMMARY_LENGTH, PHASES


class State(AgentState):
    """
    Shared state schema for the migration workflow.

    Inherits from AgentState which provides:
    - messages: List[BaseMessage] with add_messages reducer (ACCUMULATES!)

    CRITICAL: Because of add_messages reducer, when a node returns
    {"messages": [new_msgs]}, those messages are ADDED to existing state,
    not replaced. This affects how phase transitions and context pruning work.

    To "reset" messages, you would need to:
    1. Use a custom reducer that replaces instead of adds, OR
    2. Return messages with matching IDs to deduplicate, OR
    3. Use a different state key for phase-specific messages
    """

    # Context for summarization (currently unused)
    context: dict[str, Any] = {}

    # Phase tracking for deterministic routing
    current_phase: str = "INIT"
    analysis_done: bool = False
    execution_done: bool = False
    project_path: str = ""

    # Stuck detection tracking
    last_todo_count: int = 0
    loops_without_progress: int = 0
    total_execution_loops: int = 0
    stuck_intervention_active: bool = False
    no_tool_call_loops: int = 0  # Track loops where agent returned without tool calls
    thinking_loops: int = 0  # Track consecutive "thinking" responses (allow limited)

    # Error detection and routing
    has_build_error: bool = False
    error_count: int = 0
    last_error_message: str = ""
    error_type: str = "none"  # 'compile', 'test', or 'none'

    # Test failure tracking (for retry-then-route pattern)
    test_failure_count: int = 0  # Consecutive test failures on current task
    last_test_failure_task: str = ""  # Track which task caused the failure


class StateFileManager:
    """
    Manages reading and writing state files in the project directory.

    State files:
    - TODO.md: Task checklist (created by analysis, read by execution)
    - CURRENT_STATE.md: Migration status (append-only)
    - COMPLETED_ACTIONS.md: Action audit trail (system-managed)
    - VISIBLE_TASKS.md: Next 3 tasks only (agent's view)
    - ERROR_HISTORY.md: Error tracking for deduplication
    """

    def __init__(self, project_path: str = None):
        self.project_path = project_path

    def set_project_path(self, project_path: str):
        """Update the project path"""
        self.project_path = project_path

    def read_file(self, filename: str, keep_beginning: bool = False) -> str:
        """
        Read file from project directory, return empty if not exists.

        Args:
            filename: Name of file in project directory
            keep_beginning: If True, truncate from END (keep beginning).
                           If False, truncate from START (keep end - for logs).
                           Default False for backwards compatibility.

        Returns:
            File content, truncated to MAX_SUMMARY_LENGTH if needed
        """
        if not self.project_path:
            return ""

        filepath = os.path.join(self.project_path, filename)
        if not os.path.exists(filepath):
            return ""

        try:
            with open(filepath, 'r') as f:
                content = f.read()

            # Truncate if too long
            if len(content) > MAX_SUMMARY_LENGTH:
                if keep_beginning:
                    # Keep first N chars (for TODO.md - need first tasks)
                    return content[:MAX_SUMMARY_LENGTH]
                else:
                    # Keep last N chars (for logs - need recent entries)
                    return content[-MAX_SUMMARY_LENGTH:]
            return content

        except Exception as e:
            log_agent(f"[STATE_FILE] Error reading {filename}: {str(e)}", "ERROR")
            return ""

    def write_file(self, filename: str, content: str) -> bool:
        """
        Write content to file in project directory.

        Returns:
            True if successful, False otherwise
        """
        if not self.project_path:
            return False

        filepath = os.path.join(self.project_path, filename)
        try:
            with open(filepath, 'w') as f:
                f.write(content)
            return True
        except Exception as e:
            log_agent(f"[STATE_FILE] Error writing {filename}: {str(e)}", "ERROR")
            return False

    def append_file(self, filename: str, content: str) -> bool:
        """
        Append content to file in project directory.

        Returns:
            True if successful, False otherwise
        """
        if not self.project_path:
            return False

        filepath = os.path.join(self.project_path, filename)
        try:
            with open(filepath, 'a') as f:
                f.write(content + "\n")
            return True
        except Exception as e:
            log_agent(f"[STATE_FILE] Error appending to {filename}: {str(e)}", "ERROR")
            return False

    def file_exists(self, filename: str) -> bool:
        """Check if file exists in project directory"""
        if not self.project_path:
            return False
        filepath = os.path.join(self.project_path, filename)
        return os.path.exists(filepath)


def calculate_todo_progress(project_path: str) -> dict:
    """
    Calculate TODO progress by reading TODO.md file.

    Uses consistent parsing logic to avoid inconsistencies between
    different parts of the codebase.

    Args:
        project_path: Path to project directory

    Returns:
        Dict with keys: completed, total, percent, next_unchecked
    """
    if not project_path:
        return {'completed': 0, 'total': 0, 'percent': 0, 'next_unchecked': None}

    todo_path = os.path.join(project_path, "TODO.md")
    if not os.path.exists(todo_path):
        return {'completed': 0, 'total': 0, 'percent': 0, 'next_unchecked': None}

    try:
        with open(todo_path, 'r') as f:
            content = f.read()

        lines = content.split('\n')
        completed_tasks = 0
        unchecked_tasks = []

        for line in lines:
            # Match both [x] and [X] (case insensitive)
            if '- [x]' in line.lower() or '- [X]' in line:
                completed_tasks += 1
            elif '- [ ]' in line:
                task_desc = line.replace('- [ ]', '').strip()
                if task_desc:  # Ignore empty lines
                    unchecked_tasks.append(task_desc)

        total_tasks = completed_tasks + len(unchecked_tasks)
        percent = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

        # Get first unchecked task
        next_unchecked = unchecked_tasks[0] if unchecked_tasks else None

        return {
            'completed': completed_tasks,
            'total': total_tasks,
            'percent': percent,
            'next_unchecked': next_unchecked
        }

    except Exception as e:
        log_agent(f"[STATE] Error calculating TODO progress: {e}")
        return {'completed': 0, 'total': 0, 'percent': 0, 'next_unchecked': None}


def determine_phase_from_progress(todo_stats: dict) -> str:
    """
    Determine current migration phase based on TODO progress.

    Args:
        todo_stats: Dict from calculate_todo_progress()

    Returns:
        Human-readable phase description
    """
    percent = todo_stats.get('percent', 0)

    if percent == 0:
        return "Initial Phase"
    elif percent < 20:
        return "Setup Phase"
    elif percent < 40:
        return "Java Migration Phase"
    elif percent < 60:
        return "Spring Boot Migration Phase"
    elif percent < 80:
        return "Jakarta EE Migration Phase"
    elif percent < 100:
        return "Final Verification Phase"
    else:
        return "Migration Complete"


def is_valid_phase_transition(from_phase: str, to_phase: str) -> bool:
    """
    Check if a phase transition is valid.

    Valid transitions follow the PHASES order, with some exceptions:
    - ERROR_RESOLUTION can transition back to EXECUTION
    - Any phase can transition to FAILED

    Args:
        from_phase: Current phase
        to_phase: Target phase

    Returns:
        True if transition is valid
    """
    if to_phase == "FAILED":
        return True  # Can fail from any phase

    if from_phase == "ERROR_RESOLUTION" and to_phase == "EXECUTION":
        return True  # Can return to execution after fixing error

    try:
        from_idx = PHASES.index(from_phase)
        to_idx = PHASES.index(to_phase)
        # Allow moving forward or staying in same phase
        return to_idx >= from_idx
    except ValueError:
        # Unknown phase - allow for flexibility
        return True
