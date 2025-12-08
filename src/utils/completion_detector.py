"""
Completion Detection Logic - Automatic detection of agent completion

This module provides functions to automatically detect when agents have
completed their work, eliminating reliance on LLM tool calls.
"""

import os
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage
from src.utils.logging_config import log_agent


def detect_analysis_complete(project_path: str, messages: List[BaseMessage]) -> bool:
    """
    Detect if analysis phase is complete by checking artifacts.

    Analysis is complete when:
    1. TODO.md exists and has content
    2. CURRENT_STATE.md exists and has content
    3. Last messages suggest analysis work is done

    Args:
        project_path: Path to the project being migrated
        messages: Message history from agent

    Returns:
        True if analysis is complete, False otherwise
    """
    try:
        # Check for required files
        todo_path = os.path.join(project_path, "TODO.md")
        current_state_path = os.path.join(project_path, "CURRENT_STATE.md")

        if not os.path.exists(todo_path) or not os.path.exists(current_state_path):
            log_agent(f"[DETECT] Analysis NOT complete - missing files")
            return False

        # Check if files have meaningful content (not just empty)
        with open(todo_path, 'r') as f:
            todo_content = f.read().strip()
        with open(current_state_path, 'r') as f:
            current_state_content = f.read().strip()

        if len(todo_content) < 50 or len(current_state_content) < 50:
            log_agent(f"[DETECT] Analysis NOT complete - files too short")
            return False

        # Check if TODO has actual tasks (contains list markers)
        if not any(marker in todo_content for marker in ['- [ ]', '- [x]', '- ', '* ', '1.']):
            log_agent(f"[DETECT] Analysis NOT complete - TODO has no tasks")
            return False

        log_agent(f"[DETECT] ✅ Analysis COMPLETE - Required files created with content")
        return True

    except Exception as e:
        log_agent(f"[DETECT] Error checking analysis completion: {e}")
        return False


def detect_execution_complete(project_path: str, messages: List[BaseMessage]) -> bool:
    """
    Detect if execution phase is complete by checking TODO.md status.

    Execution is complete when:
    1. TODO.md exists
    2. All TODO items are checked off (✅ or [x])
    3. Recent messages mention build success

    Args:
        project_path: Path to the project being migrated
        messages: Message history from agent

    Returns:
        True if execution is complete, False otherwise
    """
    try:
        todo_path = os.path.join(project_path, "TODO.md")

        if not os.path.exists(todo_path):
            log_agent(f"[DETECT] Execution NOT complete - TODO.md missing")
            return False

        with open(todo_path, 'r') as f:
            todo_content = f.read()

        # Find all TODO items (lines starting with - or *)
        lines = todo_content.split('\n')
        todo_items = []
        checked_items = []

        for line in lines:
            stripped = line.strip()
            # Look for unchecked items
            if stripped.startswith('- [ ]') or stripped.startswith('* [ ]'):
                todo_items.append(line)
            # Look for checked items
            elif stripped.startswith('- [x]') or stripped.startswith('* [x]') or '✅' in stripped:
                todo_items.append(line)
                checked_items.append(line)
            # Simple list items (no checkbox)
            elif stripped.startswith('-') or stripped.startswith('*'):
                # If it has a checkmark emoji, count as checked
                if '✅' in stripped or '[x]' in stripped.lower():
                    checked_items.append(line)
                todo_items.append(line)

        if len(todo_items) == 0:
            log_agent(f"[DETECT] Execution NOT complete - no TODO items found")
            return False

        # Check if all items are checked
        completion_ratio = len(checked_items) / len(todo_items) if todo_items else 0

        log_agent(f"[DETECT] TODO completion: {len(checked_items)}/{len(todo_items)} items ({completion_ratio:.0%})")

        # Require 80% completion (relaxed from 100% due to message pruning potentially hiding build success)
        # Message pruning can remove build success indicators from recent messages
        if completion_ratio >= 0.8:
            log_agent(f"[DETECT] ✅ Execution COMPLETE - {completion_ratio:.0%} of TODOs done (threshold: 80%)")
            return True

        log_agent(f"[DETECT] Execution NOT complete - only {completion_ratio:.0%} of TODOs done (need 80%)")
        return False

    except Exception as e:
        log_agent(f"[DETECT] Error checking execution completion: {e}")
        return False


def get_todo_checked_count(project_path: str) -> int:
    """
    Get count of checked TODO items for progress tracking.

    Args:
        project_path: Path to the project being migrated

    Returns:
        Number of checked TODO items (for stuck detection)
    """
    try:
        todo_path = os.path.join(project_path, "TODO.md")

        if not os.path.exists(todo_path):
            return 0

        with open(todo_path, 'r') as f:
            todo_content = f.read()

        # Count checked items
        lines = todo_content.split('\n')
        checked_count = 0

        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('- [x]') or stripped.startswith('* [x]') or
                ('✅' in stripped and (stripped.startswith('-') or stripped.startswith('*')))):
                checked_count += 1

        return checked_count

    except Exception as e:
        log_agent(f"[PROGRESS] Error counting TODOs: {e}")
        return 0


def get_completion_status(project_path: str, messages: List[BaseMessage], current_phase: str) -> Dict[str, Any]:
    """
    Get comprehensive completion status for current phase.

    Args:
        project_path: Path to the project being migrated
        messages: Message history
        current_phase: Current migration phase

    Returns:
        Dictionary with completion status
    """
    status = {
        "phase": current_phase,
        "analysis_done": False,
        "execution_done": False
    }

    if current_phase in ["INIT", "ANALYSIS"]:
        status["analysis_done"] = detect_analysis_complete(project_path, messages)

    if current_phase in ["EXECUTION", "TESTING"]:
        status["execution_done"] = detect_execution_complete(project_path, messages)

    return status