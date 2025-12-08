"""
State Management Tools

Tools for checking and managing migration state deterministically.
These tools allow agents to query state without relying on file reads or LLM memory.
"""

from langchain_core.tools import tool
from typing import Dict, Any
from src.utils.logging_config import log_agent


# Global state tracker (will be initialized per migration)
_current_state_tracker = None


def set_state_tracker(tracker):
    """Set the global state tracker for the current migration."""
    global _current_state_tracker
    _current_state_tracker = tracker
    log_agent("[STATE_TOOL] State tracker registered globally")


def get_state_tracker():
    """Get the current state tracker."""
    return _current_state_tracker


@tool
def check_migration_state() -> Dict[str, Any]:
    """
    Check current migration state to understand what has been done and what's next.

    This tool provides deterministic state information that survives context compression.
    Use this BEFORE calling any agents to avoid duplicate work.

    Returns:
        Dictionary with:
        - current_phase: What phase the migration is in
        - analysis_completed: Whether analysis phase is done
        - execution_completed: Whether execution phase is done
        - next_action: What should happen next (CALL_ANALYSIS_EXPERT, CALL_EXECUTION_EXPERT, etc.)
        - phases_completed: List of completed phases
        - recent_history: Recent agent calls
        - duplicate_calls_prevented: How many duplicate calls were blocked
    """
    tracker = get_state_tracker()

    if tracker is None:
        log_agent("[STATE_TOOL] No state tracker available - returning empty state", "WARNING")
        return {
            "error": "State tracker not initialized",
            "current_phase": "UNKNOWN",
            "next_action": "CALL_ANALYSIS_EXPERT"
        }

    summary = tracker.get_state_summary()

    log_agent(f"[STATE_TOOL] Current phase: {summary['current_phase']}")
    log_agent(f"[STATE_TOOL] Next action: {summary['next_action']}")
    log_agent(f"[STATE_TOOL] Phases completed: {summary['phases_completed']}")

    return summary


@tool
def can_call_analysis_expert() -> Dict[str, Any]:
    """
    Check if analysis_expert can be called.

    Use this before calling analysis_expert to prevent duplicate calls.

    Returns:
        Dictionary with:
        - can_call: Boolean indicating if agent can be called
        - reason: Explanation of why or why not
        - already_completed: Whether this agent already finished
    """
    tracker = get_state_tracker()

    if tracker is None:
        return {"can_call": True, "reason": "No state tracker - allowing call", "already_completed": False}

    can_call, reason = tracker.can_call_agent("analysis_expert")

    log_agent(f"[STATE_TOOL] Can call analysis_expert? {can_call}")
    log_agent(f"[STATE_TOOL] Reason: {reason}")

    return {
        "can_call": can_call,
        "reason": reason,
        "already_completed": tracker.state.get("analysis_expert_completed", False)
    }


@tool
def can_call_execution_expert() -> Dict[str, Any]:
    """
    Check if execution_expert can be called.

    Use this before calling execution_expert to prevent duplicate calls.

    Returns:
        Dictionary with:
        - can_call: Boolean indicating if agent can be called
        - reason: Explanation of why or why not
        - already_completed: Whether this agent already finished
    """
    tracker = get_state_tracker()

    if tracker is None:
        return {"can_call": True, "reason": "No state tracker - allowing call", "already_completed": False}

    can_call, reason = tracker.can_call_agent("execution_expert")

    log_agent(f"[STATE_TOOL] Can call execution_expert? {can_call}")
    log_agent(f"[STATE_TOOL] Reason: {reason}")

    return {
        "can_call": can_call,
        "reason": reason,
        "already_completed": tracker.state.get("execution_expert_completed", False)
    }


@tool
def can_call_error_expert() -> Dict[str, Any]:
    """
    Check if error_expert can be called.

    error_expert can always be called multiple times for error recovery.

    Returns:
        Dictionary with:
        - can_call: Always True (error_expert can be called multiple times)
        - reason: Explanation
        - call_count: How many times error_expert has been called
    """
    tracker = get_state_tracker()

    if tracker is None:
        return {"can_call": True, "reason": "No state tracker - allowing call", "call_count": 0}

    can_call, reason = tracker.can_call_agent("error_expert")
    call_count = tracker.state.get("error_expert_call_count", 0)

    log_agent(f"[STATE_TOOL] Can call error_expert? {can_call}")
    log_agent(f"[STATE_TOOL] error_expert has been called {call_count} times")

    return {
        "can_call": can_call,
        "reason": reason,
        "call_count": call_count
    }