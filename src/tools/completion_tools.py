"""
Completion Tools - Allow agents to signal when they're done

This module provides tools for agents to explicitly mark their work as complete,
enabling deterministic state transitions and preventing amnesia loops.
"""

from typing import Annotated, Dict, Any
from langchain_core.tools import tool
from src.utils.logging_config import log_agent, log_summary
from src.tools.state_management import get_state_tracker


@tool
def mark_execution_complete(summary: str = "") -> Dict[str, Any]:
    """
    Mark execution phase as complete.

    CRITICAL: Only call this when ALL TODO items in TODO.md are checked off (☑)
    AND the build passes (mvn compile succeeds) AND tests pass (mvn test succeeds).

    This prevents the supervisor from calling you again, so make sure you're truly done.

    Args:
        summary: Brief summary of what was accomplished during execution

    Returns:
        Dictionary confirming completion was marked
    """
    tracker = get_state_tracker()

    if tracker is None:
        log_agent("[COMPLETION] No state tracker available", "WARNING")
        return {
            "success": False,
            "error": "State tracker not initialized"
        }

    # Mark execution expert as completed
    tracker.mark_agent_completed("execution_expert", summary or "Execution phase completed")

    log_agent("[COMPLETION] ✅ Execution phase marked COMPLETE")
    log_summary("EXECUTION PHASE: COMPLETE")
    if summary:
        log_agent(f"[COMPLETION] Summary: {summary}")
        log_summary(f"Execution Summary: {summary}")

    return {
        "success": True,
        "agent": "execution_expert",
        "status": "COMPLETED",
        "message": "Execution phase marked as complete. Supervisor will not call execution_expert again.",
        "summary": summary
    }


@tool
def mark_analysis_complete(summary: str = "") -> Dict[str, Any]:
    """
    Mark analysis phase as complete.

    Call this when you've finished creating the migration plan and documented everything.

    Args:
        summary: Brief summary of what was analyzed

    Returns:
        Dictionary confirming completion was marked
    """
    tracker = get_state_tracker()

    if tracker is None:
        log_agent("[COMPLETION] No state tracker available", "WARNING")
        return {
            "success": False,
            "error": "State tracker not initialized"
        }

    # Mark analysis expert as completed
    tracker.mark_agent_completed("analysis_expert", summary or "Analysis phase completed")

    log_agent("[COMPLETION] ✅ Analysis phase marked COMPLETE")
    log_summary("ANALYSIS PHASE: COMPLETE")
    if summary:
        log_agent(f"[COMPLETION] Summary: {summary}")
        log_summary(f"Analysis Summary: {summary}")

    return {
        "success": True,
        "agent": "analysis_expert",
        "status": "COMPLETED",
        "message": "Analysis phase marked as complete. Supervisor will not call analysis_expert again.",
        "summary": summary
    }


# Export tools
completion_tools = [
    mark_execution_complete,
    mark_analysis_complete
]