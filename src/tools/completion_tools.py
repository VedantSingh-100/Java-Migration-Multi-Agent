"""
Completion Tools - Allow agents to signal when they're done

This module provides tools for agents to explicitly mark their work as complete,
enabling deterministic state transitions and preventing amnesia loops.
"""

import os
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

    This function performs a FINAL test invariance check using MigrationBench's
    exact evaluation logic before allowing completion. This guarantees that the
    evaluation will pass.

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

    # STEP 1: Run final test invariance check using MigrationBench
    # This uses the EXACT same function as the evaluation script
    base_commit = os.environ.get('MIGRATION_BASE_COMMIT', '')
    repo_path = os.environ.get('MIGRATION_REPO_PATH', '')

    if base_commit and repo_path:
        try:
            from src.utils.test_verifier import verify_final_test_invariance

            log_agent(f"[COMPLETION] Running final test invariance check against {base_commit}")
            is_valid, msg = verify_final_test_invariance(repo_path, base_commit)

            if not is_valid:
                log_agent("[COMPLETION] ❌ Final test invariance check FAILED", "ERROR")
                log_summary("EXECUTION BLOCKED: Test invariance check failed")
                return {
                    "success": False,
                    "error": "TEST_INVARIANCE_FAILED",
                    "message": f"""EXECUTION CANNOT BE MARKED COMPLETE

{msg}

Your changes modified test methods, which violates migration rules.
The evaluation will FAIL if you proceed.

🔧 TO FIX THIS:

1. Call revert_test_files(repo_path="{repo_path}") to undo test changes
2. Run mvn compile to see what's actually failing
3. Fix the APPLICATION code (not tests!) to resolve errors
4. If a test truly cannot work, add @Disabled("reason") annotation

RULES:
- DO NOT rename test methods
- DO NOT delete test methods
- DO NOT add new test methods
- Only update test IMPLEMENTATION code (inside methods)

After fixing, call mark_execution_complete() again.""",
                    "agent": "execution_expert",
                    "status": "BLOCKED"
                }

            log_agent("[COMPLETION] ✅ Final test invariance check PASSED")

        except Exception as e:
            log_agent(f"[COMPLETION] Test invariance check error: {e}", "WARNING")
            # Continue anyway if check fails due to import/other errors
            # The regular TestMethodVerifier should have caught issues earlier

    # STEP 2: Mark execution expert as completed
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