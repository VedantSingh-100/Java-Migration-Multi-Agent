"""
Guarded Handoff Tools

These tools wrap the standard LangGraph handoff mechanism with state checks
that can BLOCK duplicate agent calls at the tool execution layer.

This prevents amnesia loops by intercepting at the point where the supervisor
decides to transfer to an agent, BEFORE the routing actually happens.
"""

from typing import Annotated
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION
from src.utils.logging_config import log_agent, log_summary
from src.tools.state_management import get_state_tracker


def create_guarded_handoff_tool(
    agent_name: str,
    display_name: str,
    description: str
) -> BaseTool:
    """
    Create a handoff tool that checks state BEFORE routing.

    If the target agent has already completed, the tool returns an error message
    instead of routing, effectively BLOCKING the duplicate call.

    Args:
        agent_name: Internal name of the agent (e.g., "analysis_expert")
        display_name: Human-readable name (e.g., "Analysis Expert")
        description: Tool description for the LLM

    Returns:
        A LangChain tool that can block duplicate calls
    """

    @tool(f"transfer_to_{agent_name}", description=description)
    def guarded_transfer(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """
        Transfer control to the agent, but only if state allows it.
        """
        # Get the state tracker
        tracker = get_state_tracker()

        # Check if we can call this agent
        can_call = True
        reason = "State tracker not available"

        if tracker:
            can_call, reason = tracker.can_call_agent(agent_name)

        # If we can't call the agent, BLOCK the transfer
        if not can_call:
            log_agent(f"[GUARD] ❌ BLOCKED transfer to {agent_name}")
            log_agent(f"[GUARD] Reason: {reason}")

            # Return an error message instead of routing
            # This stays in the supervisor's context
            error_msg = (
                f"❌ Cannot transfer to {display_name}: {reason}\n\n"
                f"The {display_name} has already completed its work. "
                f"Check the state using check_migration_state tool or proceed to the next phase."
            )

            # CRITICAL: Append to existing messages, don't replace
            messages = state["messages"]
            tool_message = ToolMessage(
                content=error_msg,
                tool_call_id=tool_call_id,
                name=f"transfer_to_{agent_name}"
            )

            return Command(
                # Stay in current agent (supervisor)
                update={
                    "messages": messages + [tool_message]
                }
            )

        # Agent can be called - allow the transfer
        log_agent(f"[GUARD] ✅ ALLOWED transfer to {agent_name}")

        # Update state tracking
        if tracker:
            tracker.mark_agent_called(agent_name, "Transferred via guarded handoff")

            # AUTO-COMPLETE LOGIC: When execution_expert is called, analysis_expert is done
            if agent_name == "execution_expert" and not tracker.state.get("analysis_expert_completed", False):
                tracker.mark_agent_completed("analysis_expert", "Analysis phase complete - execution started")
                log_agent(f"[GUARD] ✨ AUTO-COMPLETED analysis_expert (execution starting)")
                log_summary("ANALYSIS PHASE: AUTO-COMPLETED (execution starting)")

        # Update agent call count in state
        current_count = state.get("agent_call_count", {})
        current_count[agent_name] = current_count.get(agent_name, 0) + 1

        # Update phase in state
        phase_map = {
            "analysis_expert": "ANALYSIS",
            "execution_expert": "EXECUTION",
            "error_expert": "ERROR_RECOVERY"
        }
        new_phase = phase_map.get(agent_name, state.get("current_phase", "INIT"))

        # CRITICAL: Append to existing messages, don't replace
        messages = state["messages"]
        tool_message = ToolMessage(
            content=f"✅ Transferring control to {display_name}",
            tool_call_id=tool_call_id,
            name=f"transfer_to_{agent_name}"
        )

        # Route to the agent with state updates
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={
                "messages": messages + [tool_message],
                "agent_call_count": current_count,
                "current_phase": new_phase
            }
        )

    # CRITICAL: Set the metadata that LangGraph uses to recognize this as a handoff tool
    # This tells LangGraph this tool routes to the specified agent
    guarded_transfer.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}

    return guarded_transfer


# COMMENTED OUT: Completion tools causing API error (tool_use_id mismatch)
# These may be needed later if state tracking doesn't work reliably
# Current approach: State tracker in supervisor_orchestrator.py marks completion
# when agent returns to supervisor (lines ~928-930)

def create_completion_handoff_tool(agent_name: str) -> BaseTool:
    """
    [COMMENTED OUT - PRESERVED FOR FUTURE USE]

    Create a tool for agents to signal completion back to supervisor.

    This updates the state to mark the agent as complete, preventing
    future duplicate calls.

    Args:
        agent_name: Name of the agent that is completing

    Returns:
        A LangChain tool for transferring back with completion marker
    """

    @tool(
        f"transfer_back_to_supervisor_from_{agent_name}",
        description=f"Transfer control back to supervisor after {agent_name} completes its work"
    )
    def completion_transfer(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        summary: str = ""
    ) -> Command:
        """
        Transfer back to supervisor and mark this agent as complete.

        Args:
            summary: Brief summary of what was accomplished
        """
        # Get the state tracker
        tracker = get_state_tracker()

        # Mark completion
        if tracker:
            tracker.mark_agent_completed(agent_name, summary or "Work completed")

        log_agent(f"[GUARD] {agent_name} marked as COMPLETE")
        log_agent(f"[GUARD] Summary: {summary}")

        # Update completion flags in state
        completion_updates = {
            "messages": [
                ToolMessage(
                    content=f"✅ {agent_name.replace('_', ' ').title()} completed. {summary}",
                    tool_call_id=tool_call_id,
                    name=f"transfer_back_to_supervisor_from_{agent_name}"
                )
            ]
        }

        # Set the appropriate completion flag
        if agent_name == "analysis_expert":
            completion_updates["analysis_complete"] = True
            completion_updates["current_phase"] = "ANALYSIS_COMPLETE"
        elif agent_name == "execution_expert":
            completion_updates["execution_complete"] = True
            completion_updates["current_phase"] = "EXECUTION_COMPLETE"
        elif agent_name == "error_expert":
            completion_updates["error_recovery_complete"] = True
            completion_updates["current_phase"] = "ERROR_RECOVERY_COMPLETE"

        # Route back to supervisor
        return Command(
            goto="supervisor",
            graph=Command.PARENT,
            update=completion_updates
        )

    return completion_transfer


# Create the guarded handoff tools
guarded_analysis_handoff = create_guarded_handoff_tool(
    agent_name="analysis_expert",
    display_name="Analysis Expert",
    description=(
        "Transfer control to the Analysis Expert agent to analyze the Java project "
        "and create a migration plan. Use this ONLY ONCE at the start of migration. "
        "Check state first with check_migration_state tool."
    )
)

guarded_execution_handoff = create_guarded_handoff_tool(
    agent_name="execution_expert",
    display_name="Execution Expert",
    description=(
        "Transfer control to the Execution Expert agent to execute the migration plan. "
        "Use this ONLY AFTER analysis is complete. Check state first with check_migration_state tool."
    )
)

guarded_error_handoff = create_guarded_handoff_tool(
    agent_name="error_expert",
    display_name="Error Recovery Expert",
    description=(
        "Transfer control to the Error Expert agent to fix compilation or test errors. "
        "Can be called multiple times for error recovery. Check state first."
    )
)

# COMMENTED OUT: Completion handoff tools
# These are causing API errors (tool_use_id mismatch with Vertex AI)
# The state tracker in supervisor_orchestrator.py already marks completion
# when agents return to supervisor, so these aren't strictly necessary
# Uncomment if we find state tracking isn't reliable enough

# analysis_completion_tool = create_completion_handoff_tool("analysis_expert")
# execution_completion_tool = create_completion_handoff_tool("execution_expert")
# error_completion_tool = create_completion_handoff_tool("error_expert")