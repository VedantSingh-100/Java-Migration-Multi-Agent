"""
Migration State Tracker - Deterministic State Management

This module provides a code-based state management system that lives OUTSIDE
the conversation history, preventing amnesia loops caused by context compression.

Instead of relying on LLM prompts to read files and remember what happened,
this tracker maintains authoritative state about:
- Which agents have been called
- Which phases are complete
- What the next action should be

This is the SOURCE OF TRUTH for migration progress.
"""

from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from src.utils.logging_config import log_agent


class MigrationStateTracker:
    """
    Deterministic state management for multi-agent migrations.

    Prevents amnesia loops by maintaining state in code rather than relying
    on LLM memory or file reads.
    """

    def __init__(self, repo_path: str):
        """
        Initialize state tracker for a migration.

        Args:
            repo_path: Path to the repository being migrated
        """
        self.repo_path = repo_path
        self.start_time = datetime.now()

        # Core state flags
        self.state = {
            # Agent call tracking
            "analysis_expert_called": False,
            "analysis_expert_completed": False,
            "execution_expert_called": False,
            "execution_expert_completed": False,
            "error_expert_call_count": 0,

            # Phase tracking
            "current_phase": "INIT",  # INIT -> ANALYSIS -> EXECUTION -> ERROR_RECOVERY -> COMPLETE
            "phases_completed": [],

            # History (for debugging and validation)
            "agent_call_history": [],  # [(timestamp, agent_name, action, details)]

            # Blocking/prevention counters
            "duplicate_calls_prevented": 0,
            "amnesia_loops_prevented": 0,
        }

        log_agent(f"[STATE_TRACKER] Initialized for {repo_path}")
        log_agent(f"[STATE_TRACKER] Initial phase: {self.state['current_phase']}")

    def mark_agent_called(self, agent_name: str, details: Optional[str] = None):
        """
        Mark that an agent was called.

        Args:
            agent_name: Name of the agent (analysis_expert, execution_expert, error_expert)
            details: Optional details about why the agent was called
        """
        timestamp = datetime.now()

        # Update state flags
        if agent_name == "error_expert":
            self.state["error_expert_call_count"] += 1
        else:
            self.state[f"{agent_name}_called"] = True

        # Record in history
        self.state["agent_call_history"].append(
            (timestamp, agent_name, "CALLED", details or "")
        )

        # Update phase
        if agent_name == "analysis_expert":
            self.state["current_phase"] = "ANALYSIS"
        elif agent_name == "execution_expert":
            self.state["current_phase"] = "EXECUTION"
        elif agent_name == "error_expert":
            self.state["current_phase"] = "ERROR_RECOVERY"

        log_agent(f"[STATE_TRACKER] {agent_name} CALLED (phase: {self.state['current_phase']})")
        if details:
            log_agent(f"[STATE_TRACKER] Details: {details}")

    def mark_agent_completed(self, agent_name: str, details: Optional[str] = None):
        """
        Mark that an agent finished its work successfully.

        Args:
            agent_name: Name of the agent
            details: Optional details about what the agent accomplished
        """
        timestamp = datetime.now()

        # Update completion flag
        if agent_name != "error_expert":
            self.state[f"{agent_name}_completed"] = True

            # Mark phase as complete
            if agent_name == "analysis_expert":
                phase_name = "ANALYSIS"
            elif agent_name == "execution_expert":
                phase_name = "EXECUTION"
            else:
                phase_name = agent_name.upper()

            if phase_name not in self.state["phases_completed"]:
                self.state["phases_completed"].append(phase_name)

        # Record in history
        self.state["agent_call_history"].append(
            (timestamp, agent_name, "COMPLETED", details or "")
        )

        log_agent(f"[STATE_TRACKER] {agent_name} COMPLETED")
        log_agent(f"[STATE_TRACKER] Phases complete: {self.state['phases_completed']}")
        if details:
            log_agent(f"[STATE_TRACKER] Details: {details}")

    def can_call_agent(self, agent_name: str) -> Tuple[bool, str]:
        """
        Check if an agent can be called (GATE function).

        This prevents duplicate calls to agents that have already completed.

        Args:
            agent_name: Name of the agent to check

        Returns:
            (can_call: bool, reason: str) - Whether agent can be called and why/why not
        """
        # error_expert can always be called multiple times
        if agent_name == "error_expert":
            return (True, "error_expert can be called multiple times for error recovery")

        # Check if agent already completed
        if self.state.get(f"{agent_name}_completed", False):
            self.state["duplicate_calls_prevented"] += 1
            self.state["amnesia_loops_prevented"] += 1

            reason = f"{agent_name} has already completed its work. Phase '{agent_name.replace('_expert', '').upper()}' is done."
            log_agent(f"[STATE_TRACKER] BLOCKED duplicate call to {agent_name}")
            log_agent(f"[STATE_TRACKER] Reason: {reason}")
            log_agent(f"[STATE_TRACKER] Amnesia loops prevented: {self.state['amnesia_loops_prevented']}")

            return (False, reason)

        # Agent can be called
        return (True, f"{agent_name} has not been called yet or is in progress")

    def get_next_action(self) -> str:
        """
        Deterministic decision: what should happen next?

        Returns:
            Action string (CALL_ANALYSIS_EXPERT, CALL_EXECUTION_EXPERT, CALL_ERROR_EXPERT, or COMPLETE)
        """
        # Check phase progression
        if not self.state["analysis_expert_completed"]:
            return "CALL_ANALYSIS_EXPERT"
        elif not self.state["execution_expert_completed"]:
            return "CALL_EXECUTION_EXPERT"
        else:
            return "COMPLETE"

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current state for logging or tool responses.

        Returns:
            Dictionary with current state information
        """
        duration = (datetime.now() - self.start_time).total_seconds()

        return {
            "repo_path": self.repo_path,
            "duration_seconds": duration,
            "current_phase": self.state["current_phase"],
            "phases_completed": self.state["phases_completed"],
            "analysis_completed": self.state["analysis_expert_completed"],
            "execution_completed": self.state["execution_expert_completed"],
            "error_expert_calls": self.state["error_expert_call_count"],
            "next_action": self.get_next_action(),
            "duplicate_calls_prevented": self.state["duplicate_calls_prevented"],
            "amnesia_loops_prevented": self.state["amnesia_loops_prevented"],
            "total_agent_calls": len([h for h in self.state["agent_call_history"] if h[2] == "CALLED"]),
            "recent_history": self._format_recent_history(5),
        }

    def _format_recent_history(self, count: int = 5) -> List[str]:
        """Format recent history entries for display."""
        recent = self.state["agent_call_history"][-count:]
        formatted = []

        for timestamp, agent, action, details in recent:
            time_str = timestamp.strftime("%H:%M:%S")
            entry = f"{time_str} | {agent} {action}"
            if details:
                entry += f" - {details}"
            formatted.append(entry)

        return formatted

    def log_full_state(self):
        """Log complete state information for debugging."""
        log_agent("=" * 60)
        log_agent("[STATE_TRACKER] FULL STATE DUMP")
        log_agent("=" * 60)

        summary = self.get_state_summary()
        for key, value in summary.items():
            if key == "recent_history":
                log_agent(f"Recent History:")
                for entry in value:
                    log_agent(f"  {entry}")
            else:
                log_agent(f"{key}: {value}")

        log_agent("=" * 60)