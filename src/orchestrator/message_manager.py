"""
Message Management for Migration Orchestrator

This module handles:
- Message pruning/trimming for context management
- External memory block construction
- Prompt building with injected memory

CRITICAL DESIGN NOTE:
=====================
The LangGraph State uses `add_messages` reducer which ACCUMULATES messages
across all node invocations. This module handles trimming for what the LLM
SEES, but does NOT affect what the STATE accumulates.

When a node wrapper modifies messages:
1. The modified messages are passed to agent.invoke() - agent sees trimmed context
2. The agent returns results with its generated messages
3. LangGraph's add_messages reducer ADDS these to the ORIGINAL state
4. The accumulated state grows: original + returned

To truly "reset" messages at phase transition, you must either:
- Use a custom reducer that replaces instead of adds
- Return messages with matching IDs to deduplicate
- Use a different state management approach
"""

from datetime import datetime
from typing import List, Callable, Any
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage, BaseMessage

from src.utils.logging_config import log_agent
from .constants import MAX_HISTORY_MESSAGES, MAX_SUMMARY_LENGTH


class MessagePruner:
    """
    Handles message pruning/trimming for LLM context management.

    Uses Anthropic-style "tool result clearing" approach:
    1. Take last N messages
    2. Remove orphaned ToolMessages at start
    3. Truncate old tool results (keep structure, reduce content)
    4. Keep recent messages fully intact
    """

    def __init__(self, max_messages: int = MAX_HISTORY_MESSAGES, keep_full_count: int = 5):
        """
        Args:
            max_messages: Maximum messages to keep in history
            keep_full_count: Number of recent messages to keep fully intact (not truncated)
        """
        self.max_messages = max_messages
        self.keep_full_count = keep_full_count

    def prune(self, messages: List[BaseMessage], agent_name: str = "agent") -> List[BaseMessage]:
        """
        Prune messages to fit within max_messages limit.

        Args:
            messages: List of messages to prune
            agent_name: Name of agent (for logging)

        Returns:
            Pruned list of messages
        """
        if not messages:
            return []

        if len(messages) <= self.max_messages:
            log_agent(f"[PRUNE] {agent_name}: Keeping all {len(messages)} messages (under limit)")
            return messages

        log_agent(f"[PRUNE] {agent_name}: Pruning {len(messages)} -> {self.max_messages} messages")

        # Log what's being removed
        removed = messages[:-self.max_messages]
        self._log_removed_messages(removed)

        # Step 1: Take last N messages
        trimmed = list(messages[-self.max_messages:])

        # Step 2: Remove orphaned ToolMessages at start
        trimmed = self._remove_orphaned_tool_messages(trimmed)

        # Step 3: Truncate old tool results
        trimmed = self._truncate_old_tool_results(trimmed)

        # Log final state
        if trimmed:
            first_type = getattr(trimmed[0], 'type', type(trimmed[0]).__name__)
            last_type = getattr(trimmed[-1], 'type', type(trimmed[-1]).__name__)
            log_agent(f"[PRUNE] {agent_name}: Result: {len(trimmed)} messages, first={first_type}, last={last_type}")

        return trimmed

    def _log_removed_messages(self, removed: List[BaseMessage]):
        """Log details about removed messages for debugging"""
        if not removed:
            return

        log_agent(f"[PRUNE_DETAIL] Removing {len(removed)} old messages:")
        for idx, msg in enumerate(removed[:3]):
            msg_type = getattr(msg, 'type', type(msg).__name__)
            tool_name = getattr(msg, 'name', 'N/A') if isinstance(msg, ToolMessage) else 'N/A'
            content = str(msg.content)[:60] if hasattr(msg, 'content') else ''
            log_agent(f"[PRUNE_DETAIL]   {idx+1}: {msg_type} | tool={tool_name} | {content}...")

        if len(removed) > 3:
            log_agent(f"[PRUNE_DETAIL]   ... and {len(removed) - 3} more")

    def _remove_orphaned_tool_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Remove ToolMessages at the start that have no matching AIMessage"""
        result = list(messages)
        orphaned_count = 0

        while result and isinstance(result[0], ToolMessage):
            tool_name = getattr(result[0], 'name', 'unknown')
            log_agent(f"[PRUNE_DETAIL] Removing orphaned ToolMessage: {tool_name}")
            result = result[1:]
            orphaned_count += 1

        if orphaned_count > 0:
            log_agent(f"[PRUNE] Removed {orphaned_count} orphaned ToolMessages")

        return result

    def _truncate_old_tool_results(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Truncate content of old ToolMessages (keep recent ones intact)"""
        if len(messages) <= self.keep_full_count:
            return messages

        result = list(messages)
        cleared_count = 0
        max_tool_content = 200

        # Only truncate messages NOT in the last keep_full_count
        for i in range(len(result) - self.keep_full_count):
            msg = result[i]
            if isinstance(msg, ToolMessage):
                content = msg.content if msg.content else ""
                if len(content) > max_tool_content:
                    truncated = content[:max_tool_content]
                    truncated += f"\n... [CLEARED: {len(content) - max_tool_content} chars removed]"
                    result[i] = ToolMessage(
                        content=truncated,
                        tool_call_id=msg.tool_call_id,
                        name=msg.name if hasattr(msg, 'name') else None
                    )
                    cleared_count += 1

        if cleared_count > 0:
            log_agent(f"[PRUNE] Truncated {cleared_count} old tool results")

        return result


class ExternalMemoryBuilder:
    """
    Builds external memory blocks for injection into agent prompts.

    External memory contains:
    - COMPLETED_ACTIONS: What's already done (DO NOT REPEAT)
    - VISIBLE_TASKS: Current task + upcoming (limited view)
    - CURRENT_STATE: Migration status
    """

    def __init__(self, state_file_manager, task_manager):
        """
        Args:
            state_file_manager: StateFileManager instance for reading files
            task_manager: TaskManager instance for getting visible tasks
        """
        self.state_file_manager = state_file_manager
        self.task_manager = task_manager

    def build(self) -> str:
        """
        Build the external memory block.

        Returns:
            Formatted string for injection into system prompt, or empty string if no files exist
        """
        # Read state files
        completed = self.state_file_manager.read_file("COMPLETED_ACTIONS.md")
        todo = self.state_file_manager.read_file("TODO.md", keep_beginning=True)
        current_state = self.state_file_manager.read_file("CURRENT_STATE.md")

        # If no files exist yet, return empty
        if not completed and not todo and not current_state:
            return ""

        # Get visible tasks (restricted view)
        visible_tasks = self.task_manager.get_visible_tasks(todo, max_visible=3)

        # Log visibility for debugging
        self._log_task_visibility(visible_tasks)

        # Build the memory block
        timestamp = datetime.now().strftime("%H:%M:%S")
        tasks_section = self._format_tasks_section(visible_tasks)

        return self._format_memory_block(timestamp, completed, tasks_section, current_state)

    def _log_task_visibility(self, visible_tasks: dict):
        """Log what tasks the agent can see"""
        # Handle file_missing error state FIRST
        if visible_tasks.get('file_missing'):
            log_agent(f"[TASK_VISIBILITY] ⚠️ ERROR: TODO.md is missing or empty - this is NOT completion!")
            log_agent(f"[TASK_VISIBILITY] State files may have been lost (git stash, branch switch, etc.)")
            return

        if visible_tasks['all_done']:
            log_agent(f"[TASK_VISIBILITY] ✅ ALL TASKS COMPLETE")
        elif visible_tasks['current']:
            log_agent(f"[TASK_VISIBILITY] 👁 Current: {visible_tasks['current'][:60]}...")
            hidden = visible_tasks['remaining_count'] - len(visible_tasks['upcoming']) - 1
            if hidden > 0:
                log_agent(f"[TASK_VISIBILITY] 🔒 {hidden} tasks hidden from agent")
            log_agent(f"[TASK_VISIBILITY] Progress: {visible_tasks['completed_count']}/{visible_tasks['total_count']}")

    def _format_tasks_section(self, visible_tasks: dict) -> str:
        """Format the tasks section of the memory block"""
        # Handle file_missing error state FIRST
        if visible_tasks.get('file_missing'):
            return """
⚠️ ERROR: TODO.md FILE MISSING OR EMPTY ⚠️

The TODO.md file could not be found or is empty.
This is an ERROR state - NOT completion!

POSSIBLE CAUSES:
- State files were lost during branch switch
- Git stash removed working directory files
- Files were not properly initialized

DO NOT PROCEED - State must be recovered first.
Check: git stash list
If stashed: git stash pop
"""

        if visible_tasks['all_done']:
            return """
✅ ALL TASKS COMPLETE! ✅

All TODO items are marked [x]. Migration work is done.
The system will automatically detect completion.
"""

        if not visible_tasks['current']:
            return "(No TODO file yet - wait for analysis phase)"

        section = f"""
✔ CURRENT TASK (DO THIS NOW):
════════════════════════════════════════════════════════════════════════════════

{visible_tasks['current']}

════════════════════════════════════════════════════════════════════════════════
"""

        if visible_tasks['upcoming']:
            section += "\n▪ UPCOMING TASKS (for context only - DO NOT start these yet):\n"
            section += "────────────────────────────────────────────────────────────────\n\n"
            for idx, task in enumerate(visible_tasks['upcoming'], 1):
                section += f"  {idx}. {task}\n"
            section += "\n⚠ Complete CURRENT TASK before attempting these!\n"
            section += "────────────────────────────────────────────────────────────────"

        section += f"""

▪ PROGRESS: {visible_tasks['completed_count']}/{visible_tasks['total_count']} tasks complete ({visible_tasks['remaining_count']} remaining)
"""
        return section

    def _format_memory_block(self, timestamp: str, completed: str, tasks_section: str, current_state: str) -> str:
        """Format the complete memory block"""
        return f"""
════════════════════════════════════════════════════════════════════════════════
EXTERNAL MEMORY (Updated: {timestamp}) - READ THIS BEFORE EVERY ACTION
════════════════════════════════════════════════════════════════════════════════

⚠ CRITICAL RULES - READ BEFORE EVERY ACTION ⚠

✔ YOUR WORKFLOW:
1. Look at CURRENT TASK below (this is the ONLY task you should do now)
2. Execute that ONE task using appropriate tools
3. Mark it complete in TODO.md by changing [ ] to [x]
4. Commit your changes
5. The system will show you the NEXT task

🚫 NEVER:
- Work on UPCOMING TASKS (those are shown for context only)
- Repeat any action in COMPLETED ACTIONS below
- Try to modify COMPLETED_ACTIONS.md (it's system-managed)
- Skip the current task to do something else

▪ MIGRATION COMPLETE WHEN:
- All tasks show [x] in TODO.md
- System will automatically detect completion

════════════════════════════════════════════════════════════════════════════════
▌ COMPLETED ACTIONS (DO NOT REPEAT THESE)
════════════════════════════════════════════════════════════════════════════════

{completed if completed else "(No actions completed yet)"}

════════════════════════════════════════════════════════════════════════════════
▌ YOUR TASKS (Only showing next 3 - others hidden until these complete)
════════════════════════════════════════════════════════════════════════════════

{tasks_section}

════════════════════════════════════════════════════════════════════════════════
▌ CURRENT STATE (Migration Status)
════════════════════════════════════════════════════════════════════════════════

{current_state if current_state else "(No state file yet)"}

════════════════════════════════════════════════════════════════════════════════
""".strip()


class PromptBuilder:
    """
    Builds prompts for agents with message pruning and external memory injection.

    Creates a callable that can be used as the `prompt` parameter for create_react_agent.
    """

    def __init__(self,
                 system_prompt: str,
                 agent_name: str,
                 message_pruner: MessagePruner = None,
                 external_memory_builder: ExternalMemoryBuilder = None,
                 inject_external_memory: bool = False):
        """
        Args:
            system_prompt: Base system prompt for the agent
            agent_name: Name of agent (for logging)
            message_pruner: MessagePruner instance (uses default if None)
            external_memory_builder: ExternalMemoryBuilder instance (required if inject_external_memory=True)
            inject_external_memory: Whether to inject external memory into prompts
        """
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self.message_pruner = message_pruner or MessagePruner()
        self.external_memory_builder = external_memory_builder
        self.inject_external_memory = inject_external_memory

    def build(self) -> Callable[[dict], List[BaseMessage]]:
        """
        Build and return the prompt function.

        Returns:
            Callable that takes state dict and returns list of messages
        """
        def prompt_fn(state: dict) -> List[BaseMessage]:
            messages = state.get("messages", [])

            log_agent(f"[PROMPT] {self.agent_name}: Building prompt with {len(messages)} input messages")

            # Start with system prompt
            result = [SystemMessage(content=self.system_prompt)]

            # Inject external memory at position [1]
            if self.inject_external_memory and self.external_memory_builder:
                memory_block = self.external_memory_builder.build()
                if memory_block:
                    result.append(SystemMessage(content=memory_block))
                    log_agent(f"[PROMPT] {self.agent_name}: Injected external memory at position [1]")

            # Add pruned messages
            if messages:
                pruned = self.message_pruner.prune(messages, self.agent_name)
                result.extend(pruned)
                log_agent(f"[PROMPT] {self.agent_name}: Final prompt has {len(result)} messages")
            else:
                log_agent(f"[PROMPT] {self.agent_name}: No history messages, using system prompt only")

            return result

        return prompt_fn


def create_clean_execution_context(project_path: str) -> List[BaseMessage]:
    """
    Create a clean message context for execution phase.

    This is used at the analysis -> execution transition to give the
    execution agent a fresh start without analysis history.

    IMPORTANT: This creates the messages for the agent invocation,
    but the LangGraph STATE may still accumulate via add_messages reducer.

    Args:
        project_path: Path to the project being migrated

    Returns:
        List containing a single HumanMessage with execution instructions
    """
    return [
        HumanMessage(content=f"""EXECUTION PHASE START - Project: {project_path}

Analysis is complete. Your task list is in VISIBLE_TASKS.md.

WORKFLOW:
1. Read VISIBLE_TASKS.md to see the current task
2. Execute that task
3. Commit with git_commit
4. Task list auto-updates after commit
5. Repeat

Start now: read VISIBLE_TASKS.md""")
    ]


def compile_execution_context(
    project_path: str,
    loop_num: int,
    current_task: str,
    completed_summary: str,
    last_result: str = None
) -> List[BaseMessage]:
    """
    Compile fresh context for each execution loop.

    This replaces message accumulation with a compiled view.
    The agent gets exactly what it needs each call - no history bloat.

    Args:
        project_path: Path to project
        loop_num: Current execution loop number
        current_task: Current task from VISIBLE_TASKS.md
        completed_summary: Summary of completed tasks (last 10)
        last_result: Previous tool output (truncated) for continuity

    Returns:
        List with single HumanMessage containing compiled context
    """
    last_result_section = last_result if last_result else "This is your first action in this session"

    context = f"""EXECUTION LOOP #{loop_num} - Project: {project_path}

════════════════════════════════════════════════════════════════
CURRENT TASK (DO THIS NOW):
════════════════════════════════════════════════════════════════
{current_task or "Read VISIBLE_TASKS.md to see your current task"}

════════════════════════════════════════════════════════════════
COMPLETED TASKS (DO NOT REPEAT THESE):
════════════════════════════════════════════════════════════════
{completed_summary or "No tasks completed yet"}

════════════════════════════════════════════════════════════════
LAST ACTION RESULT:
════════════════════════════════════════════════════════════════
{last_result_section}

════════════════════════════════════════════════════════════════
YOUR WORKFLOW:
════════════════════════════════════════════════════════════════
1. Execute the CURRENT TASK above using your tools
2. Verify success with mvn_compile
3. Commit with commit_changes (this auto-advances to next task)
4. System will compile fresh context with your next task

DO NOT read TODO.md - use only VISIBLE_TASKS.md if you need task details.
The task list auto-updates after each commit."""

    log_agent(f"[COMPILE] Loop #{loop_num}: Compiled fresh context (~{len(context)} chars)")
    return [HumanMessage(content=context)]


def extract_last_tool_result(messages: List[BaseMessage], max_chars: int = 500) -> str:
    """
    Extract the most recent tool result for continuity.

    Args:
        messages: List of messages from state
        max_chars: Maximum characters to include

    Returns:
        Truncated tool result string, or None if no tool messages
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = str(msg.content)[:max_chars]
            tool_name = getattr(msg, 'name', 'tool')
            if len(str(msg.content)) > max_chars:
                content += "... [truncated]"
            return f"[{tool_name}]: {content}"
    return None


def summarize_completed_tasks(completed_content: str, max_items: int = 10) -> str:
    """
    Extract task completion summary from COMPLETED_ACTIONS.md.

    Only extracts the TASK COMPLETIONS section, not full action log.

    Args:
        completed_content: Full content of COMPLETED_ACTIONS.md
        max_items: Maximum number of recent completions to show

    Returns:
        Summary string of recent task completions
    """
    if not completed_content:
        return "No tasks completed yet"

    lines = []
    in_completions = False

    for line in completed_content.split('\n'):
        if '=== TASK COMPLETIONS ===' in line or 'TASK COMPLETIONS' in line:
            in_completions = True
            continue
        if '=== ACTION LOG ===' in line or 'ACTION LOG' in line:
            break
        if in_completions and line.strip() and not line.startswith('#'):
            # Clean up the line
            clean_line = line.strip()
            if clean_line.startswith('-') or clean_line.startswith('*') or clean_line[0].isdigit():
                lines.append(clean_line)

    if lines:
        # Return last N items
        recent = lines[-max_items:]
        return '\n'.join(recent)

    # Fallback: look for any checkbox items
    for line in completed_content.split('\n'):
        if '- [x]' in line.lower():
            task = line.split(']', 1)[-1].strip()[:60]
            lines.append(f"- {task}")

    if lines:
        return '\n'.join(lines[-max_items:])

    return "No tasks completed yet"
